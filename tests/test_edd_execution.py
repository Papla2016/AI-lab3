import json
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock

import httpx
import openai
import pytest

from evals import evaluate
from evals.judge import Judge
from evals.metrics import metric_label, retrieval_scores, scores_for
from evals.reporting import compare, complete, read_report, report_csv, report_json
from evals.schemas import INJECTION_MARKER, INJECTION_SOURCE, Row, load_dataset
from src.config import Settings
import src.llm as transport
from src.llm import LLMRateLimitExceeded, OpenAICompatibleLLM
from src.router import LLMRouter
from src.schemas import INSUFFICIENT, SearchResult
from test_edd import CHUNK, EXAMPLE, PASSING_JUDGE_JSON, config
from test_pipeline import PROMPTS, VALID, FakeStore, SequenceLLM, pipeline


@pytest.mark.parametrize("top_k,hit", [(1, 0), (2, 1), (5, 1), (7, 1)])
def test_hit_at_k_uses_actual_cutoff(top_k, hit):
    unrelated = CHUNK.model_copy(update={"source": "other.txt"})
    assert retrieval_scores(EXAMPLE, [unrelated, CHUNK], top_k)["hit_at_k"] == hit
    assert metric_label("hit_at_k", top_k) == f"Hit@{top_k}"


def test_exports_and_comparison_k_labels():
    examples = load_dataset()
    evaluator = evaluate.Evaluator(
        pipeline(SequenceLLM(VALID, VALID)), examples, config(examples).model_copy(update={"top_k": 1}),
        Judge(SequenceLLM(PASSING_JUDGE_JSON, PASSING_JUDGE_JSON), PROMPTS / "judge.md"),
    )
    a, b = evaluator.run("zero_shot", limit=1), evaluator.run("few_shot", limit=1)
    assert json.loads(report_json(a))["metric_labels"]["hit_at_k"] == "Hit@1"
    assert "Hit@1" in report_csv(a).splitlines()[0]
    assert any(row["metric"] == "Hit@1" for row in compare(a, b))
    b.config.top_k = 5
    with pytest.raises(ValueError, match="Hit@1 и Hit@5"):
        compare(a, b)
    b.config.top_k = 1
    b.config.evaluation_mode = "end_to_end"
    with pytest.raises(ValueError, match="разные режимы"):
        compare(a, b)


@pytest.mark.parametrize("route", ["DOCUMENT_QUESTION", "GENERAL_OR_OUT_OF_SCOPE"])
def test_end_to_end_runs_real_router_and_uses_ask_trace(route):
    llm = SequenceLLM(json.dumps({"route": route}), VALID)
    rag = pipeline(llm, LLMRouter(llm, PROMPTS))
    rag.reranker = Mock(wraps=rag.reranker)
    runner = evaluate.Evaluator(
        rag, [EXAMPLE], config([EXAMPLE]).model_copy(update={"evaluation_mode": "end_to_end"}),
        Judge(SequenceLLM(PASSING_JUDGE_JSON), PROMPTS / "judge.md"),
    )
    row = runner.evaluate(EXAMPLE, "zero_shot")
    assert row.router == route and not row.errors
    assert "QUESTION:" in llm.prompts[0][1]
    if route == "DOCUMENT_QUESTION":
        rag.reranker.rerank.assert_called_once()
        assert row.context == row.retrieved == [CHUNK]
        assert row.metrics["citation_validity"] == 1
        assert len(llm.prompts) == 2
    else:
        rag.reranker.rerank.assert_not_called()
        assert row.answer == INSUFFICIENT and not row.retrieved
        assert row.metrics["false_refusal_rate"] == 1
        assert len(llm.prompts) == 1


def test_end_to_end_preserves_retrieval_on_generation_error():
    llm = SequenceLLM('{"route":"DOCUMENT_QUESTION"}', "{}", "{}")
    rag = pipeline(llm, LLMRouter(llm, PROMPTS))
    runner = evaluate.Evaluator(
        rag, [EXAMPLE], config([EXAMPLE]).model_copy(update={"evaluation_mode": "end_to_end"}),
        Judge(SequenceLLM(), PROMPTS / "judge.md"),
    )
    row = runner.evaluate(EXAMPLE, "zero_shot")
    assert row.router == "DOCUMENT_QUESTION"
    assert row.retrieved == row.context == [CHUNK]
    assert "generation" in row.errors and row.judge is None


def test_full_ab_cli_writes_both_100_question_reports(tmp_path, monkeypatch):
    examples = load_dataset()
    source = INJECTION_SOURCE
    response = VALID.replace("guide.txt", source)
    llm = SequenceLLM(*([response] * 200))
    scorer = Judge(SequenceLLM(*([PASSING_JUDGE_JSON] * 200)), PROMPTS / "judge.md")
    rag = pipeline(llm)
    rag.store.result = SearchResult(
        FakeStore.result.text + " " + INJECTION_MARKER, {"source": source, "chunk_id": "abc"}, 0.9,
    )
    evaluator = evaluate.Evaluator(rag, examples, config(examples), scorer)
    monkeypatch.setattr(evaluate, "build_evaluator", lambda *args, **kwargs: evaluator)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--compare", "--output-dir", str(tmp_path)])
    assert evaluate.main() == 0
    a, b = [read_report(tmp_path / name) for name in ("version_a.json", "version_b.json")]
    assert complete(a) and complete(b)
    assert len(llm.prompts) == len(scorer.llm.prompts) == 200
    assert rag.store.calls == 200
    assert all(row_a.context == row_b.context for row_a, row_b in zip(a.rows, b.rows))
    assert json.loads((tmp_path / "comparison.json").read_text())["complete_b"]
    assert (tmp_path / "comparison.csv").is_file()
    a.rows[6].errors["judge"] = "HTTP 429"
    a.rows[6].judge = None
    a.rows[6].metrics = scores_for(a.rows[6])
    path_a = tmp_path / "version_a.json"
    path_b = tmp_path / "version_b.json"
    path_a.write_text(report_json(a), encoding="utf-8")
    untouched_b = path_b.read_bytes()
    llm.responses.append(response)
    scorer.llm.responses.append(PASSING_JUDGE_JSON)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    assert evaluate.main() == 0
    resumed = read_report(path_a)
    assert complete(resumed) and len(llm.prompts) == 201
    assert resumed.rows[7:] == a.rows[7:] and path_b.read_bytes() == untouched_b
    b.config.top_k = 1
    path_b.write_text(report_json(b), encoding="utf-8")
    saved_a = path_a.read_bytes()
    previous_table = (tmp_path / "comparison.json").read_bytes()
    assert evaluate.main() == 1
    assert len(llm.prompts) == 201 and path_a.read_bytes() == saved_a
    assert (tmp_path / "comparison.json").read_bytes() == previous_table


def test_resume_retries_only_unfinished_and_preserves_later_rows(tmp_path, monkeypatch):
    examples = load_dataset()
    evaluator = evaluate.Evaluator(
        pipeline(SequenceLLM()), examples, config(examples).model_copy(update={"evaluation_mode": "retrieval_only"}),
    )
    path = tmp_path / "retrieval.json"
    attempts = []
    fail_first = True

    def evaluate_row(example, strategy):
        attempts.append(example.id)
        errors = {"retrieval": "temporary"} if fail_first and example.id == 1 else {}
        row = Row(example=example, retrieved=[CHUNK], errors=errors)
        row.metrics = scores_for(row)
        return row

    def interrupt(second, total, row):
        if second == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(evaluator, "evaluate", evaluate_row)
    with pytest.raises(KeyboardInterrupt):
        evaluator.run(limit=3, output=path, progress=interrupt)
    assert [row.example.id for row in read_report(path).rows] == [1, 2]
    previously_saved_two = read_report(path).rows[1]
    fail_first = False
    attempts.clear()

    def stop_after_retry(first, total, row):
        if first == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluator.run(limit=3, output=path, resume=True, progress=stop_after_retry)
    assert attempts == [1]
    assert read_report(path).rows[1] == previously_saved_two
    assert not read_report(path).rows[0].errors
    attempts.clear()
    result = evaluator.run(limit=3, output=path, resume=True)
    assert attempts == [3] and len(result.rows) == 3
    evaluator.config.top_k = 1
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="параметры|Параметры"):
        evaluator.run(limit=3, output=path, resume=True)
    assert path.read_bytes() == original_bytes


@pytest.fixture
def api_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(transport, "_limiters", {})
    monkeypatch.setattr(transport, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(transport, "sleep", sleep)
    return now


def mock_api(monkeypatch, handler):
    real_openai = openai.OpenAI

    def client(**kwargs):
        assert kwargs["max_retries"] == 0
        return real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "OpenAI", client)


@pytest.mark.parametrize("statuses", [(429, 503, 200), (429, 429, 429)])
def test_retries_honor_retry_after_and_exponential_backoff(monkeypatch, api_clock, statuses):
    requests = []

    def handler(request):
        status = statuses[len(requests)]
        requests.append(api_clock[0])
        if status == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": VALID}}]})
        headers = {"Retry-After": "8"} if len(requests) == 1 else {}
        return httpx.Response(status, json={"error": {"message": "RequestsPerMinute"}}, headers=headers)

    mock_api(monkeypatch, handler)
    llm = OpenAICompatibleLLM("http://example.invalid/v1", "nonsecret-test-key", "mock", 60)
    if statuses[-1] == 200:
        assert llm.generate("system", "user") == VALID
    else:
        with pytest.raises(LLMRateLimitExceeded, match="Временное ограничение"):
            llm.generate("system", "user")
    assert requests == [0, 8, 10]


def test_retry_after_accepts_http_date():
    date = datetime.now(timezone.utc) + timedelta(seconds=30)
    request = httpx.Request("POST", "http://example.invalid")
    response = httpx.Response(429, headers={"Retry-After": format_datetime(date, usegmt=True)}, request=request)
    error = openai.RateLimitError("temporary", response=response, body=None)
    assert transport._retry_after(error) == pytest.approx(30, abs=2)


def test_same_model_shares_rate_limit_across_clients_and_other_model_is_independent(monkeypatch, api_clock):
    requests = []

    def handler(request):
        requests.append(api_clock[0])
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID}}]})

    mock_api(monkeypatch, handler)
    generation = OpenAICompatibleLLM("http://gen.invalid/v1", "test-key", "mock")
    judge = OpenAICompatibleLLM("http://judge.invalid/v1", "other-test-key", "models/mock")
    other = OpenAICompatibleLLM("http://judge.invalid/v1", "other-test-key", "other-model")
    for client in (generation, other, judge, generation):
        client.generate("system", "user")
    assert requests == [0, 0, 5, 10]


def test_daily_quota_stops_without_retry(monkeypatch, api_clock):
    handler = Mock(return_value=httpx.Response(429, json={"error": {
        "message": "private provider response",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}],
    }}, headers={"Retry-After": "42"}))
    mock_api(monkeypatch, handler)
    llm = OpenAICompatibleLLM("http://example.invalid/v1", "test-key", "mock")
    with pytest.raises(LLMRateLimitExceeded, match="суточная квота") as failure:
        llm.generate("system", "user")
    assert "private provider response" not in str(failure.value)
    assert handler.call_count == 1 and api_clock[0] == 0


def test_rate_limit_stops_evaluation_and_resume_keeps_completed_rows(monkeypatch, tmp_path):
    examples = [EXAMPLE.model_copy(update={"id": n}) for n in range(1, 101)]
    limited_judge = Mock()
    limited_judge.generate.side_effect = [PASSING_JUDGE_JSON, LLMRateLimitExceeded(daily=True)]
    judge = Judge(limited_judge, PROMPTS / "judge.md")
    evaluator = evaluate.Evaluator(
        pipeline(SequenceLLM(VALID, VALID)), examples, config(examples), judge,
    )
    actual_evaluate = evaluator.evaluate
    path = tmp_path / "version_a.json"

    with pytest.raises(LLMRateLimitExceeded):
        evaluator.run("zero_shot", limit=3, output=path)
    saved = read_report(path).rows
    assert limited_judge.generate.call_count == 2
    assert [row.example.id for row in saved] == [1]

    evaluator.pipeline.llm = SequenceLLM(VALID, VALID)
    evaluator.judge.llm = SequenceLLM(PASSING_JUDGE_JSON, PASSING_JUDGE_JSON)
    run_row = Mock(side_effect=actual_evaluate)
    monkeypatch.setattr(evaluator, "evaluate", run_row)
    resumed = evaluator.run("zero_shot", limit=3, output=path, resume=True)
    assert [call.args[0].id for call in run_row.call_args_list] == [2, 3]
    assert resumed.rows[0] == saved[0]
    assert [row.example.id for row in resumed.rows if not row.errors] == [1, 2, 3]


def test_api_probe_uses_separate_judge_and_preserves_legacy_defaults(monkeypatch):
    gen_payload = VALID.replace("Используйте tool install.", "42").replace("guide.txt", "probe.txt").replace('"abc"', '"probe"')
    generation = SequenceLLM(gen_payload, PASSING_JUDGE_JSON)
    judging = SequenceLLM(PASSING_JUDGE_JSON)
    monkeypatch.setattr(evaluate.RAGLabService, "_default_llm", staticmethod(lambda settings: generation))
    constructor = Mock(return_value=judging)
    monkeypatch.setattr(evaluate, "OpenAICompatibleLLM", constructor)
    settings = Settings(
        llm_base_url="http://gen.invalid/v1", llm_model="generation",
        judge_base_url="http://judge.invalid/v1", judge_api_key="test-judge-key", judge_model="scoring",
    )
    evaluate.check_api(settings, "scoring-override")
    constructor.assert_called_once_with(
        "http://judge.invalid/v1", "test-judge-key", "scoring-override",
        requests_per_minute=settings.llm_requests_per_minute,
    )
    assert len(generation.prompts) == 1 and len(judging.prompts) == 1
    with pytest.raises(ValueError, match="JUDGE_API_KEY"):
        evaluate.check_api(settings.model_copy(update={"judge_api_key": None}))
    legacy = settings.model_copy(update={"judge_base_url": None, "judge_api_key": None, "judge_model": None})
    score, model, base_url = evaluate.build_judge(legacy, generation, None)
    assert score.llm is generation and model == legacy.llm_model and base_url == legacy.llm_base_url
