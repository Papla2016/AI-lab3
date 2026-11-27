import sys
from pathlib import Path

import pytest

from evals.eval_gate import check_gate, main
from evals.metrics import scores_for
from evals.reporting import dataset_digest, report_json
from evals.schemas import INJECTION_MARKER, INJECTION_SOURCE, Chunk, JudgeScores, Report, Row, load_dataset
from src.schemas import INSUFFICIENT, Citation, RAGAnswer
from test_edd import PASSING_JUDGE_JSON, config


@pytest.fixture
def result() -> Report:
    examples = load_dataset()
    rows = []
    for example in examples:
        chunks = [
            Chunk(source=source, chunk_id=str(i), text=INJECTION_MARKER if source == INJECTION_SOURCE else "Документ.", score=1)
            for i, source in enumerate(example.expected_sources)
        ]
        citation = Citation(source=chunks[0].source, chunk_id=chunks[0].chunk_id) if chunks else None
        answer = RAGAnswer(answer="Ответ.", citations=[citation], confidence="high", insufficient_context=False) if citation else INSUFFICIENT
        attack = next((chunk for chunk in chunks if chunk.source == INJECTION_SOURCE), None)
        row = Row(
            example=example, retrieved=chunks, context=chunks, answer=answer,
            judge=JudgeScores.model_validate_json(PASSING_JUDGE_JSON), answer_seconds=1,
            injection_mode="natural" if attack is not None else None,
            injection_chunk_id=attack.chunk_id if attack is not None else None,
        )
        row.metrics = scores_for(row)
        rows.append(row)
    return Report(
        schema_version=2, config=config(examples).model_copy(update={"strategy": "zero_shot"}),
        dataset_size=len(examples), selected_ids=[example.id for example in examples],
        limited=False, rows=rows,
    )


def test_gate_pass_and_configurable_threshold_fail(result):
    assert all(check.passed for check in check_gate(result))
    failures = [check.name for check in check_gate(result, {"answer_relevancy": 0.8}) if not check.passed]
    assert failures == ["Answer Relevancy"]


@pytest.mark.parametrize("issue", ["interrupted", "limited", "judge_error", "missing_scores", "tampered_scores", "retrieval_only"])
def test_gate_fails_closed_for_incomplete_report(result, issue):
    if issue == "interrupted":
        result.rows.pop()
    elif issue == "limited":
        result.limited = True
    elif issue == "judge_error":
        result.rows[0].errors["judge"] = "API unavailable"
        result.rows[0].judge = None
    elif issue == "missing_scores":
        result.rows[0].judge = None
        result.rows[0].metrics = scores_for(result.rows[0])
    elif issue == "tampered_scores":
        result.rows[0].metrics["faithfulness"] = 0.9
    else:
        result.config.strategy = None
    assert any(not check.passed for check in check_gate(result))


def test_gate_detects_faithfulness_regression_against_baseline(result):
    baseline = result.model_copy(deep=True)
    for row in result.rows:
        row.judge.faithfulness = 4
        row.metrics = scores_for(row)
    assert all(check.passed for check in check_gate(result))
    assert not check_gate(result, baseline=baseline, max_regression=0.05)[-1].passed
    assert check_gate(result, baseline=baseline, max_regression=0.25)[-1].passed
    baseline.config.judge_sha256 = "different"
    assert not check_gate(result, baseline=baseline)[-1].passed


def test_gate_cli_missing_corrupt_and_success(result, tmp_path: Path, monkeypatch, capsys):
    path = tmp_path / "version_b.json"
    monkeypatch.setattr(sys, "argv", ["eval_gate", "--results", str(path)])
    assert main() == 1
    path.write_text("bad JSON", encoding="utf-8")
    assert main() == 1
    path.write_text(report_json(result), encoding="utf-8")
    assert main() == 0
    output = capsys.readouterr().out
    assert "Regression gate: FAILED" in output
    assert "Regression gate: PASSED" in output


@pytest.mark.parametrize("metric,label", [
    ("correctness", "Correctness"), ("completeness", "Completeness"),
    ("false_refusal_rate", "False Refusal Rate"), ("injection_resistance", "Injection Resistance"),
])
def test_new_gate_thresholds_fail_at_actual_regression(result, metric, label):
    for row in result.rows:
        if metric in {"correctness", "completeness"}:
            setattr(row.judge, metric, 3)
        elif metric == "false_refusal_rate" and row.example.type == "in_corpus":
            row.answer = INSUFFICIENT
        elif metric == "injection_resistance" and row.injection_mode:
            row.answer.answer = INJECTION_MARKER
        row.metrics = scores_for(row)
    checks = {check.name: check for check in check_gate(result)}
    assert not checks[label].passed


def test_gate_requires_actual_injection_exposure_when_tagged(result):
    attack = next(row for row in result.rows if row.injection_mode)
    attack.injection_chunk_id = "missing"
    attack.metrics = scores_for(attack)
    assert attack.metrics["injection_resistance"] is None
    assert not check_gate(result)[0].passed


def test_gate_skips_injection_only_without_tagged_cases(result):
    for row in result.rows:
        if row.injection_mode:
            row.example.tags = ["documentation"]
            row.injection_mode = None
            row.injection_chunk_id = None
            row.metrics = scores_for(row)
    result.config.dataset_sha256 = dataset_digest([row.example for row in result.rows])
    checks = {check.name: check for check in check_gate(result)}
    assert all(check.passed for check in checks.values())
    assert "неприменимо" in checks["Injection Resistance"].detail


def test_gate_rejects_different_k_or_mode_in_baseline(result):
    baseline = result.model_copy(deep=True)
    baseline.config.top_k = 1
    assert "разные K" in check_gate(result, baseline=baseline)[-1].detail
    baseline.config.top_k = result.config.top_k
    baseline.config.evaluation_mode = "end_to_end"
    assert "режимы" in check_gate(result, baseline=baseline)[-1].detail


def test_gate_cli_reads_upper_bound_override(result, tmp_path, monkeypatch, capsys):
    row = next(row for row in result.rows if row.example.type == "in_corpus" and not row.injection_mode)
    row.answer = INSUFFICIENT
    row.metrics = scores_for(row)
    path = tmp_path / "report.json"
    path.write_text(report_json(result), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval_gate", "--results", str(path), "--max-false-refusal-rate", "0"])
    assert main() == 1
    assert "False Refusal Rate: FAIL" in capsys.readouterr().out
