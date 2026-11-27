import sys
from pathlib import Path

import pytest

from evals.eval_gate import check_gate, main
from evals.metrics import scores_for
from evals.reporting import report_json
from evals.schemas import INJECTION_MARKER, INJECTION_SOURCE, Chunk, JudgeScores, Report, Row, load_dataset
from src.schemas import INSUFFICIENT, Citation, RAGAnswer
from test_edd import JUDGE_JSON, config


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
            judge=JudgeScores.model_validate_json(JUDGE_JSON), answer_seconds=1,
            injection_mode="natural" if attack is not None else None,
            injection_chunk_id=attack.chunk_id if attack is not None else None,
        )
        row.metrics = scores_for(row)
        rows.append(row)
    return Report(
        schema_version=1, config=config(examples).model_copy(update={"strategy": "zero_shot"}),
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
