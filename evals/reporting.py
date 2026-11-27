"""Checkpoints and reproducible, machine-readable report exports."""
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .metrics import JUDGE_METRICS, METRICS, by_category, scores_for, summarize
from .schemas import Example, Report, Row


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_digest(examples: list[Example]) -> str:
    return digest(json.dumps(
        [example.model_dump(mode="json") for example in examples], sort_keys=True, ensure_ascii=False,
    ).encode("utf-8"))


def row_complete(row: Row, retrieval_only: bool) -> bool:
    if row.errors or row.metrics != scores_for(row):
        return False
    if retrieval_only:
        return True
    if row.answer is None or row.judge is None or row.answer_seconds is None:
        return False
    return "prompt_injection" not in row.example.tags or row.metrics["injection_resistance"] is not None


def complete(report: Report) -> bool:
    ids = [row.example.id for row in report.rows]
    return (
        not report.limited
        and len(ids) == len(report.selected_ids) == report.dataset_size
        and len(set(ids)) == len(ids)
        and ids == report.selected_ids
        and dataset_digest([row.example for row in report.rows]) == report.config.dataset_sha256
        and all(row_complete(row, report.config.strategy is None) for row in report.rows)
    )


def report_json(report: Report) -> str:
    payload = report.model_dump(mode="json")
    payload.update(
        complete=complete(report), summary=summarize(report.rows),
        by_category=by_category(report),
    )
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def read_report(path: Path) -> Report:
    return Report.model_validate_json(path.read_bytes())


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
            for key, value in row.items()
        })
    return stream.getvalue()


def report_csv(report: Report) -> str:
    return csv_text([
        {
            "id": row.example.id, "type": row.example.type, "question": row.example.question,
            "ground_truth": row.example.ground_truth, "answer": row.answer.answer if row.answer else None,
            "expected_sources": json.dumps(row.example.expected_sources, ensure_ascii=False),
            "retrieved": json.dumps([chunk.model_dump(mode="json") for chunk in row.retrieved], ensure_ascii=False),
            "context": json.dumps([chunk.model_dump(mode="json") for chunk in row.context], ensure_ascii=False),
            "justification": row.judge.justification if row.judge else None,
            "injection_mode": row.injection_mode, "injection_chunk_id": row.injection_chunk_id,
            "errors": json.dumps(row.errors, ensure_ascii=False),
            **scores_for(row),
        }
        for row in report.rows
    ])


def compare(a: Report, b: Report) -> list[dict]:
    if (
        a.config.strategy is None or b.config.strategy is None
        or a.config.model_dump(exclude={"strategy"}) != b.config.model_dump(exclude={"strategy"})
        or a.selected_ids != b.selected_ids or a.limited != b.limited
    ):
        raise ValueError("У A и B должны совпадать конфигурация и список вопросов.")
    summaries_a = {"all": summarize(a.rows), **by_category(a)}
    summaries_b = {"all": summarize(b.rows), **by_category(b)}
    rows = []
    for scope, values_a in summaries_a.items():
        for metric, label in METRICS.items():
            item_a, item_b = values_a[metric], summaries_b[scope][metric]
            mean_a, mean_b = item_a["mean"], item_b["mean"]
            rows.append({
                "category": scope, "metric": label, "a": mean_a, "b": mean_b,
                "b_minus_a": mean_b - mean_a if mean_a is not None and mean_b is not None else None,
                "n_a": item_a["n"], "n_b": item_b["n"],
            })
    return rows


def comparison_conclusion(a: Report, b: Report) -> str | None:
    if not (complete(a) and complete(b)):
        return None
    summary_a, summary_b = summarize(a.rows), summarize(b.rows)
    delta = sum(summary_b[key]["mean"] - summary_a[key]["mean"] for key in JUDGE_METRICS) / len(JUDGE_METRICS)
    if abs(delta) < 1e-12:
        return "Средние оценки judge по четырём осям совпали."
    winner = "B" if delta > 0 else "A"
    refusal_delta = summary_b["refusal_accuracy"]["mean"] - summary_a["refusal_accuracy"]["mean"]
    conclusion = f"{winner} выше по среднему четырёх оценок judge на {abs(delta):.3f}."
    if delta * refusal_delta < 0:
        conclusion += f" При этом у {winner} ниже точность отказов на {abs(refusal_delta):.3f}."
    return conclusion


def save_comparison(path: Path, a: Report, b: Report) -> list[dict]:
    rows = compare(a, b)
    payload = {
        "config_a": a.config.model_dump(mode="json"), "config_b": b.config.model_dump(mode="json"),
        "complete_a": complete(a), "complete_b": complete(b),
        "selected_ids": a.selected_ids, "metrics": rows, "conclusion": comparison_conclusion(a, b),
    }
    write_atomic(path / "comparison.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_atomic(path / "comparison.csv", csv_text(rows))
    return rows
