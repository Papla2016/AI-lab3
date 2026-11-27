"""Fail closed on thresholds, incomplete runs, or incompatible baselines."""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .metrics import METRICS, summarize
from .reporting import complete, dataset_digest, read_report
from .schemas import DATASET, RESULTS, Report, load_dataset

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "refusal_accuracy": 0.8,
    "hit_at_5": 0.7,
    "citation_validity": 1.0,
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"{self.name}: {'PASS' if self.passed else 'FAIL'} ({self.detail})"


def check_gate(
    report: Report, thresholds: dict[str, float] | None = None,
    baseline: Report | None = None, max_regression: float = 0.05,
) -> list[Check]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(thresholds) != set(DEFAULT_THRESHOLDS) or not all(0 <= x <= 1 for x in (*thresholds.values(), max_regression)):
        raise ValueError("Пороги и допустимое снижение должны быть числами от 0 до 1.")
    if report.config.strategy is None or not complete(report):
        return [Check("Полнота оценки", False, "требуется полный прогон генерации и judge без ошибок")]

    summary = summarize(report.rows)
    checks = []
    for name, threshold in thresholds.items():
        value = summary[name]["mean"]
        detail = f"{value:.3f} ≥ {threshold:.3f} требуется" if value is not None else "нет применимых оценок"
        checks.append(Check(METRICS[name], value is not None and value >= threshold, detail))

    if baseline is not None:
        required_matches = {
            "dataset_sha256", "index_sha256", "embedding_model", "chunk_size", "chunk_overlap",
            "retrieval_mode", "top_n", "top_k", "rerank_model", "rrf_k", "min_dense_score",
            "judge_model", "judge_version", "judge_sha256", "temperature",
        }
        if (
            baseline.config.strategy is None or not complete(baseline)
            or any(getattr(report.config, name) != getattr(baseline.config, name) for name in required_matches)
        ):
            checks.append(Check("Сравнение с baseline", False, "baseline неполный или его условия оценки отличаются"))
        else:
            previous = summarize(baseline.rows)["faithfulness"]["mean"]
            current = summary["faithfulness"]["mean"]
            drop = previous - current
            passed = drop <= max_regression or math.isclose(drop, max_regression, abs_tol=1e-12)
            checks.append(Check(
                "Регрессия Faithfulness", passed,
                f"снижение {drop:.3f}; допустимо {max_regression:.3f}",
            ))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить пороги качества и регрессию RAG")
    parser.add_argument("--results", type=Path, default=RESULTS / "version_b.json")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.05)
    for name, threshold in DEFAULT_THRESHOLDS.items():
        parser.add_argument("--min-" + name.replace("_", "-"), type=float, default=threshold)
    args = parser.parse_args()
    try:
        report = read_report(args.results)
        if report.config.dataset_sha256 != dataset_digest(load_dataset(args.dataset)):
            raise ValueError("Результаты относятся к другой версии датасета.")
        thresholds = {name: getattr(args, "min_" + name) for name in DEFAULT_THRESHOLDS}
        baseline = read_report(args.baseline) if args.baseline else None
        checks = check_gate(report, thresholds, baseline, args.max_regression)
    except (OSError, ValueError) as exc:
        detail = str(exc) if type(exc) is ValueError else "файл отсутствует, недоступен или не соответствует схеме"
        checks = [Check("Загрузка оценки", False, detail)]
    for check in checks:
        print(check.line())
    passed = all(check.passed for check in checks)
    print(f"Regression gate: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
