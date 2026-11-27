"""Run reproducible retrieval-only or end-to-end RAG evaluations."""
import argparse
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from src.config import ROOT, Settings
from src.error_messages import describe_error
from src.llm import OpenAICompatibleLLM
from src.rag import RAGPipeline
from src.reranker import Reranker
from src.schemas import SearchResult
from src.web_service import RAGLabService

from .judge import JUDGE_VERSION, Judge
from .metrics import attack_chunk, scores_for
from .reporting import (
    comparison_conclusion, complete, dataset_digest, digest, read_report, report_json,
    row_complete, save_comparison, write_atomic,
)
from .schemas import DATASET, INJECTION_SOURCE, RESULTS, Chunk, Example, Report, RetrievalMode, Row, RunConfig, Strategy, load_dataset

STRATEGIES: dict[str, Strategy] = {"a": "zero_shot", "b": "few_shot"}
Progress = Callable[[int, int, Row], None]


def as_chunk(result: SearchResult) -> Chunk:
    return Chunk(
        source=result.metadata["source"], chunk_id=result.metadata["chunk_id"],
        page=result.metadata.get("page"), text=result.text, score=result.score,
    )


class Evaluator:
    """Share retrieval between strategies and preserve an auditable report per run."""

    def __init__(
        self, pipeline: RAGPipeline, examples: list[Example], config: RunConfig,
        judge: Judge | None = None, injection_chunks: list[SearchResult] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.examples = examples
        self.config = config
        self.judge = judge
        self.injection_chunks = injection_chunks or []
        self.retrieval_cache: dict[int, tuple[list[SearchResult], list[SearchResult], float]] = {}

    def _retrieve(self, example: Example) -> tuple[list[SearchResult], list[SearchResult], float]:
        if example.id not in self.retrieval_cache:
            started = perf_counter()
            _, retrieved = self.pipeline.retrieve(example.question, self.config.retrieval_mode)
            dense_best = self.pipeline.store.search(example.question, 1) if self.judge is not None else []
            self.retrieval_cache[example.id] = (retrieved, dense_best, perf_counter() - started)
        return self.retrieval_cache[example.id]

    def _check_injection(self, row: Row, strategy: Strategy) -> None:
        natural = next((chunk for chunk in row.context if attack_chunk(chunk)), None)
        if natural is not None:
            row.injection_mode = "natural"
            row.injection_chunk_id = natural.chunk_id
            return
        row.injection_context = [as_chunk(chunk) for chunk in self.injection_chunks]
        controlled = next((chunk for chunk in row.injection_context if attack_chunk(chunk)), None)
        if controlled is None:
            row.errors["injection"] = "В индексированном тестовом документе не найден чанк с контрольной строкой."
            return
        row.injection_mode = "controlled"
        row.injection_chunk_id = controlled.chunk_id
        try:
            row.injection_answer, _ = self.pipeline.generate(row.example.question, self.injection_chunks, strategy)
        except Exception as exc:
            row.errors["injection"] = describe_error(exc)

    def evaluate(self, example: Example, strategy: Strategy | None = None) -> Row:
        started = perf_counter()
        row = Row(example=example)
        try:
            retrieved, dense_best, retrieval_seconds = self._retrieve(example)
            row.retrieved = [as_chunk(chunk) for chunk in retrieved]
        except Exception as exc:
            row.errors["retrieval"] = describe_error(exc)
        else:
            if strategy is not None and self.judge is not None:
                if self.pipeline.has_sufficient_context(retrieved, dense_best):
                    row.context = row.retrieved
                try:
                    generation_started = perf_counter()
                    row.answer, _ = self.pipeline.generate_from_retrieval(
                        example.question, retrieved, dense_best, strategy,
                    )
                    row.answer_seconds = retrieval_seconds + perf_counter() - generation_started
                except Exception as exc:
                    row.errors["generation"] = describe_error(exc)
                if row.answer is not None:
                    try:
                        row.judge = self.judge.score(example, row.context, row.answer)
                    except Exception as exc:
                        row.errors["judge"] = describe_error(exc)
                if "prompt_injection" in example.tags:
                    self._check_injection(row, strategy)
        row.elapsed_seconds = perf_counter() - started
        row.metrics = scores_for(row)
        return row

    def run(
        self, strategy: Strategy | None = None, *, limit: int | None = None,
        output: Path | None = None, resume: bool = False, progress: Progress | None = None,
    ) -> Report:
        if limit is not None and not 1 <= limit <= len(self.examples):
            raise ValueError("Лимит должен быть от 1 до размера датасета.")
        if (strategy is not None) != (self.judge is not None):
            raise ValueError("Для полного прогона нужны стратегия и judge.")
        selected = self.examples[:limit]
        report = Report(
            schema_version=1, config=self.config.model_copy(update={"strategy": strategy}),
            dataset_size=len(self.examples), selected_ids=[example.id for example in selected],
            limited=limit is not None,
        )
        saved: dict[int, Row] = {}
        if resume and output is not None and output.exists():
            previous = read_report(output)
            if (
                previous.config != report.config or previous.dataset_size != report.dataset_size
                or previous.selected_ids != report.selected_ids or previous.limited != report.limited
            ):
                raise ValueError("Параметры сохранённого прогона отличаются. Выберите другую папку результатов.")
            saved = {row.example.id: row for row in previous.rows}

        for number, example in enumerate(selected, 1):
            row = saved.get(example.id)
            if row is None or row.example != example or not row_complete(row, strategy is None):
                row = self.evaluate(example, strategy)
            report.rows.append(row)
            if output is not None:
                write_atomic(output, report_json(report))
            if progress is not None:
                progress(number, len(selected), row)
        return report


def validate_sources(examples: list[Example], sources: set[str]) -> None:
    missing = {source for example in examples for source in example.expected_sources} - sources
    if missing:
        raise ValueError("Отсутствуют эталонные документы: " + ", ".join(sorted(missing)))


def files_digest(paths: list[Path]) -> str:
    return digest(json.dumps([(path.name, digest(path.read_bytes())) for path in paths]).encode("utf-8"))


def build_evaluator(
    settings: Settings, examples: list[Example], *, retrieval_only: bool,
    mode: RetrievalMode, top_k: int, judge_model: str | None = None,
    service: RAGLabService | None = None,
) -> Evaluator:
    service = service or RAGLabService(settings)
    validate_sources(examples, set(service.files()))
    if not 1 <= top_k <= 100:
        raise ValueError("Top-k должен быть от 1 до 100.")
    store = service.store
    index = store.all()
    if not index:
        raise ValueError("Индекс пуст. Выполните python -m src.main index.")
    validate_sources(examples, {doc.metadata["source"] for doc in index})
    index_vectors = store.collection.get(include=["embeddings"])
    index_hash = digest(json.dumps({
        "chunks": sorted((doc.metadata["chunk_id"], doc.metadata, doc.text) for doc in index),
        "vectors": sorted(zip(index_vectors["ids"], index_vectors["embeddings"].tolist())),
    }, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    top_n = max(settings.top_n, top_k)
    llm = None if retrieval_only else service.llm
    pipeline = RAGPipeline(
        store, llm, settings.prompts_dir, top_n=top_n, final_k=top_k,
        min_score=settings.min_dense_score, rrf_k=settings.rrf_k,
        reranker=Reranker(settings.rerank_model, settings.rerank_enabled),
    )
    scorer = None
    prompt_hash = judge_hash = None
    if llm is not None:
        judge_model = judge_model or settings.llm_model
        judge_llm = llm if judge_model == settings.llm_model else OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_api_key, judge_model,
        )
        judge_path = settings.prompts_dir / "judge.md"
        scorer = Judge(judge_llm, judge_path)
        judge_hash = digest(judge_path.read_bytes())
        prompt_hash = files_digest([
            settings.prompts_dir / name for name in
            ("system.md", "structured.md", "user.md", "few_shot.md", "cot_structured.md")
        ])

    config = RunConfig(
        dataset_sha256=dataset_digest(examples), index_sha256=index_hash,
        prompts_sha256=prompt_hash, judge_sha256=judge_hash,
        implementation_sha256=files_digest([
            Path(__file__).with_name(name) for name in ("evaluate.py", "judge.py", "metrics.py", "schemas.py")
        ] + sorted((ROOT / "src").glob("*.py"))),
        collection=settings.collection_name, embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, retrieval_mode=mode,
        top_n=top_n, top_k=top_k, rerank_model=settings.rerank_model if settings.rerank_enabled else None,
        rrf_k=settings.rrf_k, min_dense_score=settings.min_dense_score,
        llm_model=settings.llm_model if llm is not None else None,
        endpoint_sha256=digest(settings.llm_base_url.encode("utf-8")) if llm is not None else None,
        judge_model=judge_model if scorer is not None else None,
        judge_version=JUDGE_VERSION if scorer is not None else None,
    )
    injection_chunks = [
        SearchResult(doc.text, doc.metadata, 0.0)
        for doc in index if doc.metadata["source"] == INJECTION_SOURCE
    ]
    return Evaluator(pipeline, examples, config, scorer, injection_chunks)


def console_progress(number: int, total: int, row: Row) -> None:
    if row.errors:
        print(f"[{number}/{total}] #{row.example.id} " + ", ".join(
            f"{stage} — {error}" for stage, error in row.errors.items()
        ), flush=True)
    elif number % 10 == 0 or number == total:
        print(f"[{number}/{total}] обработано", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка качества RAG на Golden Dataset")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--retrieval-only", action="store_true")
    operation.add_argument("--version", choices=list(STRATEGIES))
    operation.add_argument("--compare", action="store_true")
    operation.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=["dense", "hybrid"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-model", help="Модель judge на том же API; по умолчанию LLM_MODEL.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        settings = Settings()
        examples = load_dataset(args.dataset)
        if args.validate_dataset:
            validate_sources(examples, set(RAGLabService(settings).files()))
            print(f"Датасет проверен: {len(examples)} вопросов.")
            return 0
        folder = args.output_dir or (RESULTS / "trial" if args.limit is not None else RESULTS)
        evaluator = build_evaluator(
            settings, examples, retrieval_only=args.retrieval_only,
            mode=args.mode or settings.retrieval_mode,
            top_k=settings.final_k if args.top_k is None else args.top_k, judge_model=args.judge_model,
        )
        names = ("a", "b") if args.compare else ((args.version,) if args.version else ("retrieval",))
        reports = []
        for name in names:
            path = folder / (f"version_{name}.json" if name in STRATEGIES else "retrieval.json")
            print(f"Прогон {name.upper()} → {path}", flush=True)
            report = evaluator.run(
                STRATEGIES.get(name), limit=args.limit, output=path,
                resume=args.resume, progress=console_progress,
            )
            reports.append(report)
            print("Полный прогон." if args.limit is None and complete(report) else "Неполный прогон.", flush=True)
        if args.compare:
            rows = save_comparison(folder, *reports)
            for row in rows:
                if row["category"] == "all":
                    print(f"{row['metric']} — A {row['a']}, B {row['b']}, Δ {row['b_minus_a']}")
            conclusion = comparison_conclusion(*reports)
            if conclusion:
                print(conclusion)
        return int(any(row.errors for report in reports for row in report.rows))
    except Exception as exc:
        print(f"Оценка не выполнена: {describe_error(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
