"""Dataset and evaluation result contracts."""
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.config import ROOT
from src.schemas import RAGAnswer

DATASET = ROOT / "evals" / "dataset.json"
RESULTS = ROOT / "evals" / "results"
INJECTION_SOURCE = "Prompt_injection_ALT_packages.pdf"
INJECTION_MARKER = "PROMPT_INJECTION_SUCCESS"
QuestionType = Literal["in_corpus", "out_of_corpus", "ambiguous"]
Strategy = Literal["zero_shot", "few_shot", "cot_structured"]
RetrievalMode = Literal["dense", "hybrid"]
EvaluationMode = Literal["retrieval_only", "prompt_comparison", "end_to_end"]


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: int = Field(gt=0)
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    expected_sources: list[str]
    source_groups: list[list[str]] = Field(default_factory=list)
    type: QuestionType
    tags: list[str] = Field(min_length=1)
    ambiguous_behavior: Literal["clarify", "list_interpretations"] | None = None

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        sources = set(self.expected_sources)
        if len(sources) != len(self.expected_sources):
            raise ValueError("Эталонные источники должны быть уникальны.")
        if bool(sources) == (self.type == "out_of_corpus"):
            raise ValueError("Эталонные источники нужны только для вопросов по корпусу.")
        if self.source_groups and (
            any(not group or len(group) != len(set(group)) for group in self.source_groups)
            or {source for group in self.source_groups for source in group} != sources
            or len(self.source_groups) != len({frozenset(group) for group in self.source_groups})
        ):
            raise ValueError("Группы должны покрывать ожидаемые источники без пустых и повторных групп.")
        if (self.ambiguous_behavior is not None) != (self.type == "ambiguous"):
            raise ValueError("Для неоднозначного вопроса задайте ожидаемое поведение.")
        if "prompt_injection" in self.tags and INJECTION_SOURCE not in sources:
            raise ValueError("Проверка инъекции должна ссылаться на тестовый документ.")
        return self

    @property
    def expected_behavior(self) -> str:
        if self.type == "out_of_corpus":
            return "Отказ из-за недостатка сведений, без неподтверждённых утверждений."
        if self.ambiguous_behavior == "clarify":
            return "Задать уточняющий вопрос, сохранив опору на контекст."
        if self.ambiguous_behavior == "list_interpretations":
            return "Различить возможные трактовки и ответить на них по контексту."
        return "Ответить на вопрос по контексту и эталонным фактам."


EXAMPLES = TypeAdapter(list[Example])


def load_dataset(path: Path = DATASET) -> list[Example]:
    examples = EXAMPLES.validate_json(path.read_bytes())
    questions = {" ".join(item.question.casefold().split()) for item in examples}
    if len(examples) < 100 or len({item.id for item in examples}) != len(examples) or len(questions) != len(examples):
        raise ValueError("Нужно не менее 100 примеров с уникальными ID и вопросами.")
    if {item.type for item in examples} != {"in_corpus", "out_of_corpus", "ambiguous"}:
        raise ValueError("Датасет должен включать все три типа вопросов.")
    return examples


class Chunk(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    source: str
    chunk_id: str
    page: int | None = None
    text: str
    score: float


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    faithfulness: int = Field(ge=1, le=5)
    answer_relevancy: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    justification: str = Field(min_length=1)


class Row(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    example: Example
    retrieved: list[Chunk] = Field(default_factory=list)
    context: list[Chunk] = Field(default_factory=list)
    router: str | None = None
    answer: RAGAnswer | None = None
    judge: JudgeScores | None = None
    injection_mode: Literal["natural", "controlled"] | None = None
    injection_chunk_id: str | None = None
    injection_context: list[Chunk] = Field(default_factory=list)
    injection_answer: RAGAnswer | None = None
    injection_judge: JudgeScores | None = None
    answer_seconds: float | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)
    errors: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset_sha256: str
    index_sha256: str
    prompts_sha256: str | None = None
    implementation_sha256: str
    collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    evaluation_mode: EvaluationMode = "prompt_comparison"
    retrieval_mode: RetrievalMode
    top_n: int = Field(ge=1)
    top_k: int = Field(ge=1)
    rerank_model: str | None = None
    rrf_k: int
    min_dense_score: float
    llm_model: str | None = None
    endpoint_sha256: str | None = None
    temperature: Literal[0] = 0
    judge_model: str | None = None
    judge_endpoint_sha256: str | None = None
    judge_version: str | None = None
    judge_sha256: str | None = None
    strategy: Strategy | None = None


class Report(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal[2]
    config: RunConfig
    dataset_size: int = Field(ge=100)
    selected_ids: list[int] = Field(min_length=1)
    limited: bool
    rows: list[Row] = Field(default_factory=list)
