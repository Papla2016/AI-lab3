"""Fixed-rubric LLM-as-judge with bounded schema retries."""
import json
from pathlib import Path

from pydantic import ValidationError

from src.llm import LLM
from src.schemas import RAGAnswer

from .schemas import Chunk, Example, JudgeScores

JUDGE_VERSION = "judge-v1"


class InvalidJudgeOutput(ValueError):
    """The model did not produce a valid score object."""


class Judge:
    def __init__(self, llm: LLM, prompt_path: Path) -> None:
        self.llm = llm
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def score(self, example: Example, context: list[Chunk], answer: RAGAnswer) -> JudgeScores:
        payload = json.dumps(
            {
                "question": example.question,
                "ground_truth": example.ground_truth,
                "expected_behavior": example.expected_behavior,
                "context": [chunk.model_dump(exclude={"score"}) for chunk in context],
                "answer": answer.model_dump(),
            },
            ensure_ascii=False,
        )
        for attempt in range(2):
            system = self.prompt
            if attempt:
                system += "\nПредыдущий ответ не прошёл проверку схемы. Верни ровно один JSON по рубрике."
            raw = self.llm.generate(system, payload)
            try:
                return JudgeScores.model_validate_json(raw)
            except ValidationError:
                continue
        raise InvalidJudgeOutput("Judge дважды вернул JSON, не соответствующий рубрике.")
