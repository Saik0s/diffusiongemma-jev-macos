import math
import re
from typing import Literal

import pytest

from diffusion_jev.canvas import (
    compile_canvas,
    entropy_confidence,
    make_answer,
    system_prompt,
)
from diffusion_jev.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


class LexemeTokenizer:
    """Deterministic word tokens, with explicit defects for alignment regressions."""

    def __init__(
        self, defect: Literal["none", "split", "collision", "multiple", "same"] = "none"
    ) -> None:
        self.defect = defect
        self.vocabulary: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        words = re.findall(r"<\|channel>|<channel\|>|<turn\|>|\w+|[^\w\s]", text)
        if self.defect == "multiple" and "no" in words:
            words = ["r0" if word == "q0" else word for word in words]
        result: list[int] = []
        for word in words:
            if self.defect == "split" and word == "no":
                lexemes = ["n", "o"]
            elif self.defect == "collision" and word == "C":
                lexemes = ["B"]
            elif self.defect == "same" and word == "no":
                lexemes = ["yes"]
            else:
                lexemes = [word]
            for lexeme in lexemes:
                result.append(self.vocabulary.setdefault(lexeme, len(self.vocabulary) + 1))
        return result


def noul() -> NoulQuestion:
    return NoulQuestion(type="noul", instructions="The change is safe")


def choice() -> ChoiceQuestion:
    return ChoiceQuestion(
        type="choice", instructions="Choose an action",
        criteria={"accept": "Accept it", "review": "Review it", "reject": "Reject it"},
    )


def score() -> ScoreQuestion:
    return ScoreQuestion(
        type="score", instructions="Rate confidence", criteria=["Low", "Mid", "High"]
    )


@pytest.mark.parametrize("defect", ["split", "multiple", "same"])
def test_rejects_unaligned_noul_labels(defect: Literal["split", "multiple", "same"]) -> None:
    with pytest.raises(ValueError, match="Answer labels"):
        compile_canvas(LexemeTokenizer(defect), {"id": noul()}, width=None, pad_token_id=0)


def test_rejects_colliding_alternative_labels() -> None:
    with pytest.raises(ValueError, match="distinct token IDs"):
        compile_canvas(LexemeTokenizer("collision"), {"id": choice()}, width=None, pad_token_id=0)


def test_canvas_uses_minimum_block_and_preserves_turn_marker() -> None:
    tokenizer = LexemeTokenizer()
    canvas = compile_canvas(tokenizer, {"id": noul()}, width=None, pad_token_id=0)
    assert len(canvas.tokens) == 16
    assert canvas.slots[0].token_ids == (
        tokenizer.vocabulary["yes"], tokenizer.vocabulary["no"]
    )
    turn_position = canvas.tokens.index(tokenizer.vocabulary["<turn|>"])
    assert turn_position > canvas.slots[0].position
    assert all(token == 0 for token in canvas.tokens[turn_position + 1:])


@pytest.mark.parametrize("width", [16, 32, 256])
def test_explicit_canvas_size(width: int) -> None:
    canvas = compile_canvas(LexemeTokenizer(), {"id": noul()}, width=width, pad_token_id=0)
    assert len(canvas.tokens) == width


@pytest.mark.parametrize("width", [0, 1, 17, 272])
def test_rejects_canvas_outside_bounds(width: int) -> None:
    with pytest.raises(ValueError):
        compile_canvas(LexemeTokenizer(), {"id": noul()}, width=width, pad_token_id=0)


def test_automatic_canvas_grows_in_blocks() -> None:
    questions: dict[str, Question] = {f"id{index}": noul() for index in range(8)}
    canvas = compile_canvas(LexemeTokenizer(), questions, width=None, pad_token_id=0)
    assert len(canvas.tokens) == 32
    assert len({slot.position for slot in canvas.slots}) == 8
    with pytest.raises(ValueError, match="does not fit"):
        compile_canvas(LexemeTokenizer(), questions, width=16, pad_token_id=0)


def test_transport_ids_are_excluded_from_model_text() -> None:
    questions: dict[str, Question] = {"TRANSPORT_SECRET_ONE": noul(), "OTHER_PRIVATE_ID": choice()}
    tokenizer = LexemeTokenizer()
    canvas = compile_canvas(tokenizer, questions, width=None, pad_token_id=0)
    prompt = system_prompt(questions)
    for identifier in questions:
        assert identifier not in prompt
        assert identifier not in tokenizer.vocabulary
    assert "Question q0:" in prompt
    assert "Question q1:" in prompt
    assert [slot.question_id for slot in canvas.slots] == list(questions)


def test_seeded_noise_changes_only_answer_slots() -> None:
    questions: dict[str, Question] = {"first": noul(), "second": choice()}
    canvas = compile_canvas(LexemeTokenizer(), questions, width=None, pad_token_id=0)
    first = canvas.seeded(42, 1000)
    assert first == canvas.seeded(42, 1000)
    assert first != canvas.seeded(43, 1000)
    positions = {slot.position for slot in canvas.slots}
    changed = {
        index for index, (a, b) in enumerate(zip(canvas.tokens, first, strict=True)) if a != b
    }
    assert changed == positions
    assert all(0 <= first[position] < 1000 for position in positions)


def test_confidence_entropy_endpoints() -> None:
    assert entropy_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert entropy_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert 0 < entropy_confidence([0.9, 0.1]) < 1


def test_noul_probability_and_choice_distribution_normalize() -> None:
    assert make_answer(noul(), [3.0, 1.0]) == NoulAnswer(noul=0.75)
    answer = make_answer(choice(), [1.0, 7.0, 2.0])
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "review"
    assert answer.probabilities == {"accept": 0.1, "review": 0.7, "reject": 0.2}


def test_score_is_expected_zero_based_ordinal_value() -> None:
    answer = make_answer(score(), [2.0, 3.0, 5.0])
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(1.3)
    assert answer.legend == {"0": "Low", "1": "Mid", "2": "High"}
    assert answer.probabilities == {"0": 0.2, "1": 0.3, "2": 0.5}


@pytest.mark.parametrize(
    "probabilities", [[], [1.0], [0.0, 0.0], [-1.0, 2.0], [math.nan, 1.0], [math.inf, 1.0]]
)
def test_rejects_invalid_distributions(probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        make_answer(noul(), probabilities)
