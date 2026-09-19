import pytest
from pydantic import ValidationError

from diffusion_jev.schemas import (
    DecisionOptions,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
)


def test_defaults() -> None:
    request = DecisionRequest.model_validate_json(
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check it"}}}'
    )
    assert request.options == DecisionOptions()
    assert request.model == "diffusiongemma-local"


@pytest.mark.parametrize(
    "payload",
    [
        '{"state":null,"questions":{}}',
        '{"state":null,"questions":{"q":{"type":"noul","instructions":""}}}',
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"   "}}}',
        '{"state":null,"questions":{"q":{"type":"unknown","instructions":"Check"}}}',
        '{"state":null,"questions":{"q":{"type":"choice","instructions":"Check",'
        '"criteria":{"a":"Only one"}}}}',
        '{"state":null,"questions":{"q":{"type":"score","instructions":"Check",'
        '"criteria":["Only one"]}}}',
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check",'
        '"unknown":true}}}',
        # A Noul rubric needs both sides, like the hosted API.
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check",'
        '"criteria":{"true":"Yes side"}}}}',
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check",'
        '"criteria":{"true":"Yes side","false":"No side","maybe":"Other"}}}}',
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check",'
        '"criteria":["Yes side","No side"]}}}',
    ],
)
def test_invalid_questions(payload: str) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        '{"samples":0}', '{"samples":9}', '{"samples":"2"}', '{"samples":true}',
        '{"seed":-1}', '{"seed":4294967296}', '{"canvas_length":17}',
        '{"canvas_length":272}', '{"mode":"other"}', '{"unknown":1}',
    ],
)
def test_invalid_options(payload: str) -> None:
    with pytest.raises(ValidationError):
        DecisionOptions.model_validate_json(payload)


def test_noul_criteria_are_optional_and_two_sided() -> None:
    request = DecisionRequest.model_validate_json(
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check it",'
        '"criteria":{"true":"It happened","false":"It did not"}}}}'
    )
    question = request.questions["q"]
    assert isinstance(question, NoulQuestion)
    assert question.criteria == NoulCriteria(true="It happened", false="It did not")
    plain = DecisionRequest.model_validate_json(
        '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check it"}}}'
    )
    assert plain.questions["q"] == NoulQuestion(type="noul", instructions="Check it")
