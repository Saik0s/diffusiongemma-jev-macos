import pytest
from pydantic import ValidationError

from diffusion_jev.schemas import DecisionOptions, DecisionRequest


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
