"""Runtime artifacts from the immutable Hugging Face checkpoint.

Sizes and LFS SHA256 values come from the repository API at MODEL_REVISION.
Small Git files were SHA256-hashed from that revision's HTTPS resolve endpoint.
Keep the revision and the entire manifest together when updating the checkpoint.
"""

from dataclasses import dataclass

MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit"
MODEL_REVISION = "30f3c7c7746bf41cfd1a290155cc3b777ab588b9"


@dataclass(frozen=True)
class Artifact:
    name: str
    size: int
    sha256: str


ARTIFACTS = (
    Artifact(
        "chat_template.jinja",
        18575,
        "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf",
    ),
    Artifact(
        "config.json",
        60763,
        "158213785d452d6a0bc4895329aaf6e0ca5dd84c0e5e7ced1a69a9d4c032a695",
    ),
    Artifact(
        "generation_config.json",
        357,
        "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
    ),
    Artifact(
        "model-00001-of-00004.safetensors",
        5261403418,
        "d69231d9f02fa6183d484e5c7c224367c91852a6ad64f6dd86e7208ea34a4569",
    ),
    Artifact(
        "model-00002-of-00004.safetensors",
        5286689521,
        "d83b4facd6beccd3cb8012279f6286b8d5c7e4e389fd76159ea152ee0256c5d7",
    ),
    Artifact(
        "model-00003-of-00004.safetensors",
        5337619397,
        "e0777ad24439e6e4fdfdfc18df2e5aa939545d1928682a94d2cd68fd1cc0dd56",
    ),
    Artifact(
        "model-00004-of-00004.safetensors",
        788295575,
        "55f1b63d865d0b5a70d0ad9772c21d75f19a89bba56bd49cc96a9fb8fa2eae27",
    ),
    Artifact(
        "model.safetensors.index.json",
        122395,
        "91534e31a36b17f6580c39467f347eebb992a1935eea2b8d944b9ef062aabf33",
    ),
    Artifact(
        "optiq/optiq_vision.safetensors",
        1145638984,
        "e766eb61b7ccb9cbc43bffb9fdce1c7f4076214c4b2d6b200ec5c836c962ac29",
    ),
    Artifact(
        "optiq_metadata.json",
        28847,
        "34a24bc6591203e83f31f43d8b5f5fe01e0b37a417c0ca945e6e47363c2ca6aa",
    ),
    Artifact(
        "processor_config.json",
        1689,
        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    ),
    Artifact(
        "tokenizer.json",
        32169626,
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    Artifact(
        "tokenizer_config.json",
        2741,
        "a284d1243b62be31faa9c13e1c28cece940c4abaa7bd9ad87b94f61b40687200",
    ),
)
