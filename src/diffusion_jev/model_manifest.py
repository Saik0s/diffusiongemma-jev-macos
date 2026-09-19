"""Runtime artifacts from immutable Hugging Face checkpoints.

Sizes and LFS SHA256 values come from each profile's pinned repository API.
Small Git files were SHA256-hashed from its revision's HTTPS resolve endpoint.
Keep each revision and its entire manifest together when updating a checkpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit"
MODEL_REVISION = "30f3c7c7746bf41cfd1a290155cc3b777ab588b9"


@dataclass(frozen=True)
class Artifact:
    name: str
    size: int
    sha256: str


Precision = Literal["optiq4", "8bit", "bf16"]


@dataclass(frozen=True)
class CheckpointProfile:
    precision: Precision
    repo: str
    revision: str
    artifacts: tuple[Artifact, ...]
    layout: Literal["optiq", "mlx"]
    quantization_bits: int | None
    model_type: str = "diffusion_gemma"


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


EIGHT_BIT_ARTIFACTS = (
    Artifact(
        "chat_template.jinja", 17336,
        "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    ),
    Artifact(
        "config.json", 31816,
        "78a0f27a2ca504ee712a37fd375377fe3f44e4b8c8b6be7701e33ec794191faf",
    ),
    Artifact(
        "generation_config.json", 357,
        "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
    ),
    Artifact(
        "model-00001-of-00006.safetensors", 5180807931,
        "3692eceb262c42c60882b912f59b3745d8012d29069dd3f24b5f1460a70019b2",
    ),
    Artifact(
        "model-00002-of-00006.safetensors", 5205336336,
        "a9f4fd5fe5dffccdc95dba4dc0015beceea2b93fed40daa68ab4cf02a3c4d657",
    ),
    Artifact(
        "model-00003-of-00006.safetensors", 5205336472,
        "3f89a5f70b455eabb3f9c8002f46b9435ad8328c02c5d073685e93400cec36bc",
    ),
    Artifact(
        "model-00004-of-00006.safetensors", 5205336448,
        "e72aa784f862479bf832ee0898f162b6c285a61635503de15a4ea12385d392ca",
    ),
    Artifact(
        "model-00005-of-00006.safetensors", 5205336504,
        "6b696a0ed00bebeb6612293bd7b1ff655081194f330eac3d70bd1f030d4e9dc1",
    ),
    Artifact(
        "model-00006-of-00006.safetensors", 1970437483,
        "c816183275ad6869bfeffc6a39e50dd9ef3b52babdadf77253a22acc5f3f079b",
    ),
    Artifact(
        "model.safetensors.index.json", 165362,
        "baabff1540009f87c27adb240dc43ccd5ad63545aecf1ac33716aa340ad4d3bb",
    ),
    Artifact(
        "processor_config.json", 911,
        "b113d8d45d04095c078674b92ec5f32b2857abee2c99c6993e1e557affd69cf9",
    ),
    Artifact(
        "tokenizer.json", 32169626,
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    Artifact(
        "tokenizer_config.json", 2749,
        "c1112569480ce4f86410f29131ff779f8944097f118f5524f4762ca85a32aa5b",
    ),
)

BF16_ARTIFACTS = (
    Artifact(
        "chat_template.jinja", 17336,
        "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    ),
    Artifact(
        "config.json", 4571,
        "07c985795affac409989c6a9887f629c5b77e12d438240ce9cbcd37bdd5dd733",
    ),
    Artifact(
        "generation_config.json", 357,
        "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200",
    ),
    Artifact(
        "model-00001-of-00011.safetensors", 4838423037,
        "87051d9c9947f6e0515e395f6cf2bc493c215af3f220be326a654929e6ef7689",
    ),
    Artifact(
        "model-00002-of-00011.safetensors", 4913414697,
        "6fbf065b2b089c92cabcdcdfa3f6c8d06dca73dfaca7f2302ab15afb2c847fc6",
    ),
    Artifact(
        "model-00003-of-00011.safetensors", 4884577972,
        "2acd09ba6529601c7986195402f9ce35212ad37c964d12e42b8faef4b64af618",
    ),
    Artifact(
        "model-00004-of-00011.safetensors", 4913414760,
        "a8327ef5380f4847126c571e0f5d9f30de90095f298e8997d4f8de08c8d89a52",
    ),
    Artifact(
        "model-00005-of-00011.safetensors", 4884578044,
        "f357583a7fa9c28bb9ddd92439f3c7d539bfaf85021a46c8bb2119bd304a3dfe",
    ),
    Artifact(
        "model-00006-of-00011.safetensors", 4913414738,
        "360b663be82b5f9587d8f7fc82a6efd686c343529b32a0263ed54dc791600f61",
    ),
    Artifact(
        "model-00007-of-00011.safetensors", 4884577994,
        "95817358c87a09aeec6603cfd100a7352b5b0cdaa5d7c28226d1f0963ff6afd9",
    ),
    Artifact(
        "model-00008-of-00011.safetensors", 4913414746,
        "0668d9e3a365ed6a517fc5473f19bb68e5cf4e06cd4b5545d8a318b6eb6c1109",
    ),
    Artifact(
        "model-00009-of-00011.safetensors", 4884578042,
        "9a02330eafd8f96655ef661d4d3ae8820c0c0cb6a278452a2f08f4c4c441ed82",
    ),
    Artifact(
        "model-00010-of-00011.safetensors", 4913414776,
        "5eb95cf91d242d443b48b8cd7cfe6c7ca1bb8a9535800d4f7fa61470c96c848d",
    ),
    Artifact(
        "model-00011-of-00011.safetensors", 2703891110,
        "62bfdda484bb85ce3dfa3ec2f862e778072a7cab858032069a74d0cdd570b059",
    ),
    Artifact(
        "model.safetensors.index.json", 109232,
        "2a1863ed71efb78bc05d949887849853763a24c816dc729694436b2e798aa2bd",
    ),
    Artifact(
        "processor_config.json", 911,
        "b113d8d45d04095c078674b92ec5f32b2857abee2c99c6993e1e557affd69cf9",
    ),
    Artifact(
        "tokenizer.json", 32169626,
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    Artifact(
        "tokenizer_config.json", 2749,
        "c1112569480ce4f86410f29131ff779f8944097f118f5524f4762ca85a32aa5b",
    ),
)

CHECKPOINTS: Mapping[Precision, CheckpointProfile] = MappingProxyType({
    "optiq4": CheckpointProfile("optiq4", MODEL_REPO, MODEL_REVISION, ARTIFACTS, "optiq", 4),
    "8bit": CheckpointProfile(
        "8bit", "mlx-community/diffusiongemma-26B-A4B-it-8bit",
        "7b95e3887078ba56283c24f2578d6e5a06b9d7e8", EIGHT_BIT_ARTIFACTS, "mlx", 8,
    ),
    "bf16": CheckpointProfile(
        "bf16", "mlx-community/diffusiongemma-26B-A4B-it-bf16",
        "2cd36f950eb065c96c80810fb6b859b114cd052d", BF16_ARTIFACTS, "mlx", None,
    ),
})
