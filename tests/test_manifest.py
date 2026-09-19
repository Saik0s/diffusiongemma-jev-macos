import re

import pytest

from diffusion_jev.model_manifest import CHECKPOINTS, Precision


@pytest.mark.parametrize(
    "precision,revision,total,count",
    [
        ("8bit", "7b95e3887078ba56283c24f2578d6e5a06b9d7e8", 28004979331, 13),
        ("bf16", "2cd36f950eb065c96c80810fb6b859b114cd052d", 51680004698, 18),
    ],
)
def test_standard_manifest_pins(
    precision: Precision, revision: str, total: int, count: int
) -> None:
    profile = CHECKPOINTS[precision]
    assert profile.revision == revision
    assert profile.repo == f"mlx-community/diffusiongemma-26B-A4B-it-{precision}"
    assert len(profile.artifacts) == count
    assert sum(artifact.size for artifact in profile.artifacts) == total
    assert profile.layout == "mlx"
    assert profile.quantization_bits == (8 if precision == "8bit" else None)


@pytest.mark.parametrize("precision", ["optiq4", "8bit", "bf16"])
def test_profiles_have_distinct_safe_artifact_names_and_sha256(precision: Precision) -> None:
    profile = CHECKPOINTS[precision]
    assert re.fullmatch(r"[0-9a-f]{40}", profile.revision)
    assert len({artifact.name for artifact in profile.artifacts}) == len(profile.artifacts)
    for artifact in profile.artifacts:
        assert artifact.size > 0
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        assert not artifact.name.startswith("/")
        assert ".." not in artifact.name.split("/")
    names = {artifact.name for artifact in profile.artifacts}
    assert {"config.json", "tokenizer.json", "model.safetensors.index.json"} <= names
    assert ("optiq_metadata.json" in names) == (precision == "optiq4")
