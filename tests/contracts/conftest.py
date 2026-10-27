from pathlib import Path

import numpy as np
import pytest

from sfm_rotations import model
from sfm_rotations.contracts import PROTOCOL_PATH
from sfm_rotations.contracts import Protocol
from sfm_rotations.contracts import ValSplit
from sfm_rotations.contracts import load_protocol
from sfm_rotations.contracts import load_val_split
from tests.contracts.synthetic import TinyAutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def protocol() -> Protocol:
    return load_protocol(REPO_ROOT / PROTOCOL_PATH)


@pytest.fixture(scope="session")
def val_split(protocol: Protocol) -> ValSplit:
    return load_val_split(REPO_ROOT / protocol.split_path)


@pytest.fixture(scope="session")
def real_data_root() -> Path:
    """Real competition data; tests that need it are skipped until `dvc pull` (and always in CI)."""
    data_root = REPO_ROOT / "data"
    if not (data_root / "train_labels.csv").exists():
        pytest.skip("нет data/train_labels.csv: сделайте dvc pull")
    return data_root


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build SfMModel on a tiny random DINOv2 instead of downloading facebook/dinov2-base."""
    monkeypatch.setattr(model, "AutoModel", TinyAutoModel)
