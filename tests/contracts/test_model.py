"""
Model -> evaluation boundary: whatever the baseline is, it ends up as Predictions.

forward() and the 6D rotation representation are internals of the current
baseline, not contracts; the tests below pin how train.py and infer.py
currently turn them into Predictions.
"""

import numpy as np
import pytest
import torch

from sfm_rotations.contracts import ContractError
from sfm_rotations.contracts import Predictions
from sfm_rotations.contracts import is_rotation
from sfm_rotations.losses import rotation_6d_to_matrix
from sfm_rotations.model import SfMModel
from tests.contracts.synthetic import random_rotations


@pytest.fixture
def images() -> torch.Tensor:
    return torch.randn(3, 3, 224, 224, generator=torch.Generator().manual_seed(0))


@pytest.mark.usefixtures("tiny_backbone")
def test_baseline_outputs_convert_to_predictions(images: torch.Tensor):
    """Так выход модели превращают в позы collect_outputs (train.py) и infer.py."""
    model = SfMModel(embedding_dim=16).eval()
    with torch.no_grad():
        embeddings, translations, rotations_6d = model(images)

    predictions = Predictions(
        embeddings=embeddings.numpy(),
        rotations=rotation_6d_to_matrix(rotations_6d).numpy(),
        translations=translations.numpy(),
    )
    predictions.validate()
    assert predictions.embeddings.shape == (3, 16)


@pytest.mark.xfail(
    reason="#6: у модели нет predict() -> Predictions, 6D протекает в train.py и infer.py",
    raises=AttributeError,
)
@pytest.mark.usefixtures("tiny_backbone")
def test_model_returns_predictions(images: torch.Tensor):
    model = SfMModel(embedding_dim=16).eval()
    predictions = model.predict(images)  # type: ignore[attr-defined]
    assert isinstance(predictions, Predictions)
    predictions.validate()


def test_rotation_6d_gives_proper_rotations():
    rotations = rotation_6d_to_matrix(torch.randn(1000, 6, generator=torch.Generator().manual_seed(0)))
    assert is_rotation(rotations.numpy(), atol=1e-5).all()


def test_rotation_6d_is_first_two_columns(rng: np.random.Generator):
    rotations = random_rotations(rng, 10)
    rotations_6d = np.concatenate([rotations[:, :, 0], rotations[:, :, 1]], axis=1)
    restored = rotation_6d_to_matrix(torch.from_numpy(rotations_6d)).numpy()
    np.testing.assert_allclose(restored, rotations, atol=1e-6)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rotations", np.tile(np.diag([1.0, 1.0, -1.0]), (4, 1, 1)), "не являются поворотами"),
        ("rotations", np.tile(np.eye(3), (3, 1, 1)), r"ожидалось \(4, 3, 3\)"),
        ("translations", np.full((4, 3), np.nan), "NaN"),
        ("embeddings", np.zeros(4), r"\[N, D\]"),
    ],
)
def test_predictions_validate_rejects(field: str, value: np.ndarray, message: str):
    valid = {
        "embeddings": np.zeros((4, 8)),
        "rotations": np.tile(np.eye(3), (4, 1, 1)),
        "translations": np.zeros((4, 3)),
    }
    Predictions(**valid).validate()
    with pytest.raises(ContractError, match=message):
        Predictions(**(valid | {field: value})).validate()
