"""Artifact validators: submission.csv, checkpoint, metrics."""

import numpy as np
import pandas as pd
import pytest

from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import PROTOCOL_VERSION
from sfm_rotations.contracts import REQUIRED_METRIC_KEYS
from sfm_rotations.contracts import ContractError
from sfm_rotations.contracts import validate_checkpoint
from sfm_rotations.contracts import validate_metrics
from sfm_rotations.contracts import validate_submission
from tests.contracts.synthetic import format_vector

IDENTITY = format_vector(np.eye(3))
ZERO = format_vector(np.zeros(3))
NAN_R = format_vector(np.full(9, np.nan))
NAN_T = format_vector(np.full(3, np.nan))


def _submission(*rows: tuple[str, str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])


@pytest.fixture
def submission() -> pd.DataFrame:
    """Пример из docs/overview.md: зарегистрированные, сгруппированные без позы и выбросы."""
    return _submission(
        ("dataset1", "cluster1", "image1.png", IDENTITY, ZERO),
        ("dataset1", "cluster1", "image2.png", IDENTITY, ZERO),
        ("dataset1", "cluster2", "image3.png", NAN_R, NAN_T),
        ("dataset1", OUTLIER_SCENE, "image5.png", NAN_R, NAN_T),
        ("dataset2", "cluster1", "image1.png", IDENTITY, ZERO),
    )


def test_validate_submission_accepts_competition_format(submission: pd.DataFrame, tmp_path):
    path = tmp_path / "submission.csv"
    submission.to_csv(path, index=False)
    validate_submission(pd.read_csv(path))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("dataset1", OUTLIER_SCENE, "image6.png", IDENTITY, ZERO), "у выбросов поза должна быть nan"),
        (("dataset1", "cluster1", "image6.png", IDENTITY, NAN_T), "целиком"),
        (("dataset1", "cluster1", "image6.png", format_vector(2 * np.eye(3)), ZERO), "не являются поворотами"),
        (("dataset1", "cluster1", "image1.png", IDENTITY, ZERO), "повтор"),
    ],
)
def test_validate_submission_rejects(submission: pd.DataFrame, row: tuple, message: str):
    with pytest.raises(ContractError, match=message):
        validate_submission(pd.concat([submission, _submission(row)], ignore_index=True))


@pytest.fixture
def metrics() -> dict[str, float]:
    return dict.fromkeys(REQUIRED_METRIC_KEYS, 0.5) | {"imc/ds_a/score": 0.5}


@pytest.fixture
def checkpoint(metrics: dict[str, float]) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_config": {"backbone": "facebook/dinov2-base", "embedding_dim": 256, "image_size": 224},
        "postprocess": {"cluster_distance": 0.3, "min_cluster_size": 3},
        "model": {},
        "epoch": 1,
        "metrics": metrics,
    }


def test_validate_metrics(metrics: dict[str, float]):
    validate_metrics(metrics | {"extra/metric": 7.0})
    with pytest.raises(ContractError, match="нет ключей"):
        validate_metrics({key: value for key, value in metrics.items() if key != "imc/score"})
    with pytest.raises(ContractError, match=r"вне \[0, 1\]"):
        validate_metrics(metrics | {"imc/ds_a/score": 1.5})


def test_validate_checkpoint(checkpoint: dict):
    validate_checkpoint(checkpoint)
    with pytest.raises(ContractError, match="нет ключей"):
        validate_checkpoint({key: value for key, value in checkpoint.items() if key != "model_config"})
    with pytest.raises(ContractError, match="протокол"):
        validate_checkpoint(checkpoint | {"protocol_version": PROTOCOL_VERSION + 1})
    with pytest.raises(ContractError, match="image_size"):
        validate_checkpoint(checkpoint | {"model_config": {"embedding_dim": 256}})
    with pytest.raises(ContractError, match="postprocess"):
        validate_checkpoint(checkpoint | {"postprocess": {}})
    with pytest.raises(ContractError, match="не int/float Python"):
        validate_checkpoint(checkpoint | {"metrics": checkpoint["metrics"] | {"imc/ds_a/score": np.float64(0.5)}})
