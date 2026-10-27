"""Data conventions: CSV schemas, outliers, pose convention."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfm_rotations.contracts import OUTLIER_LABEL
from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import ContractError
from sfm_rotations.contracts import is_rotation
from sfm_rotations.contracts import parse_vectors
from sfm_rotations.contracts import validate_labels
from sfm_rotations.contracts import validate_thresholds
from sfm_rotations.data import SceneDataset
from sfm_rotations.data import build_transforms
from sfm_rotations.data import load_labels
from sfm_rotations.metrics import camera_centers
from tests.contracts.synthetic import random_rotations
from tests.contracts.synthetic import write_competition_data


@pytest.fixture
def data_root(tmp_path: Path, rng: np.random.Generator) -> Path:
    write_competition_data(tmp_path, rng, {"ds_a": 2, "ds_b": 2})
    return tmp_path


def test_synthetic_data_follows_contract(data_root: Path):
    labels = pd.read_csv(data_root / "train_labels.csv")
    validate_labels(labels)
    validate_thresholds(pd.read_csv(data_root / "train_thresholds.csv"), labels)


def test_load_labels_follows_contract(data_root: Path):
    df = load_labels(data_root)
    is_outlier = (df["scene"] == OUTLIER_SCENE).to_numpy()

    assert (df.loc[is_outlier, "scene_label"] == OUTLIER_LABEL).all()
    # Сцена — пара (dataset, scene): одноимённые сцены разных датасетов получают разные scene_label.
    scenes = df.loc[~is_outlier, ["dataset", "scene", "scene_label"]].drop_duplicates()
    assert len(scenes) == scenes["scene_label"].nunique() == 4

    rotations = np.stack(df["rotation_matrix"].to_list()).reshape(-1, 3, 3)
    assert is_rotation(rotations[~is_outlier]).all()
    assert np.isnan(rotations[is_outlier]).all()
    assert all(Path(path).is_file() for path in df["image_path"])


def test_rotation_is_read_row_major(data_root: Path):
    raw = pd.read_csv(data_root / "train_labels.csv")
    expected = parse_vectors(raw["rotation_matrix"], 9).reshape(-1, 3, 3)[0]

    _, val_transform = build_transforms()
    _, labels = SceneDataset(load_labels(data_root), transform=val_transform)[0]
    np.testing.assert_allclose(labels["rotation_matrix"].numpy(), expected, atol=1e-6)


@pytest.mark.parametrize(
    ("column", "value", "outlier", "message"),
    [
        ("rotation_matrix", "1;0;0;0;1;0;0;0;2", False, "не являются поворотами"),
        ("rotation_matrix", "1;0;0", False, "ожидалось 9 чисел"),
        ("translation_vector", "a;b;c", False, "не число"),
        ("translation_vector", "nan;nan;nan", False, "должен быть конечным"),
        ("translation_vector", "0;0;0", True, "у выбросов поза должна быть NaN"),
    ],
)
def test_validate_labels_rejects_broken_rows(data_root: Path, column: str, value: str, outlier: bool, message: str):
    labels = pd.read_csv(data_root / "train_labels.csv")
    row = labels.index[labels["scene"].eq(OUTLIER_SCENE) == outlier][0]
    labels.loc[row, column] = value
    with pytest.raises(ContractError, match=message):
        validate_labels(labels)


def test_validate_labels_rejects_duplicates_and_missing_columns(data_root: Path):
    labels = pd.read_csv(data_root / "train_labels.csv")
    with pytest.raises(ContractError, match="повтор"):
        validate_labels(pd.concat([labels, labels.iloc[:1]]))
    with pytest.raises(ContractError, match="нет колонок"):
        validate_labels(labels.drop(columns="scene"))


def test_validate_thresholds_requires_every_scene(data_root: Path):
    labels = pd.read_csv(data_root / "train_labels.csv")
    thresholds = pd.read_csv(data_root / "train_thresholds.csv")
    with pytest.raises(ContractError, match="нет порогов"):
        validate_thresholds(thresholds.iloc[1:], labels)


def test_camera_center_convention(rng: np.random.Generator):
    """Поза world -> camera: центр камеры c = -Rᵀt переходит в начало координат камеры."""
    rotations = random_rotations(rng, 10)
    translations = rng.normal(size=(10, 3))
    centers = camera_centers(rotations, translations)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", rotations, centers) + translations, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Нужны настоящие данные (dvc pull); в CI пропускаются
# ---------------------------------------------------------------------------


def test_real_data_follows_contract(real_data_root: Path):
    labels = pd.read_csv(real_data_root / "train_labels.csv")
    validate_labels(labels)
    validate_thresholds(pd.read_csv(real_data_root / "train_thresholds.csv"), labels)
