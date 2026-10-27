"""contracts/protocol.yaml and contracts/val_split.json agree with each other, with DVC and with the code."""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from sfm_rotations import metrics
from sfm_rotations.contracts import PROTOCOL_VERSION
from sfm_rotations.contracts import REQUIRED_METRIC_KEYS
from sfm_rotations.contracts import Protocol
from sfm_rotations.contracts import ValSplit
from sfm_rotations.contracts import file_md5
from sfm_rotations.data import load_labels
from sfm_rotations.data import train_val_split_by_cluster


def test_protocol_version_matches_code(protocol: Protocol):
    assert protocol.version == PROTOCOL_VERSION


def test_data_version_matches_dvc(protocol: Protocol, repo_root: Path):
    for name, md5 in protocol.data.items():
        dvc = OmegaConf.load(repo_root / "data" / f"{name}.dvc")
        assert dvc.outs[0].md5 == md5, f"data/{name}.dvc изменился: новые данные — новая версия протокола"


def test_rotation_thresholds_match_metrics(protocol: Protocol):
    assert protocol.rotation_thresholds_deg == metrics.ROTATION_THRESHOLDS_DEG


def test_primary_metric_is_required(protocol: Protocol):
    assert protocol.primary_metric in REQUIRED_METRIC_KEYS


def test_val_split_matches_protocol(protocol: Protocol, val_split: ValSplit):
    assert val_split.protocol_version == protocol.version
    assert val_split.labels_md5 == protocol.data["train_labels.csv"]
    assert val_split.seed == protocol.split_seed
    assert val_split.val_fraction == protocol.split_val_fraction
    assert val_split.scenes


# ---------------------------------------------------------------------------
# Нужны настоящие данные (dvc pull); в CI пропускаются
# ---------------------------------------------------------------------------


def test_labels_match_protocol(protocol: Protocol, real_data_root: Path):
    assert file_md5(real_data_root / "train_labels.csv") == protocol.data["train_labels.csv"]


def test_val_split_entries_exist(val_split: ValSplit, real_data_root: Path):
    labels = pd.read_csv(real_data_root / "train_labels.csv")
    assert val_split.scenes <= set(zip(labels["dataset"], labels["scene"], strict=True))
    assert val_split.outliers <= set(zip(labels["dataset"], labels["image"], strict=True))


def test_val_split_matches_baseline_split(protocol: Protocol, val_split: ValSplit, real_data_root: Path):
    """Пока train.py считает сплит сам (расхождение #5), он обязан совпадать с замороженным."""
    df = load_labels(real_data_root)
    train_indices, val_indices = train_val_split_by_cluster(
        df["scene_label"].tolist(),
        val_fraction=protocol.split_val_fraction,
        seed=protocol.split_seed,
    )
    frozen_train, frozen_val = val_split.split(df)
    assert sorted(val_indices) == frozen_val
    assert sorted(train_indices) == frozen_train
