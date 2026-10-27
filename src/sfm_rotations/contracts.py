"""
Contracts of the SfM-Rotations pipeline.

Single source of truth for what any baseline has to keep. Protocol parameters
live in contracts/protocol.yaml and contracts/val_split.json, the rules for
changing them in contracts/PROTOCOL.md.

Data:
    - outlier and pose conventions, CSV schemas.

Model -> evaluation boundary:
    - Predictions: what a model hands to metrics and inference.

Artifacts:
    - checkpoint, metrics json, submission.csv.

Regenerate the frozen validation split (only together with a protocol version bump):
    python -m sfm_rotations.contracts
"""

import hashlib
import json
import re
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TypedDict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Должна совпадать с version в contracts/protocol.yaml (проверяется тестом).
PROTOCOL_VERSION = 1
PROTOCOL_PATH = Path("contracts/protocol.yaml")

# Выбросы: scene == "outliers" в CSV, scene_label == -1 в коде, позы — NaN.
OUTLIER_SCENE = "outliers"
OUTLIER_LABEL = -1

# Поза камеры — world -> camera: x_cam = R @ x_world + t, центр камеры c = -Rᵀt.
# В CSV R записана построчно (row-major), компоненты R и t разделены ";".
VALUE_SEPARATOR = ";"

LABELS_COLUMNS = ("dataset", "scene", "image", "rotation_matrix", "translation_vector")
THRESHOLDS_COLUMNS = ("dataset", "scene", "thresholds")
SUBMISSION_COLUMNS = ("dataset", "scene", "image", "rotation_matrix", "translation_vector")

# Ключи, которые обязан содержать metrics_epochNNN.json и checkpoint["metrics"].
REQUIRED_METRIC_KEYS = (
    "epoch",
    "train/loss",
    "val/loss",
    "pose/rel_rot_err_median_deg",
    "pose/rel_rot_mAA",
    "emb/ARI",
    "emb/recall@1",
    "imc/mAA",
    "imc/clustering",
    "imc/score",
)
PER_DATASET_METRIC_KEY = re.compile(r"imc/[^/]+/(mAA|clustering|score)")

CHECKPOINT_KEYS = ("protocol_version", "model_config", "postprocess", "model", "epoch", "metrics")
POSTPROCESS_KEYS = ("cluster_distance", "min_cluster_size")


class ContractError(ValueError):
    """Data or an artifact violates a contract."""


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def is_rotation(matrices: np.ndarray, atol: float = 1e-3) -> np.ndarray:
    """
    Check that matrices are proper rotations: RᵀR = I and det R = +1.

    Parameters
    ----------
    matrices:
        Matrices [..., 3, 3].

    atol:
        Absolute tolerance.

    Returns
    -------
    bool mask [...], False for matrices with NaN

    """
    gram = np.swapaxes(matrices, -1, -2) @ matrices
    orthogonal = np.abs(gram - np.eye(3)).max(axis=(-2, -1)) <= atol
    proper = np.abs(np.linalg.det(matrices) - 1.0) <= atol
    return orthogonal & proper


def parse_vectors(values: pd.Series, size: int) -> np.ndarray:
    """
    Parse ";"-separated vectors, as stored in the competition CSV files.

    Parameters
    ----------
    values:
        Column with one vector per row.

    size:
        Expected number of components.

    Returns
    -------
    float array [N, size]; "nan" components become NaN

    """
    rows = []
    for row, value in enumerate(values):
        try:
            vector = np.array(str(value).split(VALUE_SEPARATOR), dtype=float)
        except ValueError as e:
            raise ContractError(f"{values.name}, строка {row}: не число ({value!r})") from e
        if vector.shape != (size,):
            raise ContractError(
                f"{values.name}, строка {row}: ожидалось {size} чисел через ';', получено {vector.size}",
            )
        rows.append(vector)
    return np.stack(rows) if rows else np.empty((0, size))


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ContractError(f"{name}: нет колонок {missing}")


def _require_unique(df: pd.DataFrame, columns: list[str], name: str) -> None:
    duplicated = df.duplicated(subset=columns)
    if duplicated.any():
        raise ContractError(f"{name}: {int(duplicated.sum())} повторов по {columns}")


def validate_labels(df: pd.DataFrame) -> None:
    """
    Check train_labels.csv as read by pd.read_csv.

    Parameters
    ----------
    df:
        Raw train_labels.csv.

    """
    _require_columns(df, LABELS_COLUMNS, "train_labels.csv")
    _require_unique(df, ["dataset", "image"], "train_labels.csv")

    rotations = parse_vectors(df["rotation_matrix"], 9).reshape(-1, 3, 3)
    translations = parse_vectors(df["translation_vector"], 3)
    is_outlier = (df["scene"] == OUTLIER_SCENE).to_numpy()

    if not (np.isnan(rotations[is_outlier]).all() and np.isnan(translations[is_outlier]).all()):
        raise ContractError("train_labels.csv: у выбросов поза должна быть NaN")
    if not np.isfinite(translations[~is_outlier]).all():
        raise ContractError("train_labels.csv: у изображений сцен translation_vector должен быть конечным")
    not_rotation = ~is_rotation(rotations[~is_outlier])
    if not_rotation.any():
        raise ContractError(f"train_labels.csv: {int(not_rotation.sum())} rotation_matrix не являются поворотами")


def validate_thresholds(df: pd.DataFrame, labels: pd.DataFrame | None = None) -> None:
    """
    Check train_thresholds.csv as read by pd.read_csv.

    Parameters
    ----------
    df:
        Raw train_thresholds.csv.

    labels:
        Raw train_labels.csv; if given, every scene must have thresholds.

    """
    _require_columns(df, THRESHOLDS_COLUMNS, "train_thresholds.csv")
    _require_unique(df, ["dataset", "scene"], "train_thresholds.csv")

    for row, value in enumerate(df["thresholds"]):
        thresholds = np.array(str(value).split(VALUE_SEPARATOR), dtype=float)
        # Нулевые пороги в данных есть (imc2023_haiper): такой порог просто никогда не выполняется.
        if not (np.isfinite(thresholds).all() and (thresholds >= 0).all()):
            raise ContractError(f"train_thresholds.csv, строка {row}: пороги должны быть неотрицательными числами")

    if labels is not None:
        scenes = labels.loc[labels["scene"] != OUTLIER_SCENE, ["dataset", "scene"]].drop_duplicates()
        known = set(zip(df["dataset"], df["scene"], strict=True))
        missing = [key for key in zip(scenes["dataset"], scenes["scene"], strict=True) if key not in known]
        if missing:
            raise ContractError(f"train_thresholds.csv: нет порогов для сцен {missing[:5]}")


def validate_submission(df: pd.DataFrame) -> None:
    """
    Check submission.csv in the competition format.

    An image may be grouped into a scene without being registered: then its
    pose is all NaN. Outliers are never registered.

    Parameters
    ----------
    df:
        submission.csv read with pd.read_csv.

    """
    _require_columns(df, SUBMISSION_COLUMNS, "submission.csv")
    _require_unique(df, ["dataset", "image"], "submission.csv")

    rotations = parse_vectors(df["rotation_matrix"], 9).reshape(-1, 3, 3)
    translations = parse_vectors(df["translation_vector"], 3)
    registered = np.isfinite(rotations).all(axis=(1, 2)) & np.isfinite(translations).all(axis=1)
    unregistered = np.isnan(rotations).all(axis=(1, 2)) & np.isnan(translations).all(axis=1)

    if not (registered | unregistered).all():
        raise ContractError("submission.csv: поза должна быть либо целиком числами, либо целиком nan")
    if (registered & (df["scene"] == OUTLIER_SCENE).to_numpy()).any():
        raise ContractError("submission.csv: у выбросов поза должна быть nan")
    not_rotation = ~is_rotation(rotations[registered])
    if not_rotation.any():
        raise ContractError(f"submission.csv: {int(not_rotation.sum())} rotation_matrix не являются поворотами")


# ---------------------------------------------------------------------------
# Граница модель -> оценка
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predictions:
    """
    What a model hands to evaluation and inference, one row per image.

    Any baseline must produce this. How it parametrizes rotations internally
    (6D, quaternions, ...) is its own business.

    Parameters
    ----------
    embeddings:
        Scene embeddings [N, D], any norm: clustering normalizes them.

    rotations:
        World -> camera rotations [N, 3, 3], proper rotation matrices.

    translations:
        World -> camera translations [N, 3].

    """

    embeddings: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray

    def validate(self) -> None:
        """Check shapes, finiteness and that rotations are proper rotation matrices."""
        n = len(self.embeddings)
        if self.embeddings.ndim != 2:
            raise ContractError(f"embeddings: ожидалось [N, D], получено {self.embeddings.shape}")
        if self.rotations.shape != (n, 3, 3):
            raise ContractError(f"rotations: ожидалось ({n}, 3, 3), получено {self.rotations.shape}")
        if self.translations.shape != (n, 3):
            raise ContractError(f"translations: ожидалось ({n}, 3), получено {self.translations.shape}")
        for name in ("embeddings", "rotations", "translations"):
            if not np.isfinite(getattr(self, name)).all():
                raise ContractError(f"{name}: есть NaN или inf")
        not_rotation = ~is_rotation(self.rotations)
        if not_rotation.any():
            raise ContractError(f"rotations: {int(not_rotation.sum())} матриц не являются поворотами")


# ---------------------------------------------------------------------------
# Артефакты
# ---------------------------------------------------------------------------


class Checkpoint(TypedDict):
    """
    Layout of a checkpoint file (torch.save of this dict).

    Only tensors and plain Python types: the file must load with
    torch.load(weights_only=True), the default since torch 2.6.
    """

    protocol_version: int  # версия протокола, на которой посчитаны metrics
    model_config: dict[str, Any]  # всё, из чего пересобирается модель; обязательно image_size
    postprocess: dict[str, Any]  # cluster_distance, min_cluster_size: инференс берёт их отсюда
    model: dict[str, Any]  # state_dict
    epoch: int
    metrics: dict[str, float]


def validate_metrics(metrics: Mapping[str, Any]) -> None:
    """
    Check an epoch metrics dict (metrics_epochNNN.json, checkpoint["metrics"]).

    Extra keys are allowed: a baseline may log more than the protocol requires.

    Parameters
    ----------
    metrics:
        Metrics dict.

    """
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        raise ContractError(f"metrics: нет ключей {missing}")
    not_numbers = [key for key, value in metrics.items() if not isinstance(value, int | float)]
    if not_numbers:
        raise ContractError(f"metrics: не числа {not_numbers}")
    imc_keys = ["imc/mAA", "imc/clustering", "imc/score"]
    imc_keys += [key for key in metrics if PER_DATASET_METRIC_KEY.fullmatch(key)]
    out_of_range = [key for key in imc_keys if not 0.0 <= metrics[key] <= 1.0]
    if out_of_range:
        raise ContractError(f"metrics: вне [0, 1] {out_of_range}")


def validate_checkpoint(checkpoint: Mapping[str, Any], protocol_version: int = PROTOCOL_VERSION) -> None:
    """
    Check a loaded checkpoint against the Checkpoint layout.

    Parameters
    ----------
    checkpoint:
        Result of torch.load.

    protocol_version:
        Protocol version the checkpoint metrics must be computed with.

    """
    missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ContractError(f"checkpoint: нет ключей {missing}")
    if checkpoint["protocol_version"] != protocol_version:
        raise ContractError(
            f"checkpoint: протокол v{checkpoint['protocol_version']}, ожидался v{protocol_version}",
        )
    if "image_size" not in checkpoint["model_config"]:
        raise ContractError("checkpoint: в model_config нет image_size")
    missing = [key for key in POSTPROCESS_KEYS if key not in checkpoint["postprocess"]]
    if missing:
        raise ContractError(f"checkpoint: в postprocess нет ключей {missing}")
    validate_metrics(checkpoint["metrics"])
    # numpy-скаляры torch.load(weights_only=True) не загрузит.
    not_plain = [key for key, value in checkpoint["metrics"].items() if type(value) not in (int, float)]
    if not_plain:
        raise ContractError(f"checkpoint: в metrics не int/float Python {not_plain}")


# ---------------------------------------------------------------------------
# Протокол оценки
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    """
    Evaluation protocol from contracts/protocol.yaml.

    Parameters
    ----------
    version:
        Protocol version; runs with different versions are not comparable.

    data:
        Path inside data_root -> md5 of its DVC output (data/<path>.dvc).

    split_path:
        Frozen validation split file.

    split_seed:
        Seed the split was generated with.

    split_val_fraction:
        Validation fraction the split was generated with.

    primary_metric:
        Metric runs are compared and best.pt is selected by.

    rotation_thresholds_deg:
        Thresholds of pose/rel_rot_mAA.

    """

    version: int
    data: dict[str, str]
    split_path: Path
    split_seed: int
    split_val_fraction: float
    primary_metric: str
    rotation_thresholds_deg: tuple[float, ...]


def load_protocol(path: Path = PROTOCOL_PATH) -> Protocol:
    """
    Load contracts/protocol.yaml.

    Parameters
    ----------
    path:
        Protocol file.

    Returns
    -------
    Protocol

    """
    raw = OmegaConf.to_container(OmegaConf.load(path))
    if not isinstance(raw, dict):
        raise ContractError(f"{path}: ожидался словарь")
    return Protocol(
        version=int(raw["version"]),
        data={str(name): str(md5) for name, md5 in raw["data"].items()},
        split_path=Path(raw["split"]["path"]),
        split_seed=int(raw["split"]["seed"]),
        split_val_fraction=float(raw["split"]["val_fraction"]),
        primary_metric=str(raw["metrics"]["primary"]),
        rotation_thresholds_deg=tuple(float(t) for t in raw["metrics"]["rotation_thresholds_deg"]),
    )


@dataclass(frozen=True)
class ValSplit:
    """
    Frozen validation split from contracts/val_split.json.

    Parameters
    ----------
    protocol_version:
        Protocol version the split belongs to.

    labels_md5:
        md5 of the train_labels.csv the split was generated from.

    seed:
        Seed the split was generated with.

    val_fraction:
        Validation fraction the split was generated with.

    scenes:
        (dataset, scene) pairs that go to validation entirely.

    outliers:
        (dataset, image) pairs of outlier images that go to validation.

    """

    protocol_version: int
    labels_md5: str
    seed: int
    val_fraction: float
    scenes: frozenset[tuple[str, str]]
    outliers: frozenset[tuple[str, str]]

    def val_mask(self, labels: pd.DataFrame) -> np.ndarray:
        """
        Mark validation rows of train_labels.csv (or of load_labels output).

        Parameters
        ----------
        labels:
            Dataframe with dataset, scene and image columns.

        Returns
        -------
        bool mask [N]

        """
        # Сцены уходят в val целиком, выбросы — поштучно.
        is_val = []
        for dataset, scene, image in zip(labels["dataset"], labels["scene"], labels["image"], strict=True):
            is_val.append(
                (dataset, image) in self.outliers if scene == OUTLIER_SCENE else (dataset, scene) in self.scenes,
            )
        return np.array(is_val, dtype=bool)

    def split(self, labels: pd.DataFrame) -> tuple[list[int], list[int]]:
        """
        Split rows of train_labels.csv (or of load_labels output) into train/val.

        Parameters
        ----------
        labels:
            Dataframe with dataset, scene and image columns.

        Returns
        -------
        train_indices, val_indices (row positions, sorted)

        """
        mask = self.val_mask(labels)
        return np.flatnonzero(~mask).tolist(), np.flatnonzero(mask).tolist()


def load_val_split(path: Path) -> ValSplit:
    """
    Load the frozen validation split.

    Parameters
    ----------
    path:
        contracts/val_split.json.

    Returns
    -------
    ValSplit

    """
    with open(path) as f:
        raw = json.load(f)
    return ValSplit(
        protocol_version=int(raw["protocol_version"]),
        labels_md5=raw["labels_md5"],
        seed=int(raw["seed"]),
        val_fraction=float(raw["val_fraction"]),
        scenes=frozenset((item["dataset"], item["scene"]) for item in raw["scenes"]),
        outliers=frozenset((item["dataset"], item["image"]) for item in raw["outliers"]),
    )


def file_md5(path: Path) -> str:
    """
    md5 of a file, the same as DVC stores in .dvc files.

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    hex digest

    """
    digest = hashlib.md5()  # noqa: S324 — не для безопасности, сверяем с DVC
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_val_split(data_root: Path, protocol: Protocol) -> dict[str, Any]:
    """
    Build the val_split.json payload with the baseline split function.

    Refuses to run on labels that differ from the data version in the protocol.

    Parameters
    ----------
    data_root:
        Folder with train_labels.csv (pulled via DVC).

    protocol:
        Protocol the split is generated for.

    Returns
    -------
    json-serializable split

    """
    # Локальный импорт: data.py сам импортирует константы отсюда.
    from sfm_rotations.data import load_labels
    from sfm_rotations.data import train_val_split_by_cluster

    labels_md5 = file_md5(data_root / "train_labels.csv")
    if labels_md5 != protocol.data["train_labels.csv"]:
        raise ContractError(
            f"train_labels.csv (md5 {labels_md5}) не совпадает с версией данных протокола; сделайте dvc pull",
        )

    df = load_labels(data_root)
    _, val_indices = train_val_split_by_cluster(
        df["scene_label"].tolist(),
        val_fraction=protocol.split_val_fraction,
        seed=protocol.split_seed,
    )
    val = df.iloc[sorted(val_indices)]
    is_outlier = val["scene"] == OUTLIER_SCENE
    scenes = sorted(set(zip(val.loc[~is_outlier, "dataset"], val.loc[~is_outlier, "scene"], strict=True)))
    outliers = sorted(zip(val.loc[is_outlier, "dataset"], val.loc[is_outlier, "image"], strict=True))
    return {
        "protocol_version": protocol.version,
        "labels_md5": labels_md5,
        "seed": protocol.split_seed,
        "val_fraction": protocol.split_val_fraction,
        "scenes": [{"dataset": dataset, "scene": scene} for dataset, scene in scenes],
        "outliers": [{"dataset": dataset, "image": image} for dataset, image in outliers],
    }


if __name__ == "__main__":
    protocol = load_protocol()
    payload = freeze_val_split(Path("data"), protocol)
    with open(protocol.split_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    n_scenes, n_outliers = len(payload["scenes"]), len(payload["outliers"])
    print(f"Saved {n_scenes} val scenes and {n_outliers} val outliers to {protocol.split_path}")
