"""Synthetic data for contract tests: no DVC, GPU or model download needed."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from omegaconf import OmegaConf
from PIL import Image
from transformers import Dinov2Config
from transformers import Dinov2Model

from sfm_rotations.contracts import OUTLIER_LABEL
from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import VALUE_SEPARATOR
from sfm_rotations.metrics import EvalOutputs

THRESHOLDS = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0)


class TinyAutoModel:
    """Offline stand-in for transformers.AutoModel: a one-layer DINOv2."""

    @staticmethod
    def from_pretrained(name: str) -> Dinov2Model:
        config = Dinov2Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            mlp_ratio=2,
            image_size=224,
            patch_size=14,
        )
        return Dinov2Model(config)


def load_config(repo_root: Path, name: str, **overrides: object) -> DictConfig:
    """Load configs/<name>.yaml the way Hydra would, with overrides on top."""
    cfg = OmegaConf.merge(OmegaConf.load(repo_root / "configs" / f"{name}.yaml"), overrides)
    assert isinstance(cfg, DictConfig)
    return cfg


# ---------------------------------------------------------------------------
# Позы
# ---------------------------------------------------------------------------


def random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random proper rotations [n, 3, 3]."""
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[:, None, :]
    q[np.linalg.det(q) < 0, :, 0] *= -1
    return q


def perturb_rotations(rng: np.random.Generator, rotations: np.ndarray, max_deg: float) -> np.ndarray:
    """Rotate each matrix by a random angle up to max_deg around a random axis."""
    n = len(rotations)
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.radians(rng.uniform(0.0, max_deg, size=n))[:, None, None]
    skew = np.zeros((n, 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    skew -= np.swapaxes(skew, -1, -2)
    small = np.eye(3) + np.sin(angles) * skew + (1.0 - np.cos(angles)) * skew @ skew
    return rotations @ small


def transform_world(
    rotations: np.ndarray,
    translations: np.ndarray,
    rotation: np.ndarray,
    shift: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-express world -> camera poses in a world frame moved by x' = scale * rotation @ x + shift.

    Camera centers move the same way: c' = scale * rotation @ c + shift.
    """
    new_rotations = rotations @ rotation.T
    new_translations = scale * translations - np.einsum("nij,j->ni", new_rotations, shift)
    return new_rotations, new_translations


# ---------------------------------------------------------------------------
# Выходы модели для метрик
# ---------------------------------------------------------------------------


def make_eval_outputs(
    rng: np.random.Generator,
    datasets: dict[str, int],
    images_per_scene: int = 5,
    outliers_per_dataset: int = 2,
    rotation_noise_deg: float = 0.0,
    translation_noise: float = 0.0,
) -> tuple[EvalOutputs, dict[tuple[str, str], np.ndarray]]:
    """
    Validation outputs where every scene and every outlier has its own embedding direction.

    Clusters are therefore perfect; poses equal GT unless noise is requested.

    Returns
    -------
    outputs, thresholds (dataset, scene) -> thresholds

    """
    rows = []
    n_scenes = 0
    for dataset, scenes in datasets.items():
        for s in range(scenes):
            rows += [(dataset, f"scene{s}", n_scenes)] * images_per_scene
            n_scenes += 1
        rows += [(dataset, OUTLIER_SCENE, OUTLIER_LABEL)] * outliers_per_dataset

    dataset_names = np.array([row[0] for row in rows])
    scene_names = np.array([row[1] for row in rows])
    labels = np.array([row[2] for row in rows])
    is_outlier = labels == OUTLIER_LABEL

    directions = labels.copy()
    directions[is_outlier] = n_scenes + np.arange(is_outlier.sum())
    dim = int(directions.max()) + 1
    embeddings = np.eye(dim)[directions] + rng.normal(scale=0.01, size=(len(rows), dim))

    rotations_gt = random_rotations(rng, len(rows))
    translations_gt = rng.normal(scale=3.0, size=(len(rows), 3))
    rotations_pred = perturb_rotations(rng, rotations_gt, rotation_noise_deg)
    translations_pred = translations_gt + rng.normal(scale=translation_noise, size=(len(rows), 3))
    rotations_gt[is_outlier] = np.nan
    translations_gt[is_outlier] = np.nan

    outputs = EvalOutputs(
        datasets=dataset_names,
        scenes=scene_names,
        scene_labels=labels,
        embeddings=embeddings.astype(np.float32),
        rotations_pred=rotations_pred,
        translations_pred=translations_pred,
        rotations_gt=rotations_gt,
        translations_gt=translations_gt,
    )
    thresholds = {
        (dataset, scene): np.array(THRESHOLDS)
        for dataset, scene in set(zip(dataset_names, scene_names, strict=True))
        if scene != OUTLIER_SCENE
    }
    return outputs, thresholds


# ---------------------------------------------------------------------------
# Данные соревнования на диске
# ---------------------------------------------------------------------------


def format_vector(values: ArrayLike) -> str:
    """Write a vector the competition way: ";"-separated, row-major."""
    return VALUE_SEPARATOR.join(f"{value:.9g}" for value in np.ravel(values))


def write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def random_pixels(rng: np.random.Generator, size: int = 32) -> np.ndarray:
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def write_competition_data(
    data_root: Path,
    rng: np.random.Generator,
    datasets: dict[str, int],
    images_per_scene: int = 4,
    outliers_per_dataset: int = 2,
) -> pd.DataFrame:
    """
    Write train/<dataset>/*.png, train_labels.csv and train_thresholds.csv like the competition ships them.

    Scene names repeat across datasets (scene0, scene1, ...), as they may in the real data.

    Returns
    -------
    written train_labels.csv

    """
    labels, thresholds = [], []
    for dataset, scenes in datasets.items():
        names = [(f"scene{s}", f"scene{s}_{i:02d}.png") for s in range(scenes) for i in range(images_per_scene)]
        names += [(OUTLIER_SCENE, f"outlier_{i:02d}.png") for i in range(outliers_per_dataset)]
        rotations = random_rotations(rng, len(names))
        translations = rng.normal(scale=3.0, size=(len(names), 3))

        rotations[[scene == OUTLIER_SCENE for scene, _ in names]] = np.nan
        translations[[scene == OUTLIER_SCENE for scene, _ in names]] = np.nan

        for (scene, image), rotation, translation in zip(names, rotations, translations, strict=True):
            write_image(data_root / "train" / dataset / image, random_pixels(rng))
            labels.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "image": image,
                    "rotation_matrix": format_vector(rotation),
                    "translation_vector": format_vector(translation),
                },
            )
        for s in range(scenes):
            thresholds.append({"dataset": dataset, "scene": f"scene{s}", "thresholds": format_vector(THRESHOLDS)})

    df = pd.DataFrame(labels)
    df.to_csv(data_root / "train_labels.csv", index=False)
    pd.DataFrame(thresholds).to_csv(data_root / "train_thresholds.csv", index=False)
    return df
