"""Inference: cluster images with DINOv2 embeddings, reconstruct clusters with COLMAP, write submission.csv."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from sfm_rotations.contracts import OUTLIER_LABEL
from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import SUBMISSION_COLUMNS
from sfm_rotations.contracts import VALUE_SEPARATOR
from sfm_rotations.data import build_transforms
from sfm_rotations.dvc_utils import pull_dvc_data
from sfm_rotations.geometry import GeometryConfig
from sfm_rotations.geometry import Reconstructor
from sfm_rotations.metrics import cluster_embeddings
from sfm_rotations.model import SfMModel


class InferenceDataset(Dataset):
    """Dataset for inference on raw image files."""

    def __init__(self, image_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image) if self.transform else image


def load_model(cfg: DictConfig, device: str) -> SfMModel:
    """
    Build the model and load the checkpoint.

    Parameters
    ----------
    cfg:
        Inference config.

    device:
        Torch device.

    Returns
    -------
    model in eval mode

    """
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Нет чекпоинта {checkpoint_path}")

    model = SfMModel(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        freeze_backbone=False,
    ).to(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.eval()


@torch.no_grad()
def compute_embeddings(model: SfMModel, image_paths: list[Path], cfg: DictConfig, device: str) -> np.ndarray:
    """
    Compute L2-normalized scene embeddings.

    Parameters
    ----------
    model:
        Model in eval mode.

    image_paths:
        Image files [N].

    cfg:
        Inference config.

    device:
        Torch device.

    Returns
    -------
    embeddings [N, D]

    """
    _, val_transform = build_transforms()
    loader = DataLoader(
        InferenceDataset(image_paths, transform=val_transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    embeddings = [model(images.to(device))[0].float().cpu() for images in tqdm(loader, desc="embeddings")]
    return F.normalize(torch.cat(embeddings), dim=1).numpy()


def reconstruct_dataset(
    image_paths: list[Path],
    embeddings: np.ndarray,
    reconstructor: Reconstructor,
    distance_threshold: float,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster the images of one dataset and reconstruct every cluster.

    Parameters
    ----------
    image_paths:
        Image files of the dataset [N].

    embeddings:
        L2-normalized embeddings [N, D].

    reconstructor:
        Geometric reconstruction of a cluster.

    distance_threshold:
        Cosine distance threshold for clustering.

    min_cluster_size:
        Clusters smaller than this are treated as outliers.

    Returns
    -------
    cluster id [N], rotations [N, 3, 3], translations [N, 3]; poses are NaN where not registered

    """
    clusters = cluster_embeddings(embeddings, distance_threshold, min_cluster_size)
    rotations = np.full((len(image_paths), 3, 3), np.nan)
    translations = np.full((len(image_paths), 3), np.nan)

    for cluster in np.unique(clusters[clusters != OUTLIER_LABEL]):
        indices = np.flatnonzero(clusters == cluster)
        cluster_paths = [image_paths[i] for i in indices]
        rotations[indices], translations[indices] = reconstructor.reconstruct(cluster_paths, embeddings[indices])
    return clusters, rotations, translations


def build_reconstructor(cfg: DictConfig, device: str) -> Reconstructor:
    """
    Create the reconstructor from the geometry section of the config.

    Parameters
    ----------
    cfg:
        Inference config.

    device:
        Torch device.

    Returns
    -------
    Reconstructor

    """
    return Reconstructor(GeometryConfig(**cfg.geometry), device)


def _format_vector(values: np.ndarray) -> str:
    return VALUE_SEPARATOR.join("nan" if np.isnan(value) else f"{value:.9f}" for value in values.ravel())


@hydra.main(version_base=None, config_path="../../configs", config_name="baseline_infer")
def main(cfg: DictConfig) -> None:
    """Predict scene clusters and camera poses for test images and write submission.csv."""
    data_root = Path(cfg.data_root)
    test_dir = Path(cfg.test_dir)

    if cfg.get("dvc_pull", True):
        pull_dvc_data(
            targets=[str(data_root / "test.dvc")],
            required_paths=[test_dir],
        )

    image_extensions = {".png", ".jpg", ".jpeg"}
    image_paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in image_extensions)
    if not image_paths:
        print(f"No test images found in {test_dir}!")
        return
    print(f"Found {len(image_paths)} test images in {test_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    model = load_model(cfg, device)
    embeddings = compute_embeddings(model, image_paths, cfg, device)
    reconstructor = build_reconstructor(cfg, device)

    # Сцены кластеризуются внутри датасета: так считает метрика соревнования.
    datasets = np.array([path.relative_to(test_dir).parts[0] for path in image_paths])
    rows = []
    for dataset in tqdm(np.unique(datasets), desc="datasets"):
        indices = np.flatnonzero(datasets == dataset)
        paths = [image_paths[i] for i in indices]
        clusters, rotations, translations = reconstruct_dataset(
            paths,
            embeddings[indices],
            reconstructor,
            cfg.cluster_distance,
            cfg.min_cluster_size,
        )
        for path, cluster, rotation, translation in zip(paths, clusters, rotations, translations, strict=True):
            rows.append(
                {
                    "dataset": dataset,
                    "scene": OUTLIER_SCENE if cluster == OUTLIER_LABEL else f"cluster{cluster}",
                    "image": path.relative_to(test_dir / dataset).as_posix(),
                    "rotation_matrix": _format_vector(rotation),
                    "translation_vector": _format_vector(translation),
                },
            )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS)).to_csv(output_path, index=False)
    print(f"Inference completed! Saved {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
