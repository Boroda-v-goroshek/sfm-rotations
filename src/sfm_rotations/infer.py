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
from sfm_rotations.data import IMAGE_EXTENSIONS
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


def load_model(cfg: DictConfig, device: str) -> tuple[SfMModel, dict]:
    """
    Rebuild the model from the checkpoint and load its weights.

    Parameters
    ----------
    cfg:
        Inference config (only cfg.checkpoint is used).

    device:
        Torch device.

    Returns
    -------
    model in eval mode, checkpoint (model_config, postprocess, ...)

    """
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Нет чекпоинта {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_config" not in checkpoint:
        alt_path = Path("checkpoints/dinov2-large/best.pt")
        if alt_path.exists():
            print(
                f"Внимание: в {checkpoint_path} нет 'model_config' (старый формат). "
                f"Загружаем актуальный чекпоинт из {alt_path}.",
            )
            checkpoint = torch.load(alt_path, map_location="cpu")
        else:
            raise KeyError(
                f"В чекпоинте {checkpoint_path} отсутствует 'model_config'. "
                "Укажите актуальный чекпоинт (например, checkpoint=checkpoints/dinov2-large/best.pt).",
            )

    model_config = checkpoint["model_config"]
    model = SfMModel(
        backbone_name=model_config["backbone"],
        embedding_dim=model_config["embedding_dim"],
        pooling=model_config.get("pooling", "cls"),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def compute_embeddings(
    model: SfMModel,
    image_paths: list[Path],
    image_size: int,
    cfg: DictConfig,
    device: str,
) -> np.ndarray:
    """
    Compute L2-normalized scene embeddings.

    Parameters
    ----------
    model:
        Model in eval mode.

    image_paths:
        Image files [N].

    image_size:
        Input side the model was trained with (checkpoint model_config).

    cfg:
        Inference config.

    device:
        Torch device.

    Returns
    -------
    embeddings [N, D]

    """
    _, val_transform = build_transforms(image_size)
    loader = DataLoader(
        InferenceDataset(image_paths, transform=val_transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    embeddings = []
    for images in tqdm(loader, desc="embeddings"):
        embeddings.append(torch.from_numpy(model.predict(images.to(device)).embeddings))
    return F.normalize(torch.cat(embeddings), dim=1).numpy()


def reconstruct_dataset(
    image_paths: list[Path],
    embeddings: np.ndarray,
    reconstructor: Reconstructor,
    distance_threshold: float,
    min_cluster_size: int,
    linkage: str = "average",
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

    linkage:
        Linkage of the agglomerative clustering.

    Returns
    -------
    cluster id [N], rotations [N, 3, 3], translations [N, 3]; poses are NaN where not registered

    """
    clusters = cluster_embeddings(embeddings, distance_threshold, min_cluster_size, linkage)
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


@hydra.main(version_base=None, config_path="../../configs/inference", config_name="fast")
def main(cfg: DictConfig) -> None:
    """Predict scene clusters and camera poses for test images and write submission.csv."""
    data_root = Path(cfg.data_root)
    test_dir = Path(cfg.test_dir)

    if cfg.get("dvc_pull", True):
        pull_dvc_data(
            targets=[str(data_root / "test.dvc")],
            required_paths=[test_dir],
        )

    image_paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No test images found in {test_dir}!")
        return
    print(f"Found {len(image_paths)} test images in {test_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    model, checkpoint = load_model(cfg, device)
    postprocess = checkpoint["postprocess"]
    reconstructor = build_reconstructor(cfg, device)
    image_size = checkpoint["model_config"]["image_size"]

    # Группируем картинки по сценам/папкам:
    # 1. Если передана директория со сценами (data/test/<scene>/<image>)
    # 2. Если передана отдельная сцена (data/test/stairs/<image>)
    scene_images: dict[str, list[tuple[Path, str]]] = {}
    for path in image_paths:
        relative_parts = path.relative_to(test_dir).parts
        if len(relative_parts) > 1:
            scene_name = relative_parts[0]
            image_name = path.relative_to(test_dir / relative_parts[0]).as_posix()
        else:
            scene_name = test_dir.name
            image_name = path.name
        scene_images.setdefault(scene_name, []).append((path, image_name))

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for scene_name, image_info in scene_images.items():
        paths = [image_path for image_path, _ in image_info]
        names = [rel_name for _, rel_name in image_info]

        print(f"\n--- Running inference for scene '{scene_name}' ({len(paths)} images) ---")
        embeddings = compute_embeddings(model, paths, image_size, cfg, device)
        clusters, rotations, translations = reconstruct_dataset(
            paths,
            embeddings,
            reconstructor,
            postprocess["cluster_distance"],
            postprocess["min_cluster_size"],
            postprocess.get("linkage", "average"),
        )

        scene_rows = []
        for image_name, cluster, rotation, translation in zip(
            names,
            clusters,
            rotations,
            translations,
            strict=True,
        ):
            scene_rows.append(
                {
                    "dataset": scene_name,
                    "scene": OUTLIER_SCENE if cluster == OUTLIER_LABEL else f"cluster{cluster}",
                    "image": image_name,
                    "rotation_matrix": _format_vector(rotation),
                    "translation_vector": _format_vector(translation),
                },
            )

        # Сохраняем отдельный CSV-файл для сцены: <output_stem>_<scene_name>.csv
        scene_csv_path = output_path.parent / f"{output_path.stem}_{scene_name}{output_path.suffix}"
        pd.DataFrame(scene_rows, columns=list(SUBMISSION_COLUMNS)).to_csv(scene_csv_path, index=False)
        print(f"Saved {len(scene_rows)} predictions for '{scene_name}' to {scene_csv_path}")

    print("\nInference completed!")


if __name__ == "__main__":
    import sys

    # Support convenience flags from CLI: --dvc-pull, --no-dvc, --no-dvc-pull
    sys.argv = [
        arg.replace("--dvc-pull", "dvc_pull=true")
        .replace("--no-dvc-pull", "dvc_pull=false")
        .replace("--no-dvc", "dvc_pull=false")
        for arg in sys.argv
    ]
    main()
