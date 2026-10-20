"""Inference script for SfM-Rotations model."""

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from sfm_rotations.data import build_transforms
from sfm_rotations.dvc_utils import pull_dvc_data
from sfm_rotations.losses import rotation_6d_to_matrix
from sfm_rotations.metrics import _normalize
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
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path)


@hydra.main(version_base=None, config_path="../../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Run model inference on test images and save predictions."""
    data_root = Path(cfg.data_root)
    test_dir = Path(cfg.test_dir)

    # 1. Download test data via DVC Python API if requested
    if cfg.get("dvc_pull", True):
        pull_dvc_data(
            targets=[str(data_root / "test.dvc")],
            required_paths=[test_dir],
        )

    # 2. Collect image paths
    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_paths = sorted([p for p in test_dir.rglob("*") if p.suffix in image_extensions])

    if not image_paths:
        print(f"No test images found in {test_dir}!")
        return

    print(f"Found {len(image_paths)} test images in {test_dir}")

    # 3. Setup transforms and DataLoader
    _, val_transform = build_transforms()
    dataset = InferenceDataset(image_paths, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # 4. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    model = SfMModel(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        freeze_backbone=False,
    ).to(device)

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        # Fallback to alternate checkpoint path if not found
        alt_path = Path("checkpoints/baseline/best.pt")
        if alt_path.exists():
            checkpoint_path = alt_path

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found! Running with uninitialized backbone weights.")

    model.eval()

    # 5. Run inference
    embeddings = []
    rotations = []
    translations = []
    paths = []

    with torch.no_grad():
        for batch_images, batch_paths in tqdm(loader, desc="Inference"):
            images = batch_images.to(device)
            emb, trans, rot_6d = model(images)
            rot_matrix = rotation_6d_to_matrix(rot_6d)

            embeddings.append(emb.float().cpu())
            rotations.append(rot_matrix.float().cpu())
            translations.append(trans.float().cpu())
            paths.extend(batch_paths)

    all_embeddings = torch.cat(embeddings).numpy()
    all_rotations = torch.cat(rotations).numpy()
    all_translations = torch.cat(translations).numpy()

    # 6. Cluster embeddings to assign scene labels
    norm_embeddings = _normalize(all_embeddings)
    predicted_clusters = cluster_embeddings(
        norm_embeddings,
        distance_threshold=cfg.cluster_distance,
        min_cluster_size=cfg.min_cluster_size,
    )

    # 7. Format and save results
    results = []
    for i, p in enumerate(paths):
        results.append(
            {
                "image_path": p,
                "scene_cluster": int(predicted_clusters[i]),
                "rotation_matrix": all_rotations[i].tolist(),
                "translation_vector": all_translations[i].tolist(),
            },
        )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Inference completed! Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
