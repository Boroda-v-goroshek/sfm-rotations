"""Evaluate the full pipeline (DINOv2 clustering + COLMAP poses) on the frozen validation split."""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from sfm_rotations.contracts import load_protocol
from sfm_rotations.contracts import load_val_split
from sfm_rotations.data import load_labels
from sfm_rotations.infer import build_reconstructor
from sfm_rotations.infer import compute_embeddings
from sfm_rotations.infer import load_model
from sfm_rotations.infer import reconstruct_dataset
from sfm_rotations.metrics import EvalOutputs
from sfm_rotations.metrics import compute_all_metrics
from sfm_rotations.metrics import format_metrics
from sfm_rotations.metrics import load_thresholds


@hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="fast")
def main(cfg: DictConfig) -> None:
    """Reconstruct validation scenes and compute the protocol metrics."""
    data_root = Path(cfg.data_root)
    df = load_labels(data_root)
    protocol = load_protocol()
    _, val_indices = load_val_split(protocol.split_path).split(df)
    val = df.iloc[val_indices].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating {len(val)} validation images on {device}")

    model, checkpoint = load_model(cfg, device)
    postprocess = checkpoint["postprocess"]
    paths = [Path(path) for path in val["image_path"]]
    embeddings = compute_embeddings(model, paths, checkpoint["model_config"]["image_size"], cfg, device)
    reconstructor = build_reconstructor(cfg, device)

    rotations = np.full((len(val), 3, 3), np.nan)
    translations = np.full((len(val), 3), np.nan)
    for dataset in val["dataset"].unique():
        indices = np.flatnonzero(val["dataset"] == dataset)
        _, rotations[indices], translations[indices] = reconstruct_dataset(
            [paths[i] for i in indices],
            embeddings[indices],
            reconstructor,
            postprocess["cluster_distance"],
            postprocess["min_cluster_size"],
            postprocess.get("linkage", "average"),
        )

    outputs = EvalOutputs(
        datasets=val["dataset"].to_numpy(),
        scenes=val["scene"].to_numpy(),
        scene_labels=val["scene_label"].to_numpy(),
        embeddings=embeddings,
        rotations_pred=rotations,
        translations_pred=translations,
        rotations_gt=np.stack(val["rotation_matrix"].to_list()).reshape(-1, 3, 3),
        translations_gt=np.stack(val["translation_vector"].to_list()),
    )
    metrics = compute_all_metrics(
        outputs,
        load_thresholds(data_root),
        distance_threshold=postprocess["cluster_distance"],
        min_cluster_size=postprocess["min_cluster_size"],
        linkage=postprocess.get("linkage", "average"),
    )
    metrics["pose/registered_fraction"] = float(np.isfinite(rotations).all(axis=(1, 2)).mean())
    print(format_metrics(metrics))

    output_path = Path(cfg.metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
