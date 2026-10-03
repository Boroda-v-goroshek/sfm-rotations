import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from sfm_rotations.data import ContrastiveBatchSampler
from sfm_rotations.data import SceneDataset
from sfm_rotations.data import build_transforms
from sfm_rotations.data import load_labels
from sfm_rotations.data import train_val_split_by_cluster
from sfm_rotations.losses import compute_losses
from sfm_rotations.losses import rotation_6d_to_matrix
from sfm_rotations.metrics import EvalOutputs
from sfm_rotations.metrics import compute_all_metrics
from sfm_rotations.metrics import format_metrics
from sfm_rotations.metrics import load_thresholds
from sfm_rotations.model import SfMModel


def seed_everything(seed: int) -> None:
    """
    Fix random seeds of python, numpy and torch.

    Parameters
    ----------
    seed:
        Random seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed arguments

    """
    parser = argparse.ArgumentParser(description="Train SfM scene/pose model")
    parser.add_argument("--data-root", type=Path, default=Path("image-matching-challenge-2025"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--backbone", default="facebook/dinov2-base")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[0.4, 0.3, 0.3])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Батч: round(batch_size * outlier_fraction) outliers + clusters_per_batch сцен поровну.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clusters-per-batch", type=int, default=7)
    parser.add_argument("--outlier-fraction", type=float, default=0.125)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--val-clusters-per-batch", type=int, default=3)
    parser.add_argument("--val-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--cluster-distance", type=float, default=0.3, help="Порог косинусного расстояния")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Кластеры меньше -> outliers")
    return parser.parse_args()


def run_epoch(
    model: SfMModel,
    loader: DataLoader,
    device: str,
    loss_weights: tuple[float, float, float],
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """
    Run one epoch.

    Trains if an optimizer is passed, otherwise only evaluates the loss.

    Parameters
    ----------
    model:
        Model to run.

    loader:
        Batches of images and labels.

    device:
        Torch device.

    loss_weights:
        Weights of contrastive, rotation and translation losses.

    desc:
        Progress bar description.

    optimizer:
        Optimizer for training; None for validation.

    Returns
    -------
    mean loss over the epoch

    """
    is_train = optimizer is not None
    model.train(is_train)
    losses = []

    progress = tqdm(loader, desc=desc)
    with torch.set_grad_enabled(is_train):
        for batch_images, batch_labels in progress:
            images = batch_images.to(device)
            labels = {key: value.to(device) for key, value in batch_labels.items()}

            embeddings, translations, rotations_6d = model(images)
            total_loss, _ = compute_losses(embeddings, translations, rotations_6d, labels, loss_weights)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(total_loss.item())
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_outputs(model: SfMModel, loader: DataLoader, device: str, meta: pd.DataFrame) -> EvalOutputs:
    """
    Run the model over all images of the loader and gather predictions with GT.

    Parameters
    ----------
    model:
        Model to evaluate.

    loader:
        Sequential (unshuffled) loader over validation images.

    device:
        Torch device.

    meta:
        Dataframe rows in the same order as the loader.

    Returns
    -------
    EvalOutputs for metric computation

    """
    model.eval()
    embeddings, rotations, translations, rotations_gt, translations_gt = [], [], [], [], []

    for images, labels in tqdm(loader, desc="eval"):
        emb, trans, rot_6d = model(images.to(device))
        embeddings.append(emb.float().cpu())
        translations.append(trans.float().cpu())
        rotations.append(rotation_6d_to_matrix(rot_6d).float().cpu())
        rotations_gt.append(labels["rotation_matrix"])
        translations_gt.append(labels["translation_vector"])

    return EvalOutputs(
        datasets=meta["dataset"].to_numpy(),
        scenes=meta["scene"].to_numpy(),
        scene_labels=meta["scene_label"].to_numpy(),
        embeddings=torch.cat(embeddings).numpy(),
        rotations_pred=torch.cat(rotations).numpy(),
        translations_pred=torch.cat(translations).numpy(),
        rotations_gt=torch.cat(rotations_gt).numpy(),
        translations_gt=torch.cat(translations_gt).numpy(),
    )


def main() -> None:
    """Train the model, compute validation metrics and save checkpoints every epoch."""
    args = parse_args()
    seed_everything(args.seed)

    df = load_labels(args.data_root)
    cluster_ids = df["scene_label"].tolist()
    train_indices, val_indices = train_val_split_by_cluster(cluster_ids, val_fraction=args.val_fraction, seed=args.seed)

    train_transform, val_transform = build_transforms()
    # Sampler отдаёт глобальные индексы строк df, поэтому оба датасета строятся по полному df.
    train_dataset = SceneDataset(df, transform=train_transform)
    val_dataset = SceneDataset(df, transform=val_transform)

    train_sampler = ContrastiveBatchSampler(
        train_indices,
        cluster_ids,
        batch_size=args.batch_size,
        clusters_per_batch=args.clusters_per_batch,
        outlier_fraction=args.outlier_fraction,
        seed=args.seed,
    )
    val_sampler = ContrastiveBatchSampler(
        val_indices,
        cluster_ids,
        batch_size=args.val_batch_size,
        clusters_per_batch=args.val_clusters_per_batch,
        outlier_fraction=args.val_outlier_fraction,
        seed=args.seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Последовательный проход по всей валидации для метрик (sampler пропускает часть изображений).
    eval_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_meta = df.iloc[val_indices]
    thresholds = load_thresholds(args.data_root)

    print(f"device: {device}")
    model = SfMModel(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    loss_weights = tuple(args.loss_weights)
    best_score = float("-inf")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        desc = f"Epoch {epoch + 1}/{args.epochs}"

        train_loss = run_epoch(model, train_loader, device, loss_weights, f"{desc} train", optimizer)
        print(f"Epoch {epoch + 1}: train loss={train_loss:.5f}")

        val_loss = run_epoch(model, val_loader, device, loss_weights, f"{desc} val")
        print(f"Epoch {epoch + 1}: validation loss={val_loss:.5f}")

        outputs = collect_outputs(model, eval_loader, device, eval_meta)
        metrics = compute_all_metrics(
            outputs,
            thresholds,
            distance_threshold=args.cluster_distance,
            min_cluster_size=args.min_cluster_size,
        )
        metrics.update({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch + 1} metrics:\n{format_metrics(metrics)}")

        with open(args.checkpoints / f"metrics_epoch{epoch + 1:03d}.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        checkpoint = {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}
        torch.save(checkpoint, args.checkpoints / "last.pt")

        score = metrics.get("imc/score", float("-inf"))
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.checkpoints / "best.pt")
            print(f"New best imc/score={score:.4f}")


if __name__ == "__main__":
    main()
