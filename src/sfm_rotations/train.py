import json
import random
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from sfm_rotations.data import ContrastiveBatchSampler
from sfm_rotations.data import SceneDataset
from sfm_rotations.data import build_transforms
from sfm_rotations.data import load_labels
from sfm_rotations.data import train_val_split_by_cluster
from sfm_rotations.dvc_utils import pull_dvc_data
from sfm_rotations.losses import compute_losses
from sfm_rotations.losses import rotation_6d_to_matrix
from sfm_rotations.metrics import EvalOutputs
from sfm_rotations.metrics import compute_all_metrics
from sfm_rotations.metrics import format_metrics
from sfm_rotations.metrics import load_thresholds
from sfm_rotations.model import SfMModel

load_dotenv()


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


@hydra.main(version_base=None, config_path="../../configs", config_name="baseline_train")
def main(cfg: DictConfig) -> None:
    """Train the model, compute validation metrics and save checkpoints every epoch."""
    enable_mlflow = cfg.get("logging", True)
    if enable_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)
    else:
        print("MLflow logging disabled (--no-logging / logging=false).")

    run_context = mlflow.start_run() if enable_mlflow else nullcontext()
    with run_context:
        if enable_mlflow:
            try:
                commit_id = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()  # noqa: S603, S607
                )
            except Exception:
                commit_id = "unknown"
            mlflow.set_tag("git_commit_id", commit_id)
            mlflow.set_tag("mlflow.source.git.commit", commit_id)
            mlflow.log_param("git_commit_id", commit_id)
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        seed_everything(cfg.seed)

        data_root = Path(cfg.data_root)
        checkpoints = Path(cfg.checkpoints)

        if cfg.get("dvc_pull", True):
            pull_dvc_data(
                targets=[
                    str(data_root / "train.dvc"),
                    str(data_root / "train_labels.csv.dvc"),
                    str(data_root / "train_thresholds.csv.dvc"),
                ],
                required_paths=[
                    data_root / "train_labels.csv",
                    data_root / "train_thresholds.csv",
                    data_root / "train",
                ],
            )

        df = load_labels(data_root)
        cluster_ids = df["scene_label"].tolist()
        train_indices, val_indices = train_val_split_by_cluster(
            cluster_ids,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )

        train_transform, val_transform = build_transforms()
        # Sampler отдаёт глобальные индексы строк df, поэтому оба датасета строятся по полному df.
        train_dataset = SceneDataset(df, transform=train_transform)
        val_dataset = SceneDataset(df, transform=val_transform)

        train_sampler = ContrastiveBatchSampler(
            train_indices,
            cluster_ids,
            batch_size=cfg.batch_size,
            clusters_per_batch=cfg.clusters_per_batch,
            outlier_fraction=cfg.outlier_fraction,
            seed=cfg.seed,
        )
        val_sampler = ContrastiveBatchSampler(
            val_indices,
            cluster_ids,
            batch_size=cfg.val_batch_size,
            clusters_per_batch=cfg.val_clusters_per_batch,
            outlier_fraction=cfg.val_outlier_fraction,
            seed=cfg.seed,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pin_memory = device == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

        # Последовательный проход по всей валидации для метрик (sampler пропускает часть изображений).
        eval_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        eval_meta = df.iloc[val_indices]
        thresholds = load_thresholds(data_root)

        print(f"device: {device}")
        model = SfMModel(
            backbone_name=cfg.backbone,
            embedding_dim=cfg.embedding_dim,
            freeze_backbone=cfg.freeze_backbone,
        ).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

        checkpoints.mkdir(parents=True, exist_ok=True)
        loss_weights = tuple(cfg.loss_weights)
        best_score = float("-inf")

        history_train_loss = []
        history_val_loss = []
        history_imc_score = []
        history_rot_maa = []

        for epoch in range(cfg.epochs):
            train_sampler.set_epoch(epoch)
            desc = f"Epoch {epoch + 1}/{cfg.epochs}"

            train_loss = run_epoch(model, train_loader, device, loss_weights, f"{desc} train", optimizer)
            print(f"Epoch {epoch + 1}: train loss={train_loss:.5f}")

            val_loss = run_epoch(model, val_loader, device, loss_weights, f"{desc} val")
            print(f"Epoch {epoch + 1}: validation loss={val_loss:.5f}")

            outputs = collect_outputs(model, eval_loader, device, eval_meta)
            metrics = compute_all_metrics(
                outputs,
                thresholds,
                distance_threshold=cfg.cluster_distance,
                min_cluster_size=cfg.min_cluster_size,
            )
            metrics.update({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1} metrics:\n{format_metrics(metrics)}")

            with open(checkpoints / f"metrics_epoch{epoch + 1:03d}.json", "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            checkpoint = {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}
            torch.save(checkpoint, checkpoints / "last.pt")

            score = metrics.get("imc/score", float("-inf"))
            if score > best_score:
                best_score = score
                torch.save(checkpoint, checkpoints / "best.pt")
                print(f"New best imc/score={score:.4f}")

            history_train_loss.append(train_loss)
            history_val_loss.append(val_loss)
            history_imc_score.append(score)
            history_rot_maa.append(metrics.get("pose/rel_rot_mAA", float("nan")))

            if enable_mlflow:
                for k, v in metrics.items():
                    if isinstance(v, int | float):
                        clean_key = re.sub(r"[^a-zA-Z0-9_\-./ :]", "_", k.replace("@", "_at_"))
                        mlflow.log_metric(clean_key, v, step=epoch + 1)

        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        epochs_range = list(range(1, cfg.epochs + 1))

        plt.figure()
        plt.plot(epochs_range, history_train_loss, label="Train Loss")
        plt.plot(epochs_range, history_val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train and Validation Losses")
        plt.savefig(plots_dir / "loss.png")
        plt.close()

        plt.figure()
        plt.plot(epochs_range, history_imc_score, label="IMC Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.title("IMC Competition Score")
        plt.savefig(plots_dir / "imc_score.png")
        plt.close()

        plt.figure()
        plt.plot(epochs_range, history_rot_maa, label="Rotation mAA", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("mAA")
        plt.legend()
        plt.title("Pose Relative Rotation mAA")
        plt.savefig(plots_dir / "rot_maa.png")
        plt.close()

        history_summary = {
            "train_loss": history_train_loss,
            "val_loss": history_val_loss,
            "imc_score": history_imc_score,
            "rot_maa": history_rot_maa,
        }
        with open(plots_dir / "training_summary.json", "w") as f:
            json.dump(history_summary, f, indent=2)

        if enable_mlflow:
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")


if __name__ == "__main__":
    import sys

    # Support convenience flags from CLI: --no-logging or --no-mlflow
    sys.argv = [
        arg.replace("--no-logging", "logging=false").replace("--no-mlflow", "logging=false") for arg in sys.argv
    ]
    main()
