import dataclasses
import json
import math
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

from sfm_rotations.contracts import PROTOCOL_PATH
from sfm_rotations.contracts import PROTOCOL_VERSION
from sfm_rotations.contracts import file_md5
from sfm_rotations.contracts import load_protocol
from sfm_rotations.contracts import load_val_split
from sfm_rotations.data import ContrastiveBatchSampler
from sfm_rotations.data import SceneDataset
from sfm_rotations.data import build_transforms
from sfm_rotations.data import load_labels
from sfm_rotations.data import train_val_split_by_cluster
from sfm_rotations.dvc_utils import pull_dvc_data
from sfm_rotations.losses import compute_losses
from sfm_rotations.metrics import EvalOutputs
from sfm_rotations.metrics import compute_all_metrics
from sfm_rotations.metrics import format_metrics
from sfm_rotations.metrics import load_thresholds
from sfm_rotations.metrics import pose_metrics
from sfm_rotations.model import SfMModel

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def model_config(cfg: DictConfig) -> dict:
    """Everything the model is rebuilt from at inference (stored in the checkpoint)."""
    return {
        "backbone": str(cfg.backbone),
        "embedding_dim": int(cfg.embedding_dim),
        "pooling": str(cfg.get("pooling", "cls")),
        "image_size": int(cfg.get("image_size", 224)),
    }


def postprocess_config(cfg: DictConfig) -> dict:
    """Clustering parameters inference takes from the checkpoint."""
    return {
        "cluster_distance": float(cfg.cluster_distance),
        "min_cluster_size": int(cfg.min_cluster_size),
        "linkage": str(cfg.get("linkage", "average")),
    }


def build_model(cfg: DictConfig) -> SfMModel:
    train_blocks = cfg.get("train_blocks")
    return SfMModel(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        freeze_backbone=cfg.get("freeze_backbone", False),
        train_blocks=None if train_blocks is None or train_blocks < 0 else int(train_blocks),
        pooling=cfg.get("pooling", "cls"),
    )


def split_indices(df: pd.DataFrame, data_root: Path, cfg: DictConfig) -> tuple[list[int], list[int]]:
    """
    Train/val split: the frozen protocol split when it belongs to these data, otherwise a fresh one.

    The frozen split (contracts/val_split.json) is tied to the md5 of train_labels.csv,
    so other data (e.g. synthetic data in tests) fall back to train_val_split_by_cluster.
    """
    split_path = REPO_ROOT / load_protocol(REPO_ROOT / PROTOCOL_PATH).split_path
    if split_path.exists():
        val_split = load_val_split(split_path)
        if val_split.labels_md5 == file_md5(data_root / "train_labels.csv"):
            print(f"Using frozen validation split {split_path}")
            return val_split.split(df)
    print("train_labels.csv differs from the protocol data: computing the split from seed and val_fraction")
    return train_val_split_by_cluster(df["scene_label"].tolist(), val_fraction=cfg.val_fraction, seed=cfg.seed)


def build_optimizer(model: SfMModel, cfg: DictConfig, total_steps: int):
    """AdamW with a small lr for the backbone and a larger one for the heads, linear warmup + cosine decay."""
    groups = [{"params": model.head_parameters(), "lr": cfg.get("lr_head", cfg.lr)}]
    backbone = model.backbone_parameters()
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.0))

    warmup = max(1, int(total_steps * cfg.get("warmup_fraction", 0.0)))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


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


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()


def run_epoch(
    model: SfMModel,
    loader: DataLoader,
    device: str,
    loss_weights: tuple[float, float, float],
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    temperature: float = 0.07,
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

    scheduler:
        Per-step lr scheduler.

    temperature:
        SupCon temperature.

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

            with _autocast(device):
                embeddings, translations, rotations_6d = model(images)
            total_loss, _ = compute_losses(
                embeddings.float(),
                translations.float(),
                rotations_6d.float(),
                labels,
                loss_weights,
                temperature=temperature,
            )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

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
        with _autocast(device):
            predictions = model.predict(images.to(device))
        embeddings.append(torch.from_numpy(predictions.embeddings))
        translations.append(torch.from_numpy(predictions.translations))
        rotations.append(torch.from_numpy(predictions.rotations))
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


def evaluate(
    model: SfMModel,
    loader: DataLoader,
    device: str,
    meta: pd.DataFrame,
    thresholds: dict,
    cfg: DictConfig,
) -> dict[str, float]:
    """
    Validation metrics of one epoch.

    Poses come from geometry (ALIKED + LightGlue + COLMAP, see evaluate.py), not from the model,
    so imc/* here is computed with GT poses for every image of a cluster: it measures how well
    the embeddings cluster, i.e. the upper bound reachable with perfect geometry. pose/* still
    describes the (untrained) pose heads.
    """
    outputs = collect_outputs(model, loader, device, meta)
    oracle = dataclasses.replace(
        outputs,
        rotations_pred=outputs.rotations_gt,
        translations_pred=outputs.translations_gt,
    )
    metrics = compute_all_metrics(
        oracle,
        thresholds,
        distance_threshold=cfg.cluster_distance,
        min_cluster_size=cfg.min_cluster_size,
        linkage=cfg.get("linkage", "average"),
    )
    metrics.update(pose_metrics(outputs))
    return metrics


def log_run_params(cfg: DictConfig) -> None:
    """Log the git commit and the config of the run to MLflow."""
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()  # noqa: S603, S607
    except Exception:
        commit_id = "unknown"
    mlflow.set_tag("git_commit_id", commit_id)
    mlflow.set_tag("mlflow.source.git.commit", commit_id)
    mlflow.set_tag("protocol_version", PROTOCOL_VERSION)
    mlflow.log_param("git_commit_id", commit_id)
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))


def pull_train_data(data_root: Path) -> None:
    """Pull training images, labels and thresholds from DVC."""
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
            log_run_params(cfg)

        seed_everything(cfg.seed)

        data_root = Path(cfg.data_root)
        checkpoints = Path(cfg.checkpoints)

        if cfg.get("dvc_pull", True):
            pull_train_data(data_root)

        df = load_labels(data_root)
        cluster_ids = df["scene_label"].tolist()
        train_indices, val_indices = split_indices(df, data_root, cfg)

        train_transform, val_transform = build_transforms(int(cfg.get("image_size", 224)))
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
        loader_kwargs = {
            "num_workers": cfg.num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": cfg.num_workers > 0,
        }
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)

        # Последовательный проход по всей валидации для метрик (sampler пропускает часть изображений).
        eval_loader = DataLoader(
            Subset(val_dataset, val_indices),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        eval_meta = df.iloc[val_indices]
        thresholds = load_thresholds(data_root)

        print(f"device: {device}")
        model = build_model(cfg).to(device)
        n_trainable = sum(p.numel() for p in model.backbone_parameters())
        print(f"trainable backbone parameters: {n_trainable / 1e6:.1f}M")
        total_steps = max(1, cfg.epochs * len(train_sampler))
        optimizer, scheduler = build_optimizer(model, cfg, total_steps)

        checkpoints.mkdir(parents=True, exist_ok=True)
        loss_weights = tuple(cfg.loss_weights)
        temperature = float(cfg.get("temperature", 0.07))
        best_score = float("-inf")

        history_epochs = []
        history_train_loss = []
        history_val_loss = []
        history_imc_score = []
        history_rot_maa = []

        # Эпоха 0 — предобученный DINOv2 до обучения: точка отсчёта, и best.pt не может оказаться хуже неё.
        first_epoch = 0 if cfg.get("eval_pretrained", True) else 1
        for epoch in range(first_epoch, cfg.epochs + 1):
            desc = f"Epoch {epoch}/{cfg.epochs}"
            if epoch == 0:
                train_loss = float("nan")
            else:
                train_sampler.set_epoch(epoch - 1)
                train_loss = run_epoch(
                    model,
                    train_loader,
                    device,
                    loss_weights,
                    f"{desc} train",
                    optimizer,
                    scheduler,
                    temperature,
                )
                print(f"Epoch {epoch}: train loss={train_loss:.5f}")

            val_loss = run_epoch(model, val_loader, device, loss_weights, f"{desc} val", temperature=temperature)
            print(f"Epoch {epoch}: validation loss={val_loss:.5f}")

            metrics = evaluate(model, eval_loader, device, eval_meta, thresholds, cfg)
            metrics.update(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch,
                    "protocol_version": PROTOCOL_VERSION,
                },
            )
            print(f"Epoch {epoch} metrics:\n{format_metrics(metrics)}")

            with open(checkpoints / f"metrics_epoch{epoch:03d}.json", "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            checkpoint = {
                "protocol_version": PROTOCOL_VERSION,
                "model_config": model_config(cfg),
                "postprocess": postprocess_config(cfg),
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            torch.save(checkpoint, checkpoints / "last.pt")

            score = metrics.get("imc/score", float("-inf"))
            if score > best_score:
                best_score = score
                torch.save(checkpoint, checkpoints / "best.pt")
                print(f"New best imc/score={score:.4f}")

            history_epochs.append(epoch)
            history_train_loss.append(train_loss)
            history_val_loss.append(val_loss)
            history_imc_score.append(score)
            history_rot_maa.append(metrics.get("pose/rel_rot_mAA", float("nan")))

            if enable_mlflow:
                for k, v in metrics.items():
                    if isinstance(v, int | float) and math.isfinite(v):
                        clean_key = re.sub(r"[^a-zA-Z0-9_\-./ :]", "_", k.replace("@", "_at_"))
                        mlflow.log_metric(clean_key, v, step=epoch)

        plots_dir = Path(cfg.get("plots_dir", "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        epochs_range = history_epochs

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
