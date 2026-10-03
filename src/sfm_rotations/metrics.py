"""
Validation metrics.

Pose:
    - relative rotation error over all image pairs within a scene
      (median and mAA over ROTATION_THRESHOLDS_DEG). Independent of the scene
      coordinate frame, so it is meaningful for scenes unseen during training.

Embeddings:
    - ARI of embedding clustering and retrieval recall@1.

Approximation of the IMC2025 competition metric:
    - clustering of embeddings within each dataset, greedy scene -> cluster
      assignment, mAA of camera centers after similarity alignment (Umeyama,
      without RANSAC unlike the official metric), clustering score and their
      harmonic mean.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from sfm_rotations.data import OUTLIER_LABEL

ROTATION_THRESHOLDS_DEG = (5.0, 10.0, 15.0, 30.0)


@dataclass
class EvalOutputs:
    """
    Model predictions and GT for all validation images, in the same order.

    Parameters
    ----------
    datasets:
        Dataset name per image [N].

    scenes:
        Scene name per image [N], "outliers" for outliers.

    scene_labels:
        scene_label per image [N], OUTLIER_LABEL for outliers.

    embeddings:
        Scene embeddings [N, D].

    rotations_pred:
        Predicted rotations [N, 3, 3].

    translations_pred:
        Predicted translations [N, 3].

    rotations_gt:
        GT rotations [N, 3, 3], NaN for outliers.

    translations_gt:
        GT translations [N, 3], NaN for outliers.

    """

    datasets: np.ndarray  # [N] str
    scenes: np.ndarray  # [N] str, "outliers" для выбросов
    scene_labels: np.ndarray  # [N] int, OUTLIER_LABEL для выбросов
    embeddings: np.ndarray  # [N, D]
    rotations_pred: np.ndarray  # [N, 3, 3]
    translations_pred: np.ndarray  # [N, 3]
    rotations_gt: np.ndarray  # [N, 3, 3], NaN для выбросов
    translations_gt: np.ndarray  # [N, 3], NaN для выбросов


def load_thresholds(data_root: Path) -> dict[tuple[str, str], np.ndarray]:
    """
    Load per-scene mAA thresholds from train_thresholds.csv.

    Parameters
    ----------
    data_root:
        Competition data folder.

    Returns
    -------
    dict (dataset, scene) -> array of thresholds

    """
    df = pd.read_csv(data_root / "train_thresholds.csv")
    return {
        (dataset, scene): np.array(thresholds.split(";"), dtype=float)
        for dataset, scene, thresholds in zip(df["dataset"], df["scene"], df["thresholds"], strict=True)
    }


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """
    Geodesic angle between rotations.

    Parameters
    ----------
    r_a:
        Rotations [..., 3, 3].

    r_b:
        Rotations [..., 3, 3].

    Returns
    -------
    angles in degrees [...]

    """
    relative = np.swapaxes(r_a, -1, -2) @ r_b
    cos_theta = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def camera_centers(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """
    Camera centers c = -R^T t.

    Parameters
    ----------
    rotations:
        Rotations [N, 3, 3].

    translations:
        Translations [N, 3].

    Returns
    -------
    camera centers [N, 3]

    """
    return -np.einsum("nji,nj->ni", rotations, translations)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Similarity transform (s, R, t) minimizing ||dst - (s R src + t)||^2.

    Parameters
    ----------
    src:
        Source points [N, 3].

    dst:
        Target points [N, 3].

    Returns
    -------
    scale, rotation [3, 3], translation [3]

    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0

    rotation = u @ sign @ vt
    var_src = (src_c**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ sign) / var_src) if var_src > 0 else 1.0
    translation = mu_dst - scale * rotation @ mu_src
    return scale, rotation, translation


# ---------------------------------------------------------------------------
# 1. Позы
# ---------------------------------------------------------------------------


def pose_metrics(outputs: EvalOutputs) -> dict[str, float]:
    """
    Relative rotation error over all image pairs within each scene.

    Relative rotation of a pair: R_ij = R_j R_i^T.

    Parameters
    ----------
    outputs:
        Validation predictions and GT.

    Returns
    -------
    dict with pose/rel_rot_err_median_deg and pose/rel_rot_mAA

    """
    has_pose = outputs.scene_labels != OUTLIER_LABEL
    r_pred, r_gt = outputs.rotations_pred[has_pose], outputs.rotations_gt[has_pose]
    labels = outputs.scene_labels[has_pose]

    rel_errors = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), k=1)
        rel_pred = r_pred[idx[j]] @ np.swapaxes(r_pred[idx[i]], -1, -2)
        rel_gt = r_gt[idx[j]] @ np.swapaxes(r_gt[idx[i]], -1, -2)
        rel_errors.append(rotation_angle_deg(rel_pred, rel_gt))

    if not rel_errors:
        return {}
    rel_err = np.concatenate(rel_errors)
    return {
        "pose/rel_rot_err_median_deg": float(np.median(rel_err)),
        "pose/rel_rot_mAA": float(np.mean([(rel_err < t).mean() for t in ROTATION_THRESHOLDS_DEG])),
    }


# ---------------------------------------------------------------------------
# 2. Эмбеддинги
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows.

    Parameters
    ----------
    x:
        Matrix [N, D].

    Returns
    -------
    row-normalized matrix [N, D]

    """
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float,
    min_cluster_size: int,
) -> np.ndarray:
    """
    Agglomerative clustering of embeddings by cosine distance.

    Clusters smaller than min_cluster_size are marked as outliers.

    Parameters
    ----------
    embeddings:
        L2-normalized embeddings of one dataset [N, D].

    distance_threshold:
        Cosine distance above which clusters are not merged.

    min_cluster_size:
        Minimum size of a non-outlier cluster.

    Returns
    -------
    cluster id per image [N], OUTLIER_LABEL for outliers

    """
    if len(embeddings) < 2:
        return np.full(len(embeddings), OUTLIER_LABEL)

    clusterer = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    predicted = clusterer.fit_predict(embeddings)

    ids, counts = np.unique(predicted, return_counts=True)
    small = ids[counts < min_cluster_size]
    predicted[np.isin(predicted, small)] = OUTLIER_LABEL
    return predicted


def embedding_metrics(outputs: EvalOutputs, predicted_clusters: np.ndarray) -> dict[str, float]:
    """
    Quality of scene embeddings.

    Metrics:
        - emb/ARI: Adjusted Rand Index between predicted clusters and scenes.
        - emb/recall@1: share of images whose nearest neighbour (within the
          dataset) belongs to the same scene.

    Parameters
    ----------
    outputs:
        Validation predictions and GT.

    predicted_clusters:
        Cluster id per image, OUTLIER_LABEL for outliers.

    Returns
    -------
    dict with embedding metrics

    """
    labels = outputs.scene_labels
    embeddings = _normalize(outputs.embeddings)

    # ARI: каждый outlier — отдельный кластер, чтобы они не сливались в один «класс»;
    # кластеры уникальны только внутри датасета — добавляем датасет к ключу.
    def _keys(values: np.ndarray) -> np.ndarray:
        """
        Turn per-dataset cluster ids into global keys; every outlier gets its own key.

        Parameters
        ----------
        values:
            Cluster id per image.

        Returns
        -------
        integer key per image

        """
        values = values.copy()
        mask = values == OUTLIER_LABEL
        values[mask] = values.max(initial=0) + 1 + np.arange(mask.sum())
        return pd.factorize(pd.Series(outputs.datasets).str.cat(values.astype(str), sep="/"))[0]

    metrics = {"emb/ARI": float(adjusted_rand_score(_keys(labels), _keys(predicted_clusters)))}

    # recall@1: ближайший сосед внутри датасета из той же сцены.
    hits = []
    for dataset in np.unique(outputs.datasets):
        idx = np.flatnonzero(outputs.datasets == dataset)
        similarity = embeddings[idx] @ embeddings[idx].T
        np.fill_diagonal(similarity, -np.inf)
        nearest = labels[idx][similarity.argmax(axis=1)]
        local_labels = labels[idx]
        for label, neighbour in zip(local_labels, nearest, strict=True):
            if label != OUTLIER_LABEL and (local_labels == label).sum() >= 2:
                hits.append(label == neighbour)
    if hits:
        metrics["emb/recall@1"] = float(np.mean(hits))
    return metrics


# ---------------------------------------------------------------------------
# 3. Приближение метрики соревнования
# ---------------------------------------------------------------------------


def _scene_cluster_maa(
    scene_mask: np.ndarray,
    cluster_mask: np.ndarray,
    outputs: EvalOutputs,
    thresholds: np.ndarray,
) -> float:
    """
    MAA of the scene camera centers registered in the cluster.

    Centers are aligned to GT with a similarity transform first.

    Parameters
    ----------
    scene_mask:
        Images of the scene.

    cluster_mask:
        Images of the cluster.

    outputs:
        Validation predictions and GT.

    thresholds:
        Distance thresholds of the scene.

    Returns
    -------
    mAA in [0, 1]

    """
    common = np.flatnonzero(scene_mask & cluster_mask)
    if len(common) < 3:
        return 0.0

    centers_pred = camera_centers(outputs.rotations_pred[common], outputs.translations_pred[common])
    centers_gt = camera_centers(outputs.rotations_gt[common], outputs.translations_gt[common])
    scale, rotation, translation = umeyama_alignment(centers_pred, centers_gt)
    aligned = scale * centers_pred @ rotation.T + translation
    errors = np.linalg.norm(aligned - centers_gt, axis=1)

    scene_size = scene_mask.sum()
    return float(np.mean([(errors < threshold).sum() / scene_size for threshold in thresholds]))


def competition_metrics(
    outputs: EvalOutputs,
    predicted_clusters: np.ndarray,
    thresholds: dict[tuple[str, str], np.ndarray],
) -> dict[str, float]:
    """
    Approximation of the IMC2025 competition metric.

    Each scene is greedily assigned to the cluster with the best mAA
    (ties broken by clustering score). Per dataset:
        - mAA: weighted by scene size (recall analogue).
        - clustering: |C ∩ S| / |C| (precision analogue).
        - score: harmonic mean of the two.
    Results are averaged over datasets.

    Parameters
    ----------
    outputs:
        Validation predictions and GT.

    predicted_clusters:
        Cluster id per image, OUTLIER_LABEL for outliers.

    thresholds:
        dict (dataset, scene) -> thresholds.

    Returns
    -------
    dict with overall and per-dataset imc/* metrics

    """
    per_dataset_scores, per_dataset_maa, per_dataset_clustering = [], [], []
    metrics: dict[str, float] = {}

    for dataset in np.unique(outputs.datasets):
        in_dataset = outputs.datasets == dataset
        dataset_scenes = [s for s in np.unique(outputs.scenes[in_dataset]) if s != "outliers"]
        if not dataset_scenes:
            continue

        clusters = [c for c in np.unique(predicted_clusters[in_dataset]) if c != OUTLIER_LABEL]
        maa_weighted, scene_total, inter_total, cluster_total = 0.0, 0, 0, 0

        for scene in dataset_scenes:
            scene_mask = in_dataset & (outputs.scenes == scene)
            scene_thresholds = thresholds[(dataset, scene)]

            # Жадно: кластер с макс. mAA, при равенстве — с большим clustering score.
            best = (0.0, 0.0, 0, 0)  # maa, clustering, |C ∩ S|, |C|
            for cluster in clusters:
                cluster_mask = in_dataset & (predicted_clusters == cluster)
                maa = _scene_cluster_maa(scene_mask, cluster_mask, outputs, scene_thresholds)
                inter = int((scene_mask & cluster_mask).sum())
                size = int(cluster_mask.sum())
                candidate = (maa, inter / size, inter, size)
                if candidate[:2] > best[:2]:
                    best = candidate

            maa_weighted += best[0] * scene_mask.sum()
            scene_total += int(scene_mask.sum())
            inter_total += best[2]
            cluster_total += best[3]

        maa = maa_weighted / scene_total
        clustering = inter_total / cluster_total if cluster_total else 0.0
        score = 2 * maa * clustering / (maa + clustering) if maa + clustering > 0 else 0.0

        metrics[f"imc/{dataset}/mAA"] = maa
        metrics[f"imc/{dataset}/clustering"] = clustering
        metrics[f"imc/{dataset}/score"] = score
        per_dataset_maa.append(maa)
        per_dataset_clustering.append(clustering)
        per_dataset_scores.append(score)

    if per_dataset_scores:
        metrics["imc/mAA"] = float(np.mean(per_dataset_maa))
        metrics["imc/clustering"] = float(np.mean(per_dataset_clustering))
        metrics["imc/score"] = float(np.mean(per_dataset_scores))
    return metrics


def compute_all_metrics(
    outputs: EvalOutputs,
    thresholds: dict[tuple[str, str], np.ndarray],
    distance_threshold: float = 0.3,
    min_cluster_size: int = 3,
) -> dict[str, float]:
    """
    Cluster embeddings within each dataset and compute all metrics.

    Parameters
    ----------
    outputs:
        Validation predictions and GT.

    thresholds:
        dict (dataset, scene) -> thresholds.

    distance_threshold:
        Cosine distance threshold for clustering.

    min_cluster_size:
        Clusters smaller than this are treated as outliers.

    Returns
    -------
    dict with pose/*, emb/* and imc/* metrics

    """
    predicted_clusters = np.full(len(outputs.scene_labels), OUTLIER_LABEL)
    embeddings = _normalize(outputs.embeddings)
    for dataset in np.unique(outputs.datasets):
        idx = np.flatnonzero(outputs.datasets == dataset)
        predicted_clusters[idx] = cluster_embeddings(embeddings[idx], distance_threshold, min_cluster_size)

    return {
        **pose_metrics(outputs),
        **embedding_metrics(outputs, predicted_clusters),
        **competition_metrics(outputs, predicted_clusters, thresholds),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """
    Format metrics as a text table, skipping per-dataset rows (they stay in json).

    Parameters
    ----------
    metrics:
        Metrics dict.

    Returns
    -------
    formatted string

    """
    rows = [(k, v) for k, v in metrics.items() if not (k.startswith("imc/") and k.count("/") > 1)]
    width = max((len(k) for k, _ in rows), default=0)
    return "\n".join(f"  {k:<{width}}  {v:.4f}" for k, v in rows)
