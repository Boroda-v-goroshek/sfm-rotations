import torch
import torch.nn.functional as F  # noqa: N812

from sfm_rotations.data import OUTLIER_LABEL


def rotation_6d_to_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert the 6D rotation representation to rotation matrices.

    Uses Gram-Schmidt orthogonalization (Zhou et al., 2019).

    Parameters
    ----------
    x:
        6D rotations [B, 6].

    eps:
        Epsilon for vector normalization.

    Returns
    -------
    rotation matrices [B, 3, 3]

    """
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_geodesic_loss(r_pred: torch.Tensor, r_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean squared geodesic angle between predicted and target rotations.

    Parameters
    ----------
    r_pred:
        Predicted rotations [B, 3, 3].

    r_target:
        Target rotations [B, 3, 3].

    eps:
        Margin for clamping cos(theta), keeps the acos gradient finite.

    Returns
    -------
    scalar loss (radians squared)

    """
    relative = r_target.transpose(-1, -2) @ r_pred
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # eps отступ: градиент acos в ±1 бесконечен.
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta).square().mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    outlier_label: int = OUTLIER_LABEL,
) -> torch.Tensor:
    """
    Supervised contrastive (SupCon) loss.

    Positive pairs:
        - images of the same scene in the batch.

    Negative pairs:
        - images of different scenes in the batch.

    Outliers:
        - do not form positive pairs with each other, serve only as negatives.

    Parameters
    ----------
    embeddings:
        Scene embeddings [B, D].

    labels:
        scene_label for every image [B].

    temperature:
        Softmax temperature.

    outlier_label:
        scene_label value that marks outliers.

    Returns
    -------
    scalar loss

    """
    device = embeddings.device
    embeddings = F.normalize(embeddings, p=2, dim=1)

    logits = embeddings @ embeddings.T / temperature
    # Численная стабилизация
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    identity_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
    positive_mask = (labels[:, None] == labels[None, :]) & ~identity_mask & (labels != outlier_label)[:, None]

    exp_logits = torch.exp(logits).masked_fill(identity_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    mean_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


def compute_losses(
    embeddings: torch.Tensor,
    translations: torch.Tensor,
    rotations_6d: torch.Tensor,
    labels: dict[str, torch.Tensor],
    loss_weights: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Weighted sum of contrastive, rotation and translation losses.

    Pose losses are computed only for non-outliers (outliers have NaN poses).

    Parameters
    ----------
    embeddings:
        Scene embeddings [B, D].

    translations:
        Predicted translations [B, 3].

    rotations_6d:
        Predicted 6D rotations [B, 6].

    labels:
        Batch labels from SceneDataset.

    loss_weights:
        Weights of contrastive, rotation and translation losses.

    Returns
    -------
    total_loss, dict with float values of each loss component

    """
    scene_labels = labels["scene_label"]
    has_pose = scene_labels != OUTLIER_LABEL

    contrastive = supervised_contrastive_loss(embeddings, scene_labels)
    if has_pose.any():
        rotations = rotation_6d_to_matrix(rotations_6d[has_pose])
        rotation = rotation_geodesic_loss(rotations, labels["rotation_matrix"][has_pose])
        translation = F.smooth_l1_loss(translations[has_pose], labels["translation_vector"][has_pose])
    else:
        rotation = translation = embeddings.sum() * 0.0

    total = contrastive * loss_weights[0] + rotation * loss_weights[1] + translation * loss_weights[2]
    parts = {
        "contrastive": contrastive.item(),
        "rotation": rotation.item(),
        "translation": translation.item(),
    }
    return total, parts
