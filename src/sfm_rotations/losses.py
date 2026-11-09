import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment

from sfm_rotations.data import OUTLIER_LABEL
from sfm_rotations.model_utils import skew_symmetric


def supervised_contrastive_loss(
    embeddings,
    scene_ids,
    temperature=0.07
):
    """
    embeddings: [B, N, D]
    scene_ids: [B, N], -1 = outlier
    """
    B, N, D = embeddings.shape

    z = F.normalize(embeddings, dim=-1)
    z = z.reshape(B * N, D)

    labels = scene_ids.reshape(B * N)

    logits = (z @ z.T) / temperature

    self_mask = torch.eye(
        B * N, device=z.device, dtype=torch.bool
    )

    valid_anchor = labels >= 0

    positive_mask = (
        (labels[:, None] == labels[None, :])
        & (labels[:, None] >= 0)
        & (~self_mask)
    )

    denominator_mask = ~self_mask

    logits = logits - logits.max(
        dim=1, keepdim=True
    ).values.detach()

    exp_logits = torch.exp(logits) * denominator_mask

    log_prob = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True) + 1e-8
    )

    positive_count = positive_mask.sum(dim=1)

    per_anchor = -(
        log_prob * positive_mask
    ).sum(dim=1) / positive_count.clamp_min(1)

    valid_anchor = valid_anchor & (positive_count > 0)

    if not valid_anchor.any():
        return embeddings.sum() * 0.0

    return per_anchor[valid_anchor].mean()


def hungarian_assignment_loss(
    assignment_logits,
    scene_ids
):
    """
    assignment_logits: [B, N, K+1]
    scene_ids: [B, N], -1 = outlier

    Returns:
      assignment_loss
      activity_targets: [B, K]
    """
    B, N, K_plus_1 = assignment_logits.shape
    K = K_plus_1 - 1

    log_probs = F.log_softmax(assignment_logits, dim=-1)

    activity_targets = torch.zeros(
        B, K,
        device=assignment_logits.device,
        dtype=assignment_logits.dtype
    )

    total_loss = assignment_logits.sum() * 0.0
    total_count = 0

    for b in range(B):
        labels = scene_ids[b]
        unique_scenes = torch.unique(labels[labels >= 0])

        num_gt_scenes = len(unique_scenes)

        if num_gt_scenes > K:
            raise ValueError(
                f"В batch item {b} найдено {num_gt_scenes} сцен, "
                f"но K_max={K}. Увеличьте число scene slots."
            )

        # GT scenes x predicted scene slots.
        if num_gt_scenes > 0:
            cost = torch.zeros(
                num_gt_scenes, K,
                device=assignment_logits.device
            )

            for g, scene_id in enumerate(unique_scenes):
                mask = labels == scene_id

                # Отрицательное среднее log likelihood.
                cost[g] = -log_probs[
                    b, mask, :K
                ].mean(dim=0)

            # Matching — дискретная операция без градиента.
            rows, cols = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            gt_to_slot = {}

            for row, col in zip(rows, cols):
                scene_id = int(unique_scenes[row].item())
                gt_to_slot[scene_id] = int(col)

                activity_targets[b, col] = 1.0

            # Assignment CE для inlier images.
            for scene_id, slot_id in gt_to_slot.items():
                mask = labels == scene_id

                total_loss = total_loss + F.cross_entropy(
                    assignment_logits[b, mask],
                    torch.full(
                        (int(mask.sum().item()),),
                        slot_id,
                        device=assignment_logits.device,
                        dtype=torch.long
                    ),
                    reduction="sum"
                )

                total_count += int(mask.sum().item())

        # Все outliers должны назначаться в Dustbin.
        outlier_mask = labels < 0

        if outlier_mask.any():
            dustbin_target = torch.full(
                (int(outlier_mask.sum().item()),),
                K,
                device=assignment_logits.device,
                dtype=torch.long
            )

            total_loss = total_loss + F.cross_entropy(
                assignment_logits[b, outlier_mask],
                dustbin_target,
                reduction="sum"
            )

            total_count += int(outlier_mask.sum().item())

    loss = total_loss / max(total_count, 1)

    return loss, activity_targets


def slot_activity_loss(
    activity_logits,
    activity_targets,
    pos_weight=None
):
    """
    activity_logits: [B, K]
    activity_targets: [B, K]
    """
    if pos_weight is not None:
        pos_weight = torch.as_tensor(
            pos_weight,
            device=activity_logits.device,
            dtype=activity_logits.dtype
        )

    return F.binary_cross_entropy_with_logits(
        activity_logits,
        activity_targets,
        pos_weight=pos_weight
    )


def outlier_classification_loss(
    assignment_probs,
    scene_ids
):
    """
    assignment_probs: [B, N, K+1]
    scene_ids: [B, N], -1 = outlier
    """
    dustbin_probs = assignment_probs[..., -1]

    targets = (scene_ids < 0).float()

    return F.binary_cross_entropy(
        dustbin_probs.clamp(1e-6, 1.0 - 1e-6),
        targets
    )


def rotation_geodesic_error(R_pred, R_gt):
    """
    Returns angular error in radians.
    R_pred, R_gt: [..., 3, 3]
    """
    R_rel = R_gt.transpose(-1, -2) @ R_pred

    trace = (
        R_rel[..., 0, 0]
        + R_rel[..., 1, 1]
        + R_rel[..., 2, 2]
    )

    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)

    skew = R_rel - R_rel.transpose(-1, -2)

    vee = torch.stack(
        [
            skew[..., 2, 1],
            skew[..., 0, 2],
            skew[..., 1, 0]
        ],
        dim=-1
    )

    sin_theta = 0.5 * torch.linalg.vector_norm(
        vee, dim=-1
    )

    return torch.atan2(sin_theta, cos_theta)


def absolute_pose_loss(
    T_pred,
    T_gt,
    scene_ids,
    rotation_weight=1.0,
    translation_weight=1.0
):
    """
    T_pred, T_gt: [B, N, 4, 4]
    scene_ids: [B, N]

    Исключает выбросы и объекты с некорректными GT-позами.
    """

    # Объект должен быть inlier и иметь конечную GT-позу.
    valid = (
        (scene_ids >= 0)
        & torch.isfinite(T_gt).all(dim=(-2, -1))
    )

    if not valid.any():
        return T_pred.sum() * 0.0

    # Фильтруем объекты ДО любых геометрических операций.
    T_pred_valid = T_pred[valid]
    T_gt_valid = T_gt[valid]

    R_pred = T_pred_valid[..., :3, :3]
    R_gt = T_gt_valid[..., :3, :3]

    t_pred = T_pred_valid[..., :3, 3]
    t_gt = T_gt_valid[..., :3, 3]

    rotation_error = rotation_geodesic_error(
        R_pred, R_gt
    )

    translation_error = F.smooth_l1_loss(
        t_pred,
        t_gt,
        reduction="none"
    ).sum(dim=-1)

    loss = (
        rotation_weight * rotation_error
        + translation_weight * translation_error
    )

    return loss.mean()


def so3_log(R):
    """
    R: [..., 3, 3]
    Returns rotation vector [..., 3].
    """
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)

    skew = R - R.transpose(-1, -2)

    vee = torch.stack(
        [
            skew[..., 2, 1],
            skew[..., 0, 2],
            skew[..., 1, 0]
        ],
        dim=-1
    )

    sin_theta = 0.5 * torch.linalg.vector_norm(
        vee, dim=-1
    )

    theta = torch.atan2(sin_theta, cos_theta)

    scale_regular = theta / (
        2.0 * sin_theta.clamp_min(1e-8)
    )

    scale_small = 0.5 + theta * theta / 12.0

    scale = torch.where(
        theta < 1e-4,
        scale_small,
        scale_regular
    )

    return scale.unsqueeze(-1) * vee


def se3_log(T):
    """
    T: [..., 4, 4]
    Returns xi = [rho, phi], [..., 6].
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    phi = so3_log(R)

    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))

    phi_hat = skew_symmetric(phi)
    phi_hat2 = phi_hat @ phi_hat

    # Inverse left Jacobian:
    # J^-1 = I - 0.5*phi_hat + C*phi_hat^2
    C_regular = (
        1.0 / theta2.clamp_min(1e-12)
        - (1.0 + torch.cos(theta)) /
        (2.0 * theta * torch.sin(theta)).clamp_min(1e-8)
    )

    C_small = 1.0 / 12.0 + theta2 / 720.0

    C = torch.where(
        theta2 < 1e-6,
        C_small,
        C_regular
    )[..., None]

    eye = torch.eye(
        3, device=T.device, dtype=T.dtype
    ).expand(*T.shape[:-2], 3, 3)

    J_inv = eye - 0.5 * phi_hat + C * phi_hat2

    rho = (J_inv @ t.unsqueeze(-1)).squeeze(-1)

    return torch.cat([rho, phi], dim=-1)


def invert_se3(T):
    """
    Аналитическая inverse для rigid transformation.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0

    return T_inv


def cluster_aware_geometric_loss(
    T_pred,
    T_gt,
    assignment_probs,
    scene_ids,
    translation_weight=1.0,
    rotation_weight=1.0
):
    """
    T_pred, T_gt: [B, N, 4, 4]
    assignment_probs: [B, N, K+1]
    scene_ids: [B, N], -1 = outlier

    Invalid GT transformations are excluded.
    """

    B, N = scene_ids.shape

    scene_probs = assignment_probs[..., :-1]

    total_loss = T_pred.sum() * 0.0
    total_weight = T_pred.sum() * 0.0

    for b in range(B):

        labels = scene_ids[b]

        # 1. Проверяем конечность GT transforms.
        finite_mask = torch.isfinite(T_gt[b]).all(dim=(-2, -1))

        # 2. Исключаем outliers и некорректные GT.
        valid_mask = (labels >= 0) & finite_mask

        valid_indices = torch.where(valid_mask)[0]

        # Нужно минимум два валидных объекта.
        if len(valid_indices) < 2:
            continue

        # 3. Работаем только с валидными объектами.
        T_pred_valid = T_pred[b, valid_indices]
        T_gt_valid = T_gt[b, valid_indices]

        probs_valid = scene_probs[b, valid_indices]
        labels_valid = labels[valid_indices]

        # 4. Формируем маску пар одной GT-сцены.
        pair_mask = (
            labels_valid[:, None] == labels_valid[None, :]
        )

        M = len(valid_indices)

        eye = torch.eye(
            M,
            device=T_pred.device,
            dtype=torch.bool
        )

        pair_mask = pair_mask & (~eye)

        if not pair_mask.any():
            continue

        # 5. Relative predicted transforms.
        T_i_inv = invert_se3(T_pred_valid)

        T_ij_pred = (
            T_i_inv[:, None] @ T_pred_valid[None, :]
        )

        # 6. Relative GT transforms.
        T_i_gt_inv = invert_se3(T_gt_valid)

        T_ij_gt = (
            T_i_gt_inv[:, None] @ T_gt_valid[None, :]
        )

        # 7. Error transform.
        T_error = (
            invert_se3(T_ij_gt) @ T_ij_pred
        )

        error_twist = se3_log(T_error)

        rho_error = error_twist[..., :3]
        phi_error = error_twist[..., 3:]

        geometric_error = (
            translation_weight * (rho_error ** 2).sum(dim=-1)
            + rotation_weight * (phi_error ** 2).sum(dim=-1)
        )

        # 8. Assignment co-membership.
        pair_weights = (
            probs_valid @ probs_valid.transpose(0, 1)
        )

        weights = pair_weights * pair_mask.float()

        total_loss = total_loss + (
            weights * geometric_error
        ).sum()

        total_weight = total_weight + weights.sum()

    return total_loss / total_weight.clamp_min(1e-8)


def compute_total_loss(
    outputs,
    scene_ids,
    T_gt,
    weights=None
):
    if weights is None:
        weights = {
            "contrastive": 1.0,
            "assignment": 1.0,
            "activity": 1.0,
            "outlier": 0.2,
            "pose": 1.0,
            "cag": 0.05
        }
    elif isinstance(weights, (list, tuple)):
        weights = {
            "contrastive": weights[0],
            "assignment": weights[1],
            "activity": weights[2],
            "outlier": weights[3],
            "pose": weights[4],
            "cag": weights[5]
        }

    # 1. Contrastive.
    loss_contrastive = supervised_contrastive_loss(
        outputs["contrastive_embeddings"],
        scene_ids
    )

    # 2. Hungarian assignment.
    loss_assignment, activity_targets = (
        hungarian_assignment_loss(
            outputs["assignment_logits"],
            scene_ids
        )
    )

    # 3. Scene slot activity.
    loss_activity = slot_activity_loss(
        outputs["activity_logits"],
        activity_targets
    )

    # 4. Outlier classification.
    loss_outlier = outlier_classification_loss(
        outputs["assignment_probs"],
        scene_ids
    )

    # 5. Absolute pose.
    loss_pose = absolute_pose_loss(
        outputs["poses"],
        T_gt,
        scene_ids
    )

    # 6. Cluster-aware geometric loss.
    loss_cag = cluster_aware_geometric_loss(
        outputs["poses"],
        T_gt,
        outputs["assignment_probs"],
        scene_ids
    )

    losses = {
        "contrastive": loss_contrastive,
        "assignment": loss_assignment,
        "activity": loss_activity,
        "outlier": loss_outlier,
        "pose": loss_pose,
        "cag": loss_cag
    }

    total = sum(
        weights[name] * loss
        for name, loss in losses.items()
    )

    total = torch.sum(
        torch.stack(
            tuple(weights[name] * loss
            for name, loss in losses.items())
        )
    )

    return total, losses
