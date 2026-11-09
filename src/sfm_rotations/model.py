import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

from sfm_rotations.model_utils import se3_exp


class ImageEncoder(nn.Module):
    """
    Обёртка над внешним backbone.
    Ожидает:
      [B, D] или [B, D, H, W]
    Также поддерживает dict с ключом 'x_norm_clstoken'.
    """
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        out_dim: int = 256,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.out_dim = out_dim

        backbone_dim = self.backbone.config.hidden_size

        if backbone_dim != out_dim:
            self.projection = nn.Linear(backbone_dim, out_dim)
        else:
            self.projection = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False


    def forward(self, images):
        features = self.backbone(images)

        outputs = self.backbone(pixel_values=images)

        # Извлекаем CLS-токен.
        features = outputs.last_hidden_state[:, 0]

        features = self.projection(features)

        return features


class SetTransformer(nn.Module):
    """
    Permutation-equivariant Transformer для обработки набора изображений.
    Позиционные embeddings намеренно отсутствуют.
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N, D]
        return self.norm(self.encoder(x))


class GatedFusion(nn.Module):
    """
    Fusion исходных и контекстных embeddings:
      g = sigmoid(Wg [x; h])
      f = LN(x + g * Wh(h))
    """
    def __init__(self, dim: int):
        super().__init__()

        self.gate = nn.Linear(dim * 2, dim)
        self.context_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h):
        gate = torch.sigmoid(
            self.gate(torch.cat([x, h], dim=-1))
        )

        fused = x + gate * self.context_proj(h)
        return self.norm(fused)


class SlotAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_scene_slots: int,
        num_iterations: int = 3
    ):
        super().__init__()

        self.num_scene_slots = num_scene_slots
        self.num_slots = num_scene_slots + 1
        self.num_iterations = num_iterations
        self.dim = dim

        # Все scene slots + отдельный Dustbin slot.
        self.slot_mu = nn.Parameter(
            torch.randn(1, self.num_slots, dim) * 0.02
        )
        self.slot_log_sigma = nn.Parameter(
            torch.zeros(1, self.num_slots, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.scale = dim ** -0.5

    def forward(self, inputs):
        """
        inputs: [B, N, D]

        Returns:
          slots: [B, K_max + 1, D]
          assignment_logits: [B, N, K_max + 1]
          assignment_probs: [B, N, K_max + 1]
        """
        B, N, D = inputs.shape

        normalized_inputs = self.norm_inputs(inputs)

        k = self.to_k(normalized_inputs)
        v = self.to_v(normalized_inputs)

        mu = self.slot_mu.expand(B, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(B, -1, -1)

        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.num_iterations):
            q = self.to_q(self.norm_slots(slots))

            # [B, N, K+1]
            logits = torch.einsum(
                "bnd,bkd->bnk", k, q
            ) * self.scale

            # Для каждого изображения распределяем массу
            # между scene slots и Dustbin.
            attn = logits.softmax(dim=-1)

            # Нормализация по изображениям для агрегации
            # признаков в каждый слот.
            attn_for_update = attn + 1e-8
            attn_for_update = attn_for_update / (
                attn_for_update.sum(dim=1, keepdim=True) + 1e-8
            )

            updates = torch.einsum(
                "bnk,bnd->bkd",
                attn_for_update,
                v
            )

            slots = self.gru(
                updates.reshape(-1, D),
                slots.reshape(-1, D)
            ).reshape(B, self.num_slots, D)

            slots = slots + self.mlp(self.norm_mlp(slots))

        # Пересчитываем assignment после последнего обновления slots.
        q = self.to_q(self.norm_slots(slots))

        assignment_logits = torch.einsum(
            "bnd,bkd->bnk", k, q
        ) * self.scale

        assignment_probs = assignment_logits.softmax(dim=-1)

        return slots, assignment_logits, assignment_probs


class SlotActivityHead(nn.Module):
    """
    Предсказывает существование каждой потенциальной сцены.
    Не классифицирует отдельные изображения и не заменяет Dustbin.
    """
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, scene_slots):
        """
        scene_slots: [B, K_max, D]

        Returns:
          logits: [B, K_max]
          probs:  [B, K_max]
        """
        logits = self.net(scene_slots).squeeze(-1)
        probs = torch.sigmoid(logits)

        return logits, probs


class ContrastiveHead(nn.Module):
    def __init__(self, dim: int, projection_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SceneConditioning(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, image_features, scene_slots, assignment_probs):
        """
        image_features: [B, N, D]
        scene_slots: [B, K_max, D]
        assignment_probs: [B, N, K_max + 1]
        """
        # Исключаем Dustbin.
        scene_probs = assignment_probs[..., :-1]

        scene_confidence = scene_probs.sum(
            dim=-1, keepdim=True
        )

        normalized_probs = scene_probs / (
            scene_confidence + 1e-8
        )

        scene_context = torch.einsum(
            "bnk,bkd->bnd",
            normalized_probs,
            scene_slots
        )

        combined = torch.cat(
            [
                image_features,
                scene_context,
                scene_confidence
            ],
            dim=-1
        )

        return self.net(combined)


class PoseHead(nn.Module):
    """
    Предсказывает twist xi = [rho, phi].
    rho — translation component.
    phi — rotation component.
    """
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 6)
        )

    def forward(self, x):
        return self.net(x)


class SfMModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int = 768,
        num_scene_slots: int = 8,
        projection_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        slot_iterations: int = 3,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.num_scene_slots = num_scene_slots

        self.image_encoder = ImageEncoder(
            backbone_name=backbone_name,
            out_dim=embedding_dim,
            freeze_backbone=freeze_backbone
        )

        self.set_transformer = SetTransformer(
            dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads
        )

        self.gated_fusion = GatedFusion(embedding_dim)

        self.slot_attention = SlotAttention(
            dim=embedding_dim,
            num_scene_slots=num_scene_slots,
            num_iterations=slot_iterations
        )

        self.slot_activity_head = SlotActivityHead(embedding_dim)

        self.contrastive_head = ContrastiveHead(
            dim=embedding_dim,
            projection_dim=projection_dim
        )

        self.scene_conditioning = SceneConditioning(embedding_dim)

        self.pose_head = PoseHead(embedding_dim)

    def forward(self, images):
        """
        images: [B, N, C, H, W]
        """
        B, N, C, H, W = images.shape

        # 1. Shared image encoder.
        flat_images = images.reshape(B * N, C, H, W)
        x = self.image_encoder(flat_images)
        x = x.reshape(B, N, -1)

        # 2. Set Transformer.
        h = self.set_transformer(x)

        # 3. Gated Fusion.
        f = self.gated_fusion(x, h)

        # 4. Slot Attention.
        slots, assignment_logits, assignment_probs = (
            self.slot_attention(f)
        )

        scene_slots = slots[:, :self.num_scene_slots]

        # 5. Scene Activity Prediction.
        activity_logits, activity_probs = (
            self.slot_activity_head(scene_slots)
        )

        # 6. Contrastive projection.
        z = self.contrastive_head(f)

        # 7. Scene Conditioning.
        conditioned_features = self.scene_conditioning(
            f,
            scene_slots,
            assignment_probs
        )

        # 8. Pose regression.
        xi = self.pose_head(conditioned_features)

        # 9. SE(3) exponential map.
        T_pred = se3_exp(xi)

        return {
            "image_features": x,
            "context_features": h,
            "fused_features": f,
            "scene_slots": scene_slots,
            "all_slots": slots,
            "assignment_logits": assignment_logits,
            "assignment_probs": assignment_probs,
            "activity_logits": activity_logits,
            "activity_probs": activity_probs,
            "contrastive_embeddings": z,
            "conditioned_features": conditioned_features,
            "pose_twists": xi,
            "poses": T_pred
        }
