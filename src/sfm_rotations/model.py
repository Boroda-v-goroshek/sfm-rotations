import torch
from torch import nn
from transformers import AutoModel


class SfMModel(nn.Module):
    """
    DINOv2 backbone with three heads on the CLS token.

    Heads:
        - projection_head: scene embedding for contrastive learning.
        - translation_head: translation vector (3).
        - rotation_head: rotation in the 6D representation (6).

    Parameters
    ----------
    backbone_name:
        HuggingFace model name.

    embedding_dim:
        Scene embedding size (also the hidden size of pose heads).

    freeze_backbone:
        Freeze the backbone and train only the heads.

    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        embedding_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 6),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the model.

        Parameters
        ----------
        images:
            Normalized images [B, 3, H, W].

        Returns
        -------
        embeddings [B, embedding_dim], translations [B, 3], rotations_6d [B, 6]

        """
        output = self.backbone(pixel_values=images)
        # CLS-токен DINOv2
        features = output.last_hidden_state[:, 0]
        return self.projection_head(features), self.translation_head(features), self.rotation_head(features)
