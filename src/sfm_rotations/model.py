import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import AutoModel

from sfm_rotations.contracts import Predictions
from sfm_rotations.losses import rotation_6d_to_matrix

POOLINGS = ("cls", "mean", "cls+mean")


class SfMModel(nn.Module):
    """
    DINOv2 backbone, a pooled image descriptor and three heads.

    The scene embedding is what clustering uses. On this data (~30 training scenes)
    the pretrained DINOv2 space is already very good and a randomly initialised
    projection or a fully unfrozen backbone overfits and generalizes worse to unseen
    datasets. Hence the defaults: no projection (embedding = pooled descriptor) and
    only the last ``train_blocks`` transformer blocks are fine-tuned.

    Heads:
        - projection_head: scene embedding (Identity when embedding_dim == 0).
        - translation_head: translation vector (3).
        - rotation_head: rotation in the 6D representation (6).

    Parameters
    ----------
    backbone_name:
        HuggingFace model name.

    embedding_dim:
        Size of a linear projection of the descriptor; 0 keeps the pooled descriptor as is.

    freeze_backbone:
        Freeze the whole backbone (overrides train_blocks).

    train_blocks:
        Number of last transformer blocks to fine-tune; None fine-tunes the whole backbone.

    pooling:
        "cls", "mean" (of patch tokens) or "cls+mean" (concatenation of both, each L2-normalized).

    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        embedding_dim: int = 256,
        freeze_backbone: bool = False,
        train_blocks: int | None = None,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        self.pooling = pooling

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.hidden_size
        feature_dim = 2 * hidden_dim if pooling == "cls+mean" else hidden_dim

        self.projection_head = nn.Linear(feature_dim, embedding_dim) if embedding_dim > 0 else nn.Identity()
        self.translation_head = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU(), nn.Linear(256, 3))
        self.rotation_head = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU(), nn.Linear(256, 6))

        if freeze_backbone:
            train_blocks = 0
        if train_blocks is not None:
            self._freeze_all_but_last(train_blocks)

    def _freeze_all_but_last(self, train_blocks: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if train_blocks <= 0:
            return
        blocks = self.backbone.encoder.layer
        for block in blocks[max(0, len(blocks) - train_blocks) :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad = True

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        heads = (self.projection_head, self.translation_head, self.rotation_head)
        return [p for head in heads for p in head.parameters() if p.requires_grad]

    def pooled(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=images).last_hidden_state
        cls, patches = hidden[:, 0], hidden[:, 1:]
        if self.pooling == "cls":
            return cls
        if self.pooling == "mean":
            return patches.mean(dim=1)
        return torch.cat([F.normalize(cls, dim=-1), F.normalize(patches.mean(dim=1), dim=-1)], dim=-1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the model.

        Parameters
        ----------
        images:
            Normalized images [B, 3, H, W]; H and W must be multiples of the patch size.

        Returns
        -------
        embeddings [B, D], translations [B, 3], rotations_6d [B, 6]

        """
        features = self.pooled(images)
        return self.projection_head(features), self.translation_head(features), self.rotation_head(features)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Predictions:
        """
        Model outputs in the contract format.

        Parameters
        ----------
        images:
            Normalized images [B, 3, H, W].

        Returns
        -------
        Predictions with float32 numpy arrays

        """
        embeddings, translations, rotations_6d = self(images)
        return Predictions(
            embeddings=embeddings.float().cpu().numpy(),
            rotations=rotation_6d_to_matrix(rotations_6d.float()).cpu().numpy(),
            translations=translations.float().cpu().numpy(),
        )
