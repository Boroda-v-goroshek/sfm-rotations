from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

OUTLIER_SCENE = "outliers"
OUTLIER_LABEL = -1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_labels(data_root: Path) -> pd.DataFrame:
    """
    Load train_labels.csv and parse it into a training-ready dataframe.

    Added columns:
        - scene_label: integer scene id, unique across datasets; outliers get OUTLIER_LABEL.
        - image_path: absolute path to the image file.
        - rotation_matrix: float32 array of shape (9,), row-major; NaN for outliers.
        - translation_vector: float32 array of shape (3,); NaN for outliers.

    Parameters
    ----------
    data_root:
        Competition data folder containing train/ and train_labels.csv.

    Returns
    -------
    dataframe with one row per image

    """
    df = pd.read_csv(data_root / "train_labels.csv")

    # Сцены в разных датасетах могут называться одинаково, поэтому ключ — (dataset, scene).
    scene_key = df["dataset"] + "/" + df["scene"]
    is_outlier = df["scene"] == OUTLIER_SCENE
    categories = sorted(scene_key[~is_outlier].unique())
    mapping = {value: i for i, value in enumerate(categories)}
    df["scene_label"] = scene_key.map(mapping).where(~is_outlier, OUTLIER_LABEL).astype(int)

    df["image_path"] = [str(data_root / "train" / d / img) for d, img in zip(df["dataset"], df["image"], strict=True)]
    df["rotation_matrix"] = df["rotation_matrix"].apply(lambda x: np.array(x.split(";"), dtype=np.float32))
    df["translation_vector"] = df["translation_vector"].apply(lambda x: np.array(x.split(";"), dtype=np.float32))
    return df


def train_val_split_by_cluster(
    cluster_ids: Sequence[int],
    val_fraction: float = 0.2,
    outlier_label: int = OUTLIER_LABEL,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split the dataset into train/val.

    Regular clusters:
        - go entirely either to train or to val.

    Outliers:
        - are split randomly between train and val.

    Parameters
    ----------
    cluster_ids:
        cluster_id for every image.

    val_fraction:
        Validation fraction.

    outlier_label:
        cluster_id value that marks outliers.

    seed:
        Random seed.

    Returns
    -------
    train_indices, val_indices

    """
    cluster_ids_arr = np.asarray(cluster_ids)
    rng = np.random.default_rng(seed)

    # Перемешиваем именно кластеры, а не изображения.
    normal_clusters = rng.permutation(np.unique(cluster_ids_arr[cluster_ids_arr != outlier_label]))
    n_val_clusters = int(round(len(normal_clusters) * val_fraction))
    val_clusters = normal_clusters[:n_val_clusters]

    is_normal = cluster_ids_arr != outlier_label
    is_val_cluster = np.isin(cluster_ids_arr, val_clusters)
    train_indices = np.flatnonzero(is_normal & ~is_val_cluster).tolist()
    val_indices = np.flatnonzero(is_normal & is_val_cluster).tolist()

    outlier_indices = rng.permutation(np.flatnonzero(~is_normal))
    n_val_outliers = int(round(len(outlier_indices) * val_fraction))
    val_indices.extend(outlier_indices[:n_val_outliers].tolist())
    train_indices.extend(outlier_indices[n_val_outliers:].tolist())

    return rng.permutation(train_indices).tolist(), rng.permutation(val_indices).tolist()


class SceneDataset(Dataset):
    """
    Image dataset with scene and pose labels.

    Indexed by global dataframe row indices, which is what
    ContrastiveBatchSampler yields.

    Parameters
    ----------
    dataframe:
        Output of load_labels (the full dataframe, not a train/val slice).

    transform:
        Image transform applied to the RGB PIL image.

    """

    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """
        Number of images in the dataset.

        Returns
        -------
        dataset length

        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Load one image and its labels.

        Parameters
        ----------
        idx:
            Global dataframe row index.

        Returns
        -------
        image:
            Transformed image tensor [3, H, W].

        labels:
            dict with scene_label (long), rotation_matrix [3, 3] and translation_vector [3].

        """
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as pil_image:
            image: torch.Tensor = self.transform(pil_image.convert("RGB"))  # type: ignore[assignment]

        labels = {
            "scene_label": torch.tensor(row["scene_label"], dtype=torch.long),
            "rotation_matrix": torch.from_numpy(row["rotation_matrix"]).reshape(3, 3),
            "translation_vector": torch.from_numpy(row["translation_vector"]),
        }
        return image, labels


class ContrastiveBatchSampler(Sampler[list[int]]):
    """
    Batch sampler for contrastive learning.

    Each batch contains:
        round(batch_size * outlier_fraction) outliers
        + clusters_per_batch clusters × samples_per_cluster images.

    For example, batch_size=32, clusters_per_batch=7, outlier_fraction=0.125:
        4 outliers + 7 × 4 = 32 images.

    Parameters
    ----------
    indices:
        Global dataframe indices to sample from (train or val split).

    cluster_ids:
        cluster_id for every image of the full dataframe.

    batch_size:
        Total batch size.

    clusters_per_batch:
        Number of distinct scenes per batch.

    outlier_label:
        cluster_id value that marks outliers.

    outlier_fraction:
        Fraction of the batch filled with outliers.

    seed:
        Random seed; combined with the epoch set by set_epoch.

    drop_last:
        Drop the leftover images that do not form a full batch.

    """

    def __init__(
        self,
        indices: Sequence[int],
        cluster_ids: Sequence[int],
        batch_size: int = 64,
        clusters_per_batch: int = 8,
        outlier_label: int = OUTLIER_LABEL,
        outlier_fraction: float = 0.125,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.indices = np.asarray(indices)
        self.cluster_ids = np.asarray(cluster_ids)

        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch

        self.outlier_label = outlier_label
        self.outlier_per_batch = int(round(batch_size * outlier_fraction))
        # Сцены заполняют только то, что осталось после outliers.
        self.normal_per_batch = batch_size - self.outlier_per_batch
        if self.normal_per_batch % clusters_per_batch != 0:
            raise ValueError(
                f"batch_size - outliers ({self.normal_per_batch}) должно делиться на "
                f"clusters_per_batch ({clusters_per_batch})",
            )
        self.samples_per_cluster = self.normal_per_batch // clusters_per_batch

        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        self.cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        self.outlier_indices: list[int] = []
        for idx in self.indices:
            cluster_id = self.cluster_ids[idx]
            if cluster_id == outlier_label:
                self.outlier_indices.append(int(idx))
            else:
                self.cluster_to_indices[int(cluster_id)].append(int(idx))

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch so that each epoch gets a different batch order.

        Parameters
        ----------
        epoch:
            Current epoch index.

        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """
        Generate batches until there are too few non-empty clusters left.

        Clusters with fewer than samples_per_cluster remaining images are
        sampled with replacement; outliers too, once they run out.

        Returns
        -------
        iterator over lists of global dataframe indices

        """
        rng = np.random.default_rng(self.seed + self.epoch)

        pools = {cluster: rng.permutation(idx).tolist() for cluster, idx in self.cluster_to_indices.items()}
        available_clusters = [cluster for cluster, pool in pools.items() if pool]
        outliers = rng.permutation(self.outlier_indices).tolist()

        while len(available_clusters) >= self.clusters_per_batch:
            selected_clusters = rng.choice(available_clusters, size=self.clusters_per_batch, replace=False)

            batch: list[int] = []
            for cluster in selected_clusters:
                pool = pools[cluster]
                if len(pool) >= self.samples_per_cluster:
                    batch.extend(pool[: self.samples_per_cluster])
                    pools[cluster] = pool[self.samples_per_cluster :]
                else:
                    # Изображений недостаточно — sampling with replacement.
                    batch.extend(rng.choice(pool, size=self.samples_per_cluster, replace=True).tolist())
                    pools[cluster] = []

            if self.outlier_per_batch > 0 and outliers:
                if len(outliers) >= self.outlier_per_batch:
                    batch.extend(outliers[: self.outlier_per_batch])
                    outliers = outliers[self.outlier_per_batch :]
                else:
                    batch.extend(rng.choice(outliers, size=self.outlier_per_batch, replace=True).tolist())

            yield rng.permutation(batch).tolist()

            available_clusters = [cluster for cluster in available_clusters if pools[cluster]]

        if not self.drop_last and available_clusters:
            remainder = [idx for cluster in available_clusters for idx in pools[cluster]]
            yield rng.permutation(remainder).tolist()

    def __len__(self) -> int:
        # Приблизительно: реальное число батчей зависит от размеров кластеров.
        """
        Approximate number of batches per epoch.

        The actual number depends on cluster sizes.

        Returns
        -------
        number of batches

        """
        total_normal = sum(len(idx) for idx in self.cluster_to_indices.values())
        return total_normal // self.normal_per_batch


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build image transforms for training and validation.

    Train:
        - resize, light color jitter, ImageNet normalization.

    Validation:
        - resize, ImageNet normalization.

    Parameters
    ----------
    image_size:
        Side of the square output image.

    Returns
    -------
    train_transform, val_transform

    """
    resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose(
        [
            resize,
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            normalize,
        ],
    )
    val_transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return train_transform, val_transform
