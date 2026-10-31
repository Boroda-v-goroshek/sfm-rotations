"""
Camera poses from local features and incremental SfM.

Absolute pose regression does not generalize: world frames of scenes are
arbitrary, so a network cannot predict them for unseen scenes. Here DINOv2 only
groups images into clusters, and each cluster is reconstructed geometrically:
    - shortlist image pairs by embedding similarity;
    - extract ALIKED keypoints and match them with LightGlue;
    - verify the matches and run the COLMAP incremental mapper.

Images the mapper does not register get NaN poses.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch
from lightglue import ALIKED
from lightglue import LightGlue
from lightglue.utils import load_image
from lightglue.utils import rbd
from tqdm import tqdm


@dataclass(frozen=True)
class GeometryConfig:
    """
    Parameters of the geometric reconstruction.

    Parameters
    ----------
    neighbors:
        Number of most similar images each image is matched with.

    max_keypoints:
        Maximum number of ALIKED keypoints per image.

    resize:
        Longest image side for feature extraction.

    min_matches:
        Minimum number of matches for an image pair to be kept.

    num_threads:
        CPU threads of the mapper; -1 uses all of them.

    """

    neighbors: int = 20
    max_keypoints: int = 2048
    resize: int = 1024
    min_matches: int = 15
    num_threads: int = -1


def shortlist_pairs(embeddings: np.ndarray, neighbors: int) -> list[tuple[int, int]]:
    """
    Select image pairs to match by embedding similarity.

    Parameters
    ----------
    embeddings:
        L2-normalized embeddings [N, D].

    neighbors:
        Number of nearest images per image.

    Returns
    -------
    sorted unique pairs (i, j) with i < j

    """
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    # argsort, а не argpartition: выбор среди равных сходств должен быть детерминированным.
    # Сама картинка стоит последней (сходство -inf), поэтому срез не длиннее N - 1.
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, : min(neighbors, len(embeddings) - 1)]
    return sorted({(min(i, int(j)), max(i, int(j))) for i, row in enumerate(nearest) for j in row})


def write_database(
    path: Path,
    names: list[str],
    sizes: np.ndarray,
    keypoints: list[np.ndarray],
    matches: dict[tuple[int, int], np.ndarray],
) -> None:
    """
    Write images, keypoints and raw matches to a new COLMAP database.

    Every image gets its own camera: collections are unordered and mix devices.

    Parameters
    ----------
    path:
        Database file.

    names:
        Image names relative to the image folder [N].

    sizes:
        Image sizes (width, height) [N, 2].

    keypoints:
        Keypoints [K, 2] of each image, in COLMAP coordinates (pixel corner origin).

    matches:
        (i, j) -> keypoint index pairs [M, 2].

    """
    image_ids = []
    with pycolmap.Database.open(path) as database:
        for name, (width, height), points in zip(names, sizes, keypoints, strict=True):
            # Стандартная начальная оценка фокуса COLMAP при неизвестной калибровке.
            camera = pycolmap.Camera.create_from_model_id(
                pycolmap.INVALID_CAMERA_ID,
                pycolmap.CameraModelId.SIMPLE_RADIAL,
                1.2 * max(width, height),
                int(width),
                int(height),
            )
            camera_id = database.write_camera(camera)
            image_id = database.write_image(pycolmap.Image(name=name, camera_id=camera_id))
            database.write_keypoints(image_id, points.astype(np.float32))
            image_ids.append(image_id)

            # С COLMAP 3.12 маппер работает с кадрами: одна камера — тривиальный rig из одного сенсора.
            sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
            rig = pycolmap.Rig()
            rig.add_ref_sensor(sensor)
            frame = pycolmap.Frame()
            frame.rig_id = database.write_rig(rig)
            frame.add_data_id(pycolmap.data_t(sensor, image_id))
            database.write_frame(frame)

        for (i, j), pair_matches in matches.items():
            database.write_matches(image_ids[i], image_ids[j], pair_matches.astype(np.uint32))


def map_images(
    workdir: Path,
    image_dir: Path,
    names: list[str],
    sizes: np.ndarray,
    keypoints: list[np.ndarray],
    matches: dict[tuple[int, int], np.ndarray],
    num_threads: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Verify matches and reconstruct the largest model of a group of images.

    Parameters
    ----------
    workdir:
        Empty folder for the database and mapper output.

    image_dir:
        Folder names are relative to.

    names:
        Image names [N].

    sizes:
        Image sizes (width, height) [N, 2].

    keypoints:
        Keypoints [K, 2] of each image, in COLMAP coordinates.

    matches:
        (i, j) -> keypoint index pairs [M, 2].

    num_threads:
        CPU threads of the mapper.

    Returns
    -------
    rotations [N, 3, 3], translations [N, 3] (world -> camera); NaN for unregistered images

    """
    rotations = np.full((len(names), 3, 3), np.nan)
    translations = np.full((len(names), 3), np.nan)
    if not matches:
        return rotations, translations

    database_path = workdir / "database.db"
    pairs_path = workdir / "pairs.txt"
    write_database(database_path, names, sizes, keypoints, matches)
    pairs_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in matches))
    pycolmap.verify_matches(database_path, pairs_path)

    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = 3
    options.num_threads = num_threads
    options.extract_colors = False
    models = pycolmap.incremental_mapping(database_path, image_dir, workdir / "models", options)
    if not models:
        return rotations, translations

    # Маппер может разбить группу на несколько моделей, а кластеру сабмита соответствует одна поза на картинку.
    model = max(models.values(), key=lambda reconstruction: reconstruction.num_reg_images())
    index = {name: i for i, name in enumerate(names)}
    for image in model.images.values():
        if image.has_pose:
            pose = image.cam_from_world()
            rotations[index[image.name]] = pose.rotation.matrix()
            translations[index[image.name]] = pose.translation
    return rotations, translations


class Reconstructor:
    """
    Reconstructs clusters of images; keeps ALIKED and LightGlue loaded between them.

    Parameters
    ----------
    config:
        Reconstruction parameters.

    device:
        Torch device.

    """

    def __init__(self, config: GeometryConfig, device: str):
        self.config = config
        self.device = device
        self.extractor = ALIKED(max_num_keypoints=config.max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)

    @torch.no_grad()
    def _extract(self, paths: list[Path]) -> list[dict[str, torch.Tensor]]:
        features = []
        for path in tqdm(paths, desc="features", leave=False):
            image = load_image(path).to(self.device)
            feats = self.extractor.extract(image, resize=self.config.resize)
            features.append({key: value.cpu() for key, value in feats.items()})
        return features

    @torch.no_grad()
    def _match(
        self,
        features: list[dict[str, torch.Tensor]],
        pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], np.ndarray]:
        matches = {}
        for i, j in tqdm(pairs, desc="matches", leave=False):
            data = {
                "image0": {key: value.to(self.device) for key, value in features[i].items()},
                "image1": {key: value.to(self.device) for key, value in features[j].items()},
            }
            pair_matches = rbd(self.matcher(data))["matches"].cpu().numpy()
            if len(pair_matches) >= self.config.min_matches:
                matches[(i, j)] = pair_matches
        return matches

    def reconstruct(self, paths: list[Path], embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct one cluster of images.

        Parameters
        ----------
        paths:
            Image files of the cluster [N].

        embeddings:
            L2-normalized embeddings of these images [N, D].

        Returns
        -------
        rotations [N, 3, 3], translations [N, 3]; NaN for unregistered images

        """
        image_dir = Path(os.path.commonpath([path.parent for path in paths]))
        names = [path.relative_to(image_dir).as_posix() for path in paths]

        features = self._extract(paths)
        matches = self._match(features, shortlist_pairs(embeddings, self.config.neighbors))
        sizes = np.stack([feats["image_size"][0].numpy() for feats in features])
        # LightGlue считает от центра пикселя, COLMAP — от его угла.
        keypoints = [feats["keypoints"][0].numpy() + 0.5 for feats in features]

        with tempfile.TemporaryDirectory() as workdir:
            return map_images(Path(workdir), image_dir, names, sizes, keypoints, matches, self.config.num_threads)
