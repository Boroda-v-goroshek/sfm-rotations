"""
Reconstruction on a synthetic scene: known 3D points are projected into known cameras.

No images or model weights: keypoints and matches are exact, so only the
COLMAP part of the pipeline runs, on a single thread.
"""

from pathlib import Path

import numpy as np
import pytest

from sfm_rotations.contracts import is_rotation
from sfm_rotations.geometry import map_images
from sfm_rotations.geometry import shortlist_pairs
from sfm_rotations.metrics import camera_centers
from sfm_rotations.metrics import rotation_angle_deg
from sfm_rotations.metrics import umeyama_alignment

WIDTH, HEIGHT, FOCAL = 640, 480, 500.0


def _look_at_origin(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = -center / np.linalg.norm(center)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z])
    return rotation, -rotation @ center


def _scene(rng: np.random.Generator, n_images: int) -> dict:
    """Cameras on an arc looking at a point cloud; every point is visible in every image."""
    points = rng.uniform(-1.0, 1.0, size=(300, 3))
    angles = np.linspace(-0.6, 0.6, n_images)
    centers = 5.0 * np.stack([np.sin(angles), 0.2 * np.cos(3 * angles), np.cos(angles)], axis=1)
    rotations, translations, keypoints = [], [], []
    for center in centers:
        rotation, translation = _look_at_origin(center)
        in_camera = points @ rotation.T + translation
        keypoints.append(FOCAL * in_camera[:, :2] / in_camera[:, 2:] + [WIDTH / 2, HEIGHT / 2])
        rotations.append(rotation)
        translations.append(translation)
    indices = np.stack([np.arange(len(points))] * 2, axis=1)
    return {
        "names": [f"image_{i}.png" for i in range(n_images)],
        "sizes": np.tile([WIDTH, HEIGHT], (n_images, 1)),
        "keypoints": keypoints,
        "matches": {(i, j): indices for i in range(n_images) for j in range(i + 1, n_images)},
        "rotations": np.stack(rotations),
        "translations": np.stack(translations),
    }


def _map(scene: dict, tmp_path: Path, matches: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    return map_images(
        tmp_path,
        tmp_path,
        scene["names"],
        scene["sizes"],
        scene["keypoints"],
        scene["matches"] if matches is None else matches,
        num_threads=1,
    )


def test_map_images_recovers_poses(tmp_path: Path, rng: np.random.Generator):
    scene = _scene(rng, n_images=6)
    rotations, translations = _map(scene, tmp_path)

    assert np.isfinite(rotations).all()
    assert is_rotation(rotations).all()

    # Мировая система COLMAP произвольна: сравниваем с GT с точностью до подобия.
    centers_pred = camera_centers(rotations, translations)
    centers_gt = camera_centers(scene["rotations"], scene["translations"])
    scale, rotation, shift = umeyama_alignment(centers_pred, centers_gt)
    aligned = scale * centers_pred @ rotation.T + shift
    assert np.linalg.norm(aligned - centers_gt, axis=1).max() < 0.05

    relative_pred = rotations[1:] @ rotations[0].T
    relative_gt = scene["rotations"][1:] @ scene["rotations"][0].T
    assert rotation_angle_deg(relative_pred, relative_gt).max() < 1.0


def test_map_images_keeps_unmatched_image_unregistered(tmp_path: Path, rng: np.random.Generator):
    scene = _scene(rng, n_images=5)
    connected = {pair: value for pair, value in scene["matches"].items() if 4 not in pair}
    rotations, translations = _map(scene, tmp_path, connected)

    assert np.isfinite(rotations[:4]).all()
    assert np.isnan(rotations[4]).all()
    assert np.isnan(translations[4]).all()


def test_map_images_without_matches_returns_nan(tmp_path: Path, rng: np.random.Generator):
    rotations, translations = _map(_scene(rng, n_images=3), tmp_path, matches={})
    assert np.isnan(rotations).all()
    assert np.isnan(translations).all()


@pytest.mark.parametrize(
    ("neighbors", "expected"),
    [
        (1, [(0, 1), (2, 3)]),
        (10, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ],
)
def test_shortlist_pairs(neighbors: int, expected: list[tuple[int, int]]):
    embeddings = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
    assert shortlist_pairs(embeddings, neighbors) == expected
