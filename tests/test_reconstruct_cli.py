"""
Tests for the `reconstruct` CLI (sfm_rotations.cli + sfm_rotations.pipeline).

No DVC, GPU or weight downloads: the DINOv2 backbone is the tiny random stand-in
from tests.contracts.synthetic, and geometry.Reconstructor is replaced with a
fake. The COLMAP part itself is covered by test_geometry.py.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from sfm_rotations import cli
from sfm_rotations import pipeline
from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import validate_submission
from sfm_rotations.pipeline import discover_images
from sfm_rotations.pipeline import validate_result
from tests.contracts.synthetic import TinyAutoModel
from tests.contracts.synthetic import write_image

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_CONFIG = REPO_ROOT / "configs" / "baseline_infer.yaml"


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Случайный tiny-DINOv2, завёрнутый как чекпоинт для infer.load_model."""
    from sfm_rotations import model as model_module
    from sfm_rotations.model import SfMModel

    monkeypatch.setattr(model_module, "AutoModel", TinyAutoModel)
    path = tmp_path / "tiny.pt"
    torch.save({"model": SfMModel(embedding_dim=256).state_dict()}, path)
    return path


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Детерминированные эмбеддинги вместо tiny-модели: normalized grayscale паттерн картинки.

    Непрострелянная tiny-модель почти не различает содержимое (CLS-вектора шумовых
    картинок сливаются), а логика кластеризации/кэша в тестах должна быть проверена
    на управляемых эмбеддингах. load_model при этом остаётся настоящим.
    """

    def fake(model: object, image_paths: list[Path], cfg: object, device: str) -> np.ndarray:
        from PIL import Image

        vectors = []
        for path in image_paths:
            with Image.open(path) as im:
                small = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float64) / 255.0
            small = small - small.mean()
            vectors.append(small.ravel() / np.linalg.norm(small))
        return np.stack(vectors)

    monkeypatch.setattr("sfm_rotations.infer.compute_embeddings", fake)


@pytest.fixture
def fake_reconstructor(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Подмена reconstruction-фабрики pipeline: считает вызовы, позы — единичные.

    geometry не импортируется вообще (тяжёлые pycolmap/LightGlue).
    Ключи dict-состояния: calls (списки путей каждого кластера), fail (Exception или None).
    """
    state: dict[str, Any] = {"calls": [], "fail": None}

    class FakeReconstructor:
        def __init__(self, config: object, device: str):
            self.config = config

        def reconstruct(self, paths: list[Path], embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            state["calls"].append([str(p) for p in paths])
            if state["fail"] is not None:
                raise state["fail"]
            return np.tile(np.eye(3), (len(paths), 1, 1)), np.zeros((len(paths), 3))

    monkeypatch.setattr(pipeline, "_make_reconstructor", lambda config, device: FakeReconstructor(config, device))
    return state


def write_scene(dir_: Path, names_and_pixels: list[tuple[str, np.ndarray]]) -> None:
    for name, pixels in names_and_pixels:
        write_image(dir_ / name, pixels)


def scene_pixels(kind: int, rng: np.random.Generator) -> np.ndarray:
    """
    Различимые картинки: вместо случайного шума (tiny-модель их не разделяет —
    шумовые эмбеддинги сливаются) — детерминированные узоры с разной ориентацией полос.
    """
    base = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    if kind == 1:
        base = base.T.copy()
    elif kind == 2:
        base = np.indices((32, 32)).sum(axis=0) % 2 * 255
    noise = rng.integers(0, 8, size=(32, 32, 1), dtype=np.uint8)
    return (np.repeat(base[:, :, None], 3, axis=2).astype(np.uint16) + noise).astype(np.uint8)


def make_input(root: Path) -> Path:
    """
    input-dir в layout data/test: два датасета, в каждом две сцены по 2 одинаковые картинки + 1 выброс.

    Одинаковые картинки дают косинусное расстояние 0 — кластер из 2 гарантирован
    при --min-cluster-size 2; разные узоры tiny-модель не склеивает.
    """
    rng = np.random.default_rng(7)
    input_dir = root / "group_01"
    for dataset in ("ds_a", "ds_b"):
        s0, s1, s2 = scene_pixels(0, rng), scene_pixels(1, rng), scene_pixels(2, rng)
        write_scene(
            input_dir / dataset,
            [
                ("scene0_00.png", s0),
                ("scene0_01.png", s0),
                ("scene1_00.png", s1),
                ("scene1_01.png", s1),
                ("outlier.png", s2),
            ],
        )
    return input_dir


def run_cli(*args: str) -> int:
    return cli.main(list(args))


def base_args(input_dir: Path, output: Path, checkpoint: Path) -> list[str]:
    return [
        "--input-dir",
        str(input_dir),
        "--output",
        str(output),
        "--config",
        str(INFER_CONFIG),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
        "--min-cluster-size",
        "2",
    ]


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


def test_discover_datasets_from_subfolders(tmp_path: Path):
    write_scene(tmp_path / "in" / "ds_a", [("a.png", np.zeros((8, 8, 3), np.uint8))])
    write_scene(tmp_path / "in" / "ds_b" / "sub", [("b.jpg", np.zeros((8, 8, 3), np.uint8))])
    (tmp_path / "in" / "notes.txt").write_text("not an image")

    found = discover_images(tmp_path / "in", {".png", ".jpg"})
    assert sorted(found) == ["ds_a", "ds_b"]
    assert found["ds_a"]["prefixes"] == ["ds_a/a.png"]
    assert found["ds_b"]["prefixes"] == ["ds_b/sub/b.jpg"]


def test_discover_flat_folder_is_one_dataset(tmp_path: Path):
    write_scene(tmp_path / "group_01", [("a.png", np.zeros((8, 8, 3), np.uint8))])
    found = discover_images(tmp_path / "group_01", {".png"})
    assert found == {"group_01": {"paths": [tmp_path / "group_01/a.png"], "prefixes": ["a.png"]}}


def test_discover_missing_dir(tmp_path: Path):
    from sfm_rotations.pipeline import PipelineError

    with pytest.raises(PipelineError):
        discover_images(tmp_path / "nope", {".png"})


# ---------------------------------------------------------------------------
# Полный прогон CLI
# ---------------------------------------------------------------------------


def test_cli_end_to_end(
    tmp_path: Path,
    tiny_checkpoint: Path,
    fake_embeddings: None,
    fake_reconstructor: dict[str, Any],
):
    input_dir = make_input(tmp_path)
    output = tmp_path / "result.json"
    csv_path = tmp_path / "submission.csv"

    code = run_cli(
        *base_args(input_dir, output, tiny_checkpoint),
        "--output-csv",
        str(csv_path),
    )
    assert code == 0

    with open(output) as f:
        result = json.load(f)
    validate_result(result)

    assert [d["name"] for d in result["datasets"]] == ["ds_a", "ds_b"]
    for entry in result["datasets"]:
        assert entry["num_images"] == 5
        scenes = [item["scene"] for item in entry["images"]]
        assert scenes.count(OUTLIER_SCENE) == 1
        registered = [item for item in entry["images"] if item["rotation_matrix"] is not None]
        assert len(registered) == 4
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert all(item["rotation_matrix"] == identity for item in registered)
        assert all(item["translation_vector"] == [0.0, 0.0, 0.0] for item in registered)

    # 2 датасета x 2 сцены = 4 вызова реконструктора.
    assert len(fake_reconstructor["calls"]) == 4

    submission = pd.read_csv(csv_path)
    validate_submission(submission)
    assert len(submission) == 10

    cache = tmp_path / "result.json.cache"
    assert (cache / "fingerprint.json").is_file()
    assert (cache / "embeddings.npz").is_file()
    assert (cache / "clusters.json").is_file()
    assert len(list(cache.glob("cluster-*.json"))) == 4


def test_cli_resume_reuses_all_stages(
    tmp_path: Path,
    tiny_checkpoint: Path,
    fake_embeddings: None,
    fake_reconstructor: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    input_dir = make_input(tmp_path)
    output = tmp_path / "result.json"
    args = base_args(input_dir, output, tiny_checkpoint)
    assert run_cli(*args) == 0
    with open(output) as f:
        first = json.load(f)
    n_calls = len(fake_reconstructor["calls"])

    # На втором прогоне с --resume заново не считается ничего: ни модель, ни геометрия.
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("resume не должен ничего пересчитывать")

    monkeypatch.setattr("sfm_rotations.infer.compute_embeddings", boom)
    monkeypatch.setattr("sfm_rotations.infer.load_model", boom)
    fake_reconstructor["fail"] = RuntimeError("must not be called")

    assert run_cli(*args, "--resume") == 0
    with open(output) as f:
        second = json.load(f)
    assert second["datasets"] == first["datasets"]
    assert len(fake_reconstructor["calls"]) == n_calls


def test_cli_cache_invalidation(
    tmp_path: Path,
    tiny_checkpoint: Path,
    fake_embeddings: None,
    fake_reconstructor: dict[str, Any],
):
    input_dir = make_input(tmp_path)
    output = tmp_path / "result.json"
    assert run_cli(*base_args(input_dir, output, tiny_checkpoint)) == 0
    n_calls = len(fake_reconstructor["calls"])

    # --resume с другими параметрами кластеризации — ошибка, а не тихий fresh run.
    assert run_cli(*base_args(input_dir, output, tiny_checkpoint), "--cluster-distance", "0.1", "--resume") == 2

    # Без --resume устаревший кэш сбрасывается и все этапы считаются заново.
    assert run_cli(*base_args(input_dir, output, tiny_checkpoint), "--cluster-distance", "0.1") == 0
    assert len(fake_reconstructor["calls"]) == 2 * n_calls


def test_cli_cluster_failure_keeps_run_alive(
    tmp_path: Path,
    tiny_checkpoint: Path,
    fake_embeddings: None,
    fake_reconstructor: dict[str, Any],
):
    fake_reconstructor["fail"] = RuntimeError("colmap exploded")
    input_dir = make_input(tmp_path)
    output = tmp_path / "result.json"
    assert run_cli(*base_args(input_dir, output, tiny_checkpoint)) == 0

    with open(output) as f:
        result = json.load(f)
    validate_result(result)
    assert len(result["meta"]["errors"]) == 4
    for entry in result["datasets"]:
        for item in entry["images"]:
            if item["scene"] != OUTLIER_SCENE:
                assert item["rotation_matrix"] is None
                assert item["translation_vector"] is None


def test_cli_missing_inputs(tmp_path: Path, tiny_checkpoint: Path):
    assert run_cli(*base_args(tmp_path / "nope", tmp_path / "r.json", tiny_checkpoint)) == 2

    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_cli(*base_args(empty, tmp_path / "r.json", tiny_checkpoint)) == 2

    input_dir = make_input(tmp_path)
    assert run_cli(*base_args(input_dir, tmp_path / "r.json", tmp_path / "missing.pt")) == 2
