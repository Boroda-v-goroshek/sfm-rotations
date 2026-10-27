"""
Smoke run of train.py and infer.py on synthetic data, checked against the contracts.

Known gaps between the code and the contracts are marked xfail with their
number from contracts/PROTOCOL.md. xfail is strict: once a gap is fixed, the
test starts to pass and fails the run until the marker is removed.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sfm_rotations import infer
from sfm_rotations import model
from sfm_rotations import train
from sfm_rotations.contracts import OUTLIER_LABEL
from sfm_rotations.contracts import ContractError
from sfm_rotations.contracts import Protocol
from sfm_rotations.contracts import validate_checkpoint
from sfm_rotations.contracts import validate_metrics
from sfm_rotations.contracts import validate_submission
from tests.contracts.synthetic import TinyAutoModel
from tests.contracts.synthetic import load_config
from tests.contracts.synthetic import random_pixels
from tests.contracts.synthetic import write_competition_data
from tests.contracts.synthetic import write_image

InferRunner = Callable[..., Path]


@pytest.fixture(scope="module")
def train_run(tmp_path_factory: pytest.TempPathFactory, repo_root: Path) -> Path:
    """Two epochs of the baseline config on 3 datasets x 5 scenes x 4 images + 2 outliers each."""
    root = tmp_path_factory.mktemp("train_run")
    write_competition_data(root / "data", np.random.default_rng(0), {"ds_a": 5, "ds_b": 5, "ds_c": 5})
    cfg = load_config(
        repo_root,
        "baseline_train",
        data_root=str(root / "data"),
        checkpoints=str(root / "checkpoints"),
        epochs=2,
        logging=False,
        dvc_pull=False,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(model, "AutoModel", TinyAutoModel)
        train.main(cfg)
    return root


@pytest.fixture
def run_infer(
    train_run: Path,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_backbone: None,
) -> InferRunner:
    # В tmp_path нет checkpoints/baseline/best.pt, так что fallback infer.py ничего не подхватит.
    monkeypatch.chdir(tmp_path)

    def run(test_dir: Path, output_name: str = "predictions.json", checkpoint: Path | None = None) -> Path:
        output = test_dir.parent / output_name
        cfg = load_config(
            repo_root,
            "baseline_infer",
            test_dir=str(test_dir),
            checkpoint=str(checkpoint or train_run / "checkpoints" / "best.pt"),
            output_path=str(output),
            dvc_pull=False,
        )
        infer.main(cfg)
        return output

    return run


def _load_checkpoint(path: Path) -> dict:
    # Так же, как infer.py: torch.load по умолчанию грузит с weights_only=True.
    return torch.load(path)


def _write_test_dir(test_dir: Path, images: dict[str, list[np.ndarray]]) -> None:
    for dataset, pixels in images.items():
        for i, image in enumerate(pixels):
            write_image(test_dir / dataset / f"image_{i:02d}.png", image)


# ---------------------------------------------------------------------------
# Конфиги
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["backbone", "embedding_dim", "cluster_distance", "min_cluster_size"])
def test_infer_config_matches_train_config(repo_root: Path, key: str):
    """Пока чекпоинт не хранит конфиг (расхождение #3), эти параметры продублированы и обязаны совпадать."""
    train_cfg, infer_cfg = load_config(repo_root, "baseline_train"), load_config(repo_root, "baseline_infer")
    assert infer_cfg[key] == train_cfg[key]


@pytest.mark.xfail(
    reason="#4: infer по умолчанию ищет checkpoints/best.pt, train пишет в checkpoints/baseline",
    raises=AssertionError,
)
def test_infer_default_checkpoint_is_train_output(repo_root: Path):
    train_cfg, infer_cfg = load_config(repo_root, "baseline_train"), load_config(repo_root, "baseline_infer")
    assert Path(infer_cfg.checkpoint).parent == Path(train_cfg.checkpoints)


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------


def test_train_writes_artifacts(train_run: Path):
    checkpoints = train_run / "checkpoints"
    for name in ["last.pt", "best.pt", "metrics_epoch001.json", "metrics_epoch002.json"]:
        assert (checkpoints / name).is_file(), name
    for name in ["loss.png", "imc_score.png", "rot_maa.png", "training_summary.json"]:
        assert (train_run / "plots" / name).is_file(), name


def test_train_metrics_follow_contract(train_run: Path):
    for path in sorted((train_run / "checkpoints").glob("metrics_epoch*.json")):
        with open(path) as f:
            validate_metrics(json.load(f))


def test_best_checkpoint_selected_by_primary_metric(train_run: Path, protocol: Protocol):
    scores = []
    for path in sorted((train_run / "checkpoints").glob("metrics_epoch*.json")):
        with open(path) as f:
            scores.append(json.load(f)[protocol.primary_metric])
    best = _load_checkpoint(train_run / "checkpoints" / "best.pt")
    assert best["metrics"][protocol.primary_metric] == max(scores)


@pytest.mark.xfail(
    reason="#3: в чекпоинте нет protocol_version, model_config и postprocess",
    raises=ContractError,
)
def test_train_checkpoint_follows_contract(train_run: Path):
    validate_checkpoint(_load_checkpoint(train_run / "checkpoints" / "best.pt"))


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------


def test_infer_loads_train_checkpoint(
    run_infer: InferRunner,
    tmp_path: Path,
    rng: np.random.Generator,
    capsys: pytest.CaptureFixture[str],
):
    _write_test_dir(tmp_path / "test", {"ds_x": [random_pixels(rng) for _ in range(3)], "ds_y": [random_pixels(rng)]})
    output = run_infer(tmp_path / "test")

    assert "Loading checkpoint" in capsys.readouterr().out
    with open(output) as f:
        assert len(json.load(f)) == 4


@pytest.mark.xfail(
    reason="#2: infer пишет JSON, а не submission.csv; у выбросов не nan",
    raises=(ContractError, pd.errors.ParserError),
)
def test_infer_writes_submission(run_infer: InferRunner, tmp_path: Path, rng: np.random.Generator):
    _write_test_dir(tmp_path / "test", {"ds_x": [random_pixels(rng) for _ in range(3)], "ds_y": [random_pixels(rng)]})
    output = run_infer(tmp_path / "test", output_name="submission.csv")

    submission = pd.read_csv(output)
    validate_submission(submission)
    assert sorted(zip(submission["dataset"], submission["image"], strict=True)) == [
        ("ds_x", "image_00.png"),
        ("ds_x", "image_01.png"),
        ("ds_x", "image_02.png"),
        ("ds_y", "image_00.png"),
    ]


@pytest.mark.xfail(reason="#1: infer кластеризует все изображения разом, а не внутри датасета", raises=AssertionError)
def test_infer_clusters_within_dataset(
    run_infer: InferRunner,
    repo_root: Path,
    tmp_path: Path,
    rng: np.random.Generator,
):
    """
    В каждом из двух датасетов по две одинаковые картинки, во всех четырёх — одна и та же.

    Внутри датасета кластер из 2 картинок меньше min_cluster_size, и все они выбросы;
    общая кластеризация склеивает 4 картинки в один кластер.
    """
    assert load_config(repo_root, "baseline_infer").min_cluster_size > 2
    image = random_pixels(rng)
    _write_test_dir(tmp_path / "test", {"ds_x": [image, image], "ds_y": [image, image]})
    output = run_infer(tmp_path / "test")

    # Формат вывода — текущий JSON; после исправления #2 тест читает submission.csv.
    with open(output) as f:
        clusters = [prediction["scene_cluster"] for prediction in json.load(f)]
    assert clusters == [OUTLIER_LABEL] * 4


@pytest.mark.xfail(reason="#4: без чекпоинта infer молча работает на случайных весах", raises=pytest.fail.Exception)
def test_infer_fails_without_checkpoint(run_infer: InferRunner, tmp_path: Path, rng: np.random.Generator):
    _write_test_dir(tmp_path / "test", {"ds_x": [random_pixels(rng)]})
    with pytest.raises(FileNotFoundError):
        run_infer(tmp_path / "test", checkpoint=tmp_path / "missing.pt")
