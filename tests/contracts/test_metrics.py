"""
Evaluation metrics: golden values and invariances.

Any change that breaks these tests changes the metric, i.e. the protocol:
update the expected values only together with a protocol version bump.
"""

from dataclasses import replace

import numpy as np
import pytest

from sfm_rotations.contracts import PER_DATASET_METRIC_KEY
from sfm_rotations.contracts import validate_metrics
from sfm_rotations.metrics import EvalOutputs
from sfm_rotations.metrics import compute_all_metrics
from tests.contracts.synthetic import make_eval_outputs
from tests.contracts.synthetic import random_rotations
from tests.contracts.synthetic import transform_world

DATASETS = {"ds_a": 3, "ds_b": 2}

# Снимок метрик текущего кода на зашумлённых позах (seed 0). Меняется только вместе с версией протокола.
NOISY_SNAPSHOT = {
    "pose/rel_rot_err_median_deg": 16.816437738753226,
    "pose/rel_rot_mAA": 0.42,
    "emb/ARI": 1.0,
    "emb/recall@1": 1.0,
    "imc/mAA": 0.2861111111111111,
    "imc/clustering": 1.0,
    "imc/score": 0.44491715181370356,
    "imc/ds_a/score": 0.4482758620689655,
    "imc/ds_b/score": 0.4415584415584416,
}


def _noisy(seed: int = 0) -> tuple[EvalOutputs, dict]:
    rng = np.random.default_rng(seed)
    return make_eval_outputs(rng, DATASETS, rotation_noise_deg=20.0, translation_noise=0.3)


def _per_scene(outputs: EvalOutputs) -> list[np.ndarray]:
    return [np.flatnonzero(outputs.scene_labels == label) for label in np.unique(outputs.scene_labels) if label >= 0]


def test_perfect_predictions_score_one(rng: np.random.Generator):
    outputs, thresholds = make_eval_outputs(rng, DATASETS)
    metrics = compute_all_metrics(outputs, thresholds)

    for key in ["imc/score", "imc/mAA", "imc/clustering", "emb/ARI", "emb/recall@1", "pose/rel_rot_mAA"]:
        assert metrics[key] == pytest.approx(1.0), key
    assert metrics["pose/rel_rot_err_median_deg"] == pytest.approx(0.0, abs=1e-3)


def test_merged_scenes_halve_clustering(rng: np.random.Generator):
    """Две сцены по 4 картинки в одном кластере: mAA = 1, clustering = 8 / 16, score = 2/3."""
    outputs, thresholds = make_eval_outputs(rng, {"ds_a": 2}, images_per_scene=4, outliers_per_dataset=0)
    outputs = replace(outputs, embeddings=np.ones_like(outputs.embeddings))
    metrics = compute_all_metrics(outputs, thresholds)

    assert metrics["imc/mAA"] == pytest.approx(1.0)
    assert metrics["imc/clustering"] == pytest.approx(0.5)
    assert metrics["imc/score"] == pytest.approx(2 / 3)
    assert metrics["emb/ARI"] == pytest.approx(0.0)


def test_all_outliers_score_zero(rng: np.random.Generator):
    outputs, thresholds = make_eval_outputs(rng, DATASETS)
    dim = len(outputs.embeddings)
    outputs = replace(outputs, embeddings=np.eye(dim, dtype=np.float32))
    metrics = compute_all_metrics(outputs, thresholds)

    assert metrics["imc/score"] == 0.0
    assert metrics["imc/mAA"] == 0.0
    assert metrics["imc/clustering"] == 0.0


def test_noisy_snapshot():
    metrics = compute_all_metrics(*_noisy())
    assert {key: metrics[key] for key in NOISY_SNAPSHOT} == pytest.approx(NOISY_SNAPSHOT, rel=1e-6)


def test_invariant_to_row_order():
    outputs, thresholds = _noisy()
    order = np.random.default_rng(1).permutation(len(outputs.scene_labels))
    shuffled = EvalOutputs(**{field: getattr(outputs, field)[order] for field in outputs.__dataclass_fields__})

    expected = compute_all_metrics(outputs, thresholds)
    assert compute_all_metrics(shuffled, thresholds) == pytest.approx(expected, rel=1e-9)


def test_invariant_to_gt_world_frame():
    """У каждой сцены своя система координат: жёсткое движение GT-сцены метрики не меняет."""
    outputs, thresholds = _noisy()
    rng = np.random.default_rng(1)
    rotations, translations = outputs.rotations_gt.copy(), outputs.translations_gt.copy()
    for idx in _per_scene(outputs):
        rotations[idx], translations[idx] = transform_world(
            rotations[idx],
            translations[idx],
            random_rotations(rng, 1)[0],
            rng.normal(scale=10.0, size=3),
        )
    moved = replace(outputs, rotations_gt=rotations, translations_gt=translations)

    expected = compute_all_metrics(outputs, thresholds)
    assert compute_all_metrics(moved, thresholds) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_invariant_to_predicted_similarity():
    """Предсказание сцены сравнивается с GT после выравнивания подобием: масштаб и сдвиг не важны."""
    outputs, thresholds = _noisy()
    rng = np.random.default_rng(1)
    rotations, translations = outputs.rotations_pred.copy(), outputs.translations_pred.copy()
    for idx in _per_scene(outputs):
        rotations[idx], translations[idx] = transform_world(
            rotations[idx],
            translations[idx],
            random_rotations(rng, 1)[0],
            rng.normal(scale=10.0, size=3),
            scale=rng.uniform(0.1, 10.0),
        )
    moved = replace(outputs, rotations_pred=rotations, translations_pred=translations)

    expected = compute_all_metrics(outputs, thresholds)
    assert compute_all_metrics(moved, thresholds) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_metric_keys_follow_contract():
    outputs, thresholds = _noisy()
    metrics = compute_all_metrics(outputs, thresholds) | {"epoch": 1, "train/loss": 0.0, "val/loss": 0.0}
    validate_metrics(metrics)

    per_dataset = {key for key in metrics if key.count("/") > 1}
    assert all(PER_DATASET_METRIC_KEY.fullmatch(key) for key in per_dataset)
    assert {f"imc/{dataset}/score" for dataset in DATASETS} <= per_dataset


def test_metric_values_are_plain_floats():
    """Метрики кладутся в чекпоинт: numpy-скаляры ломают torch.load(weights_only=True) в infer.py."""
    metrics = compute_all_metrics(*_noisy())
    assert {key: type(value) for key, value in metrics.items()} == dict.fromkeys(metrics, float)
