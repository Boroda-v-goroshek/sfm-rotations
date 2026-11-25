"""
Reconstruct pipeline: the reusable, stage-cached core behind the `reconstruct` CLI.

Stages (each cached under --cache-dir and skipped on --resume):
    1. discover   - find datasets/images under --input-dir (layout as in data/test:
       subfolders are datasets, loose images in the root form a dataset named
       after the input folder);
    2. embeddings - DINOv2 scene embeddings (infer.compute_embeddings);
    3. clusters   - agglomerative cosine clustering within each dataset (metrics.cluster_embeddings);
    4. poses      - per-cluster ALIKED + LightGlue + COLMAP (geometry.Reconstructor),
       one cache file per cluster;
    5. result     - strict JSON (contracts.RESULT_SCHEMA) + optional submission.csv.

The JSON mirrors submission.csv of infer.py 1:1 (same clusters, same poses):
rotation is 3x3 nested lists / translation is 3 numbers, both null when the
image was not registered; scene is "clusterN" or "outliers".
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from sfm_rotations import __version__
from sfm_rotations.contracts import OUTLIER_LABEL
from sfm_rotations.contracts import OUTLIER_SCENE
from sfm_rotations.contracts import RESULT_SCHEMA
from sfm_rotations.contracts import ContractError
from sfm_rotations.contracts import is_rotation
from sfm_rotations.metrics import cluster_embeddings

logger = logging.getLogger("reconstruct")

CACHE_FINGERPRINT_NAME = "fingerprint.json"
CACHE_EMBEDDINGS_NAME = "embeddings.npz"
CACHE_CLUSTERS_NAME = "clusters.json"
CACHE_CLUSTER_PREFIX = "cluster-"


def load_checkpoint_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Read a checkpoint dict without building the model: its model_config and postprocess.

    The same weights_only dance as infer.load_model (checkpoints from train.py carry
    numpy scalars torch>=2.6 rejects by default), and the same fallback to
    checkpoints/dinov2-large/best.pt for old checkpoints without model_config, so that
    the metadata matches the weights infer.load_model will actually load.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise PipelineError(f"Нет чекпоинта {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise PipelineError(f"чекпоинт {path}: ожидался dict")

    if "model_config" not in checkpoint:
        alt_path = Path("checkpoints/dinov2-large/best.pt")
        if not alt_path.exists():
            raise PipelineError(
                f"В чекпоинте {path} нет 'model_config' (старый формат) и нет "
                "checkpoints/dinov2-large/best.pt; укажите актуальный чекпоинт (--checkpoint).",
            )
        logger.warning(
            "чекпоинт %s: нет 'model_config' (старый формат), берём метаданные из %s",
            path,
            alt_path,
        )
        checkpoint = load_checkpoint_meta(alt_path)
        return checkpoint

    return {
        "model_config": checkpoint["model_config"],
        "postprocess": checkpoint.get("postprocess", {}),
    }


def resolve_model_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Take model and clustering defaults from the checkpoint, explicit CLI values win.

    Mirrors infer.py on main: backbone/embedding_dim/pooling/image_size are rebuilt
    from checkpoint['model_config'], cluster_distance/min_cluster_size/linkage come
    from checkpoint['postprocess'] unless the user passed them. Returns a copy of cfg
    with resolved values plus '_checkpoint_meta' for the cache fingerprint and meta.
    """
    resolved = dict(cfg)
    meta = load_checkpoint_meta(cfg["checkpoint"])
    model_config = meta["model_config"]
    postprocess = meta["postprocess"]

    resolved["backbone"] = model_config["backbone"]
    resolved["embedding_dim"] = model_config["embedding_dim"]
    resolved["image_size"] = model_config.get("image_size", 224)
    for key, fallback in (("cluster_distance", 0.3), ("min_cluster_size", 3), ("linkage", "average")):
        if resolved.get(key) is None:
            resolved[key] = postprocess.get(key, fallback)
    resolved["_checkpoint_meta"] = meta
    return resolved


def _make_reconstructor(geometry_cfg: dict[str, Any], device: str) -> Any:
    """
    Build the geometric reconstructor, importing heavy optional deps lazily.

    Tests replace this function to avoid pycolmap/LightGlue.
    """
    from sfm_rotations.geometry import GeometryConfig
    from sfm_rotations.geometry import Reconstructor

    return Reconstructor(GeometryConfig(**geometry_cfg), device)


class PipelineError(RuntimeError):
    """Usage or environment error of the reconstruct pipeline (bad args, stale cache...)."""


# ---------------------------------------------------------------------------
# Этап 1: поиск датасетов и изображений
# ---------------------------------------------------------------------------


def discover_images(input_dir: Path, extensions: set[str]) -> dict[str, dict[str, Any]]:
    """
    Find images of the datasets under input_dir, in the layout of data/test.

    Subfolders (recursively) are datasets named by their top-level folder;
    loose image files in the root form a dataset named after input_dir.

    Parameters
    ----------
    input_dir:
        Folder given to the CLI as --input-dir.

    extensions:
        Lowercased image suffixes to accept.

    Returns
    -------
    dataset name -> {paths: absolute image files, prefixes: name of image relative to input_dir}

    """
    if not input_dir.is_dir():
        raise PipelineError(f"--input-dir {input_dir}: папки не существует")

    datasets: dict[str, dict[str, Any]] = {}
    root_paths: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(input_dir)
        if len(relative.parts) == 1:
            root_paths.append(path)
        else:
            entry = datasets.setdefault(relative.parts[0], {"paths": [], "prefixes": []})
            entry["paths"].append(path)
            entry["prefixes"].append(relative.as_posix())

    if root_paths:
        name = input_dir.resolve().name
        if name in datasets:
            raise PipelineError(
                f"--input-dir {input_dir}: картинка в корне конфликтует с подпапкой '{name}'; переименуйте её",
            )
        datasets[name] = {
            "paths": root_paths,
            "prefixes": [path.name for path in root_paths],
        }
    return dict(sorted(datasets.items()))


# ---------------------------------------------------------------------------
# Кэш этапов
# ---------------------------------------------------------------------------


def _json_dump(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _json_load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _npz_get(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise PipelineError(f"кэш повреждён: нет файла {path}")
    with np.load(path) as npz:
        if key not in npz:
            raise PipelineError(f"кэш повреждён: в {path} нет массива '{key}'")
        return npz[key]


def cache_fingerprint(datasets: dict[str, dict[str, Any]], params: dict[str, Any]) -> str:
    """
    Hash of everything the cached stages depend on: image list and model/clustering/geometry params.

    batch_size, device and cache location are deliberately excluded: they do not
    change results, so resume after switching CPU/GPU or workers keeps working.
    """
    manifest = {
        "pipeline_version": __version__,
        "datasets": {
            name: {
                # Пути относительны input-dir: перенос папки не должен делать кэш непригодным.
                "images": entry["prefixes"],
            }
            for name, entry in sorted(datasets.items())
        },
        "params": params,
    }
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _clear_cache(cache_dir: Path) -> None:
    for path in [
        cache_dir / CACHE_FINGERPRINT_NAME,
        cache_dir / CACHE_EMBEDDINGS_NAME,
        cache_dir / CACHE_CLUSTERS_NAME,
        *cache_dir.glob(f"{CACHE_CLUSTER_PREFIX}*.json"),
    ]:
        path.unlink(missing_ok=True)


def init_cache(cache_dir: Path, fingerprint: str, resume: bool) -> None:
    """
    Make the cache consistent with the current inputs before the stages run.

    Without --resume a stale fingerprint means a fresh run over different inputs:
    the tool-owned cache files are cleared. With --resume it is an error: the user
    asked to continue exactly where the previous run stopped.
    """
    path = cache_dir / CACHE_FINGERPRINT_NAME
    if not path.is_file():
        _json_dump(path, {"fingerprint": fingerprint, "pipeline_version": __version__})
        return
    stored = _json_load(path)
    if stored.get("fingerprint") == fingerprint:
        logger.info("stage cache: fingerprint matches, cached stages will be reused")
        return
    if not resume:
        logger.warning("stage cache: inputs changed, clearing stale cache in %s", cache_dir)
        _clear_cache(cache_dir)
        _json_dump(path, {"fingerprint": fingerprint, "pipeline_version": __version__})
        return
    raise PipelineError(
        f"--resume: кэш {cache_dir} не соответствует текущему входу "
        "(изменились изображения или backbone/cluster_distance/geometry.*); очистите кэш",
    )


# ---------------------------------------------------------------------------
# Результат и его валидация
# ---------------------------------------------------------------------------


def _pose_is_null(rotation: np.ndarray, translation: np.ndarray) -> bool:
    return bool(np.isnan(rotation).all() and np.isnan(translation).all())


def build_result(
    datasets: dict[str, dict[str, Any]],
    clusters: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    translations: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Assemble the strict JSON result, one entry per input image.

    Parameters
    ----------
    datasets:
        output of discover_images merged over all input dirs.
    clusters:
        dataset -> cluster id per image, OUTLIER_LABEL for outliers.
    rotations:
        dataset -> rotations [N, 3, 3], NaN when not registered.
    translations:
        dataset -> translations [N, 3], NaN when not registered.
    meta:
        run metadata (device, params, cache...).

    Returns
    -------
    json-serializable dict following contracts.RESULT_SCHEMA

    """
    dataset_entries = []
    for name, entry in datasets.items():
        images = []
        for prefix, cluster, rotation, translation in zip(
            entry["prefixes"],
            clusters[name],
            rotations[name],
            translations[name],
            strict=True,
        ):
            registered = bool(np.isfinite(rotation).all() and np.isfinite(translation).all())
            images.append(
                {
                    "image": prefix,
                    "scene": OUTLIER_SCENE if cluster == OUTLIER_LABEL else f"cluster{int(cluster)}",
                    "rotation_matrix": None
                    if not registered
                    else [[float(value) for value in row] for row in rotation],
                    "translation_vector": None if not registered else [float(value) for value in translation],
                },
            )
        null_poses = np.array([_pose_is_null(r, t) for r, t in zip(rotations[name], translations[name], strict=True)])
        dataset_entries.append(
            {
                "name": name,
                "num_images": len(entry["paths"]),
                "num_clusters": int((np.unique(clusters[name]) != OUTLIER_LABEL).sum()),
                "num_outliers": int((clusters[name] == OUTLIER_LABEL).sum()),
                "num_registered": int((~null_poses).sum()),
                "images": images,
            },
        )

    return {
        "schema_version": RESULT_SCHEMA,
        "pipeline_version": __version__,
        "meta": meta,
        "datasets": dataset_entries,
    }


def _validate_pose(entry_name: str, image: str, rotation: Any, translation: Any) -> None:
    """
    Check one image pose against submission.csv semantics: all numbers or all null.

    Raises
    ------
    ContractError
        On any violation.

    """
    where = f"result {entry_name}/{image}"
    if rotation is None and translation is None:
        return
    if rotation is None or translation is None:
        raise ContractError(f"{where}: поза должна быть целиком числами либо целиком null")
    if len(rotation) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in rotation):
        raise ContractError(f"{where}: rotation_matrix должна быть 3x3")
    if len(translation) != 3:
        raise ContractError(f"{where}: translation_vector должен быть длины 3")
    rotation_array = np.array(rotation, dtype=float)
    if not (np.isfinite(rotation_array).all() and np.isfinite(translation).all()):
        raise ContractError(f"{where}: в позе есть не-числа")
    if not bool(is_rotation(rotation_array)):
        raise ContractError(f"{where}: rotation_matrix не является поворотом")


def _validate_image(entry_name: str, item: Any, seen_images: set[str]) -> None:
    """Check one image entry of a dataset and remember its path for duplicate detection."""
    image = item.get("image")
    if not isinstance(image, str) or not image or Path(image).is_absolute():
        raise ContractError(f"result {entry_name}: image {image!r} — не относительный путь")
    if image in seen_images:
        raise ContractError(f"result {entry_name}: повтор изображения {image}")
    seen_images.add(image)

    scene = item.get("scene")
    rotation, translation = item.get("rotation_matrix"), item.get("translation_vector")
    if scene == OUTLIER_SCENE:
        if rotation is not None or translation is not None:
            raise ContractError(f"result {entry_name}/{image}: у выброса поза должна быть null")
        return
    if not isinstance(scene, str) or not scene.startswith("cluster"):
        raise ContractError(f"result {entry_name}/{image}: scene должен быть 'clusterN' или 'outliers'")
    _validate_pose(entry_name, image, rotation, translation)


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ContractError(f"{where}: нет ключей {missing}")


def validate_result(result: dict[str, Any]) -> None:
    """
    Check a result dict against RESULT_SCHEMA: structure plus the pose semantics of submission.csv.

    Raises
    ------
    ContractError
        On any violation.

    """
    if not isinstance(result, dict):
        raise ContractError("result: ожидался JSON-объект")
    _require_keys(result, ("schema_version", "pipeline_version", "meta", "datasets"), "result")
    if result["schema_version"] != RESULT_SCHEMA:
        raise ContractError(f"result: schema_version {result['schema_version']}, ожидался {RESULT_SCHEMA}")
    if not isinstance(result["datasets"], list) or not result["datasets"]:
        raise ContractError("result: datasets должен быть непустым списком")

    seen_datasets = set()
    for entry in result["datasets"]:
        if not isinstance(entry, dict):
            raise ContractError("result: элемент datasets не объект")
        _require_keys(entry, ("name", "num_images", "images"), "result: элемент datasets")
        if entry["name"] in seen_datasets:
            raise ContractError(f"result: датасет '{entry['name']}' повторяется")
        seen_datasets.add(entry["name"])
        if not isinstance(entry["images"], list) or not entry["images"]:
            raise ContractError(f"result {entry['name']}: images пустой")
        if entry["num_images"] != len(entry["images"]):
            raise ContractError(f"result {entry['name']}: num_images != len(images)")

        seen_images: set[str] = set()
        for item in entry["images"]:
            _validate_image(entry["name"], item, seen_images)


def _format_pose(values: np.ndarray) -> str:
    return ";".join("nan" if np.isnan(value) else f"{value:.9f}" for value in values.ravel())


def to_submission_csv(result: dict[str, Any]) -> pd.DataFrame:
    """
    Export the JSON result as competition submission.csv (same rows as infer.py).

    The image column is relative to the dataset folder, matching infer.py output
    for the data/test layout.

    Parameters
    ----------
    result:
        Validated result dict.

    Returns
    -------
    dataframe with SUBMISSION_COLUMNS

    """
    rows = []
    for entry in result["datasets"]:
        prefix = entry["name"] + "/"
        for item in entry["images"]:
            rotation, translation = item["rotation_matrix"], item["translation_vector"]
            # infer.py пишет image относительно папки датасета; наш путь относительно input-dir.
            image = item["image"][len(prefix) :] if item["image"].startswith(prefix) else item["image"]
            rows.append(
                {
                    "dataset": entry["name"],
                    "scene": item["scene"],
                    "image": image,
                    "rotation_matrix": _format_pose(np.full(9, np.nan))
                    if rotation is None
                    else _format_pose(np.array(rotation, dtype=float)),
                    "translation_vector": _format_pose(np.full(3, np.nan))
                    if translation is None
                    else _format_pose(np.array(translation, dtype=float)),
                },
            )
    return pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])


# ---------------------------------------------------------------------------
# Основной workflow
# ---------------------------------------------------------------------------


def _stage_embeddings(
    cache_dir: Path,
    all_paths: list[Path],
    resume: bool,
    device: str,
    compute: Any,
) -> np.ndarray:
    """
    Cached embeddings of all images (stage 2 of the pipeline).

    Parameters
    ----------
    cache_dir:
        Stage cache folder.

    all_paths:
        Every image of every dataset, in dataset order.

    resume:
        Reuse embeddings.npz when it exists.

    compute:
        Zero-arg callable producing the [N, D] embeddings.

    device:
        Torch device, for the log line.

    Returns
    -------
    L2-normalized embeddings [N, D]

    """
    embeddings_path = cache_dir / CACHE_EMBEDDINGS_NAME
    if resume and embeddings_path.is_file():
        embeddings = _npz_get(embeddings_path, "embeddings")
        if embeddings.shape[0] != len(all_paths):
            raise PipelineError("кэш эмбеддингов не совпадает по размеру со списком изображений")
        logger.info("stage embeddings: cached (%d x %d)", *embeddings.shape)
    else:
        logger.info("stage embeddings: %d images on %s", len(all_paths), device)
        embeddings = compute()
        np.savez_compressed(embeddings_path, embeddings=embeddings)
    return embeddings


def _stage_clusters(
    cache_dir: Path,
    datasets: dict[str, dict[str, Any]],
    embeddings: np.ndarray,
    cfg: dict[str, Any],
    resume: bool,
) -> dict[str, np.ndarray]:
    """
    Cached per-dataset clustering of embeddings (stage 3 of the pipeline).

    Parameters
    ----------
    cache_dir:
        Stage cache folder.

    datasets:
        Discovered datasets.

    embeddings:
        Embeddings of all images in dataset order.

    cfg:
        Pipeline config (cluster_distance, min_cluster_size).

    resume:
        Reuse clusters.json when it exists.

    Returns
    -------
    dataset -> cluster id per image, OUTLIER_LABEL for outliers

    """
    clusters_path = cache_dir / CACHE_CLUSTERS_NAME
    if resume and clusters_path.is_file():
        clusters = {name: np.array(values, dtype=int) for name, values in _json_load(clusters_path).items()}
        if sorted(clusters) != sorted(datasets) or any(
            len(clusters[name]) != len(datasets[name]["paths"]) for name in datasets
        ):
            raise PipelineError("кэш кластеров не совпадает со списком изображений")
        logger.info("stage clusters: cached")
        return clusters

    logger.info(
        "stage clusters: agglomerative cosine within each dataset (distance=%s, min_size=%s, linkage=%s)",
        cfg["cluster_distance"],
        cfg["min_cluster_size"],
        cfg["linkage"],
    )
    clusters = {}
    offset = 0
    for name, entry in datasets.items():
        local = embeddings[offset : offset + len(entry["paths"])]
        clusters[name] = cluster_embeddings(local, cfg["cluster_distance"], cfg["min_cluster_size"], cfg["linkage"])
        offset += len(entry["paths"])
    _json_dump(clusters_path, {name: values.tolist() for name, values in clusters.items()})
    return clusters


def _reconstruct_cluster(
    cache_dir: Path,
    name: str,
    cluster: int,
    entry: dict[str, Any],
    indices: np.ndarray,
    embeddings: np.ndarray,
    fingerprint: str,
    resume: bool,
    get_reconstructor: Any,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> str | None:
    """
    Reconstruct one cluster from cache or geometry; writes the per-cluster cache file.

    Returns
    -------
    error string if reconstruction failed (poses stay NaN), else None

    """
    local_prefixes = [entry["prefixes"][i] for i in indices]
    cache_path = cache_dir / f"{CACHE_CLUSTER_PREFIX}{name}__{cluster}.json"
    if resume and cache_path.is_file():
        try:
            payload = _json_load(cache_path)
            cached_ok = payload.get("fingerprint") == fingerprint and payload.get("cluster_images") == local_prefixes
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("cluster %s/%s: повреждённый кэш (%s), пересчитываем", name, cluster, e)
            cached_ok = False
        if cached_ok:
            rotations[indices] = np.array(payload["rotations"], dtype=float)
            translations[indices] = np.array(payload["translations"], dtype=float)
            logger.info("stage poses: cached cluster %s/%s (%d images)", name, cluster, len(indices))
            return None
        logger.warning("cluster %s/%s: кэш устарел, пересчитываем", name, cluster)

    cluster_paths = [entry["paths"][i] for i in indices]
    logger.info("stage poses: reconstructing cluster %s/%s (%d images)", name, cluster, len(indices))
    try:
        rotation, translation = get_reconstructor().reconstruct(cluster_paths, embeddings[indices])
    except Exception as e:  # noqa: BLE001 — один плохой кластер не роняет весь прогон
        logger.warning("cluster %s/%s failed: %s (позы остались NaN)", name, cluster, e)
        return f"{type(e).__name__}: {e}"
    rotations[indices] = rotation
    translations[indices] = translation
    _json_dump(
        cache_path,
        {
            "fingerprint": fingerprint,
            "cluster_images": local_prefixes,
            "rotations": rotation.tolist(),
            "translations": translation.tolist(),
        },
    )
    return None


def _stage_poses(
    cache_dir: Path,
    datasets: dict[str, dict[str, Any]],
    clusters: dict[str, np.ndarray],
    embeddings: np.ndarray,
    fingerprint: str,
    resume: bool,
    get_reconstructor: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Cached per-cluster geometric reconstruction (stage 4 of the pipeline).

    Returns
    -------
    rotations, translations per dataset (NaN for unregistered), list of cluster errors

    """
    rotations = {name: np.full((len(entry["paths"]), 3, 3), np.nan) for name, entry in datasets.items()}
    translations = {name: np.full((len(entry["paths"]), 3), np.nan) for name, entry in datasets.items()}
    errors: list[dict[str, Any]] = []
    for name, entry in datasets.items():
        for cluster in np.unique(clusters[name]):
            if cluster == OUTLIER_LABEL:
                continue
            indices = np.flatnonzero(clusters[name] == cluster)
            error = _reconstruct_cluster(
                cache_dir,
                name,
                int(cluster),
                entry,
                indices,
                embeddings,
                fingerprint,
                resume,
                get_reconstructor,
                rotations[name],
                translations[name],
            )
            if error is not None:
                errors.append({"dataset": name, "cluster": int(cluster), "error": error})
    return rotations, translations, errors


def run(cfg: dict[str, Any], resume: bool) -> dict[str, Any]:
    """
    Run all stages over the given input dirs and return the validated JSON result.

    Parameters
    ----------
    cfg:
        Normalized config from the CLI: input_dirs, cache_dir, device, checkpoint,
        backbone, embedding_dim, batch_size, num_workers, cluster_distance,
        min_cluster_size, geometry (dict).
    resume:
        Reuse the cached stages that are already on disk.

    Returns
    -------
    result dict following contracts.RESULT_SCHEMA

    """
    torch.set_grad_enabled(False)

    cache_dir = Path(cfg["cache_dir"])
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    extensions = {suffix.lower() for suffix in cfg["image_extensions"]}

    datasets: dict[str, dict[str, Any]] = {}
    for input_dir in cfg["input_dirs"]:
        for name, entry in discover_images(Path(input_dir), extensions).items():
            if name in datasets:
                raise PipelineError(
                    f"имя датасета '{name}' встречается в нескольких --input-dir; переименуйте вложенные папки",
                )
            datasets[name] = entry
    if not datasets:
        raise PipelineError(
            f"изображения не найдены в {cfg['input_dirs']} (расширения: {sorted(extensions)})",
        )

    all_paths = [path for entry in datasets.values() for path in entry["paths"]]
    cfg = resolve_model_params(cfg)
    fingerprint_params = {
        key: cfg[key] for key in ("cluster_distance", "min_cluster_size", "linkage", "backbone", "embedding_dim")
    } | {
        f"geometry.{key}": value
        for key, value in cfg["geometry"].items()
        # num_threads — параметр исполнения, на результат не влияет (как device).
        if key != "num_threads"
    }
    fingerprint = cache_fingerprint(datasets, fingerprint_params)

    cache_dir.mkdir(parents=True, exist_ok=True)
    init_cache(cache_dir, fingerprint, resume)

    omega_cfg = OmegaConf.create(cfg)
    state: dict[str, Any] = {}

    def get_reconstructor() -> Any:
        if "reconstructor" not in state:
            state["reconstructor"] = _make_reconstructor(cfg["geometry"], device)
        return state["reconstructor"]

    def compute() -> np.ndarray:
        # Ленивые импорты: infer тянет geometry/pycolmap/LightGlue — они нужны
        # только на своих этапах, а не при любом импорте CLI.
        from sfm_rotations.infer import compute_embeddings
        from sfm_rotations.infer import load_model

        model, _ = load_model(omega_cfg, device)
        return compute_embeddings(model, all_paths, cfg["image_size"], omega_cfg, device)

    embeddings = _stage_embeddings(cache_dir, all_paths, resume, device, compute)
    clusters = _stage_clusters(cache_dir, datasets, embeddings, cfg, resume)
    rotations, translations, errors = _stage_poses(
        cache_dir, datasets, clusters, embeddings, fingerprint, resume, get_reconstructor
    )

    meta = {
        "input_dirs": [str(Path(path).resolve()) for path in cfg["input_dirs"]],
        "device": device,
        "checkpoint": str(Path(cfg["checkpoint"]).resolve()),
        "model_config": cfg["_checkpoint_meta"]["model_config"],
        "params": fingerprint_params,
        "cache_dir": str(cache_dir.resolve()),
        "cache_fingerprint": fingerprint,
        "errors": errors,
    }
    result = build_result(datasets, clusters, rotations, translations, meta)
    validate_result(result)
    logger.info(
        "done: %d datasets, %d images, %d registered poses",
        len(datasets),
        len(all_paths),
        sum(int(np.isfinite(value).all(axis=(1, 2)).sum()) for value in rotations.values()),
    )
    return result


def write_json(result: dict[str, Any], path: Path) -> None:
    """Atomically write the result JSON (tmp + replace), creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.replace(path)
