"""
`reconstruct` CLI: images folder(s) in, strict JSON out.

    reconstruct --input-dir data/group_01/ --output result.json
    reconstruct --input-dir a/ --input-dir b/ --output result.json --verbose
    reconstruct --input-dir data/group_01/ --output result.json --resume

Exit codes: 0 ok (per-cluster geometry failures are reported in meta.errors),
2 usage/cache error, 1 unexpected failure.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import click
from omegaconf import OmegaConf

from sfm_rotations import __version__
from sfm_rotations.contracts import ContractError
from sfm_rotations.pipeline import PipelineError
from sfm_rotations.pipeline import run
from sfm_rotations.pipeline import to_submission_csv
from sfm_rotations.pipeline import write_json

logger = logging.getLogger("reconstruct")

# Дефолтный чекпоинт: baseline (dinov2-base). backbone/pooling/image_size и пороги
# кластеризации берутся из самого чекпоинта (model_config/postprocess), как в infer.py;
# --config опционален и нужен, чтобы задать свои дефолты geometry/batch.
DEFAULT_CHECKPOINT = "checkpoints/baseline/best.pt"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0

# Явные флаги, чьи значения перебивают --config; остальные опции — только из --config.
_GEOMETRY_FLAGS = ("neighbors", "max_keypoints", "resize", "min_matches", "num_threads")


def _load_params(options: dict[str, Any]) -> dict[str, Any]:
    """
    Merge optional --config defaults with the explicit CLI flags into the pipeline cfg.

    Model architecture (backbone/embedding_dim/image_size) and clustering defaults
    (cluster_distance/min_cluster_size/linkage) are resolved from the checkpoint in
    pipeline.resolve_model_params; here they stay None unless passed explicitly.
    """
    config_path = Path(options["config"]) if options["config"] else None
    defaults = OmegaConf.to_container(OmegaConf.load(config_path)) if config_path and config_path.is_file() else {}
    if not isinstance(defaults, dict):
        raise PipelineError(f"--config {config_path}: ожидался словарь")
    geometry = defaults.get("geometry", {})

    def value_or(key: str, fallback: Any) -> Any:
        return options[key] if options[key] is not None else defaults.get(key, fallback)

    geometry_overrides = {key: options[key] for key in _GEOMETRY_FLAGS if options[key] is not None}

    output = Path(options["output"])
    cfg: dict[str, Any] = {
        "input_dirs": list(options["input_dirs"]),
        "output": str(output),
        "cache_dir": options["cache_dir"] or str(output.parent / (output.name + ".cache")),
        "device": options["device"],
        "checkpoint": options["checkpoint"] or defaults.get("checkpoint", DEFAULT_CHECKPOINT),
        "batch_size": value_or("batch_size", DEFAULT_BATCH_SIZE),
        "num_workers": value_or("num_workers", DEFAULT_NUM_WORKERS),
        "cluster_distance": options["cluster_distance"],
        "min_cluster_size": options["min_cluster_size"],
        "linkage": options["linkage"],
        "image_extensions": defaults.get("image_extensions", [".png", ".jpg", ".jpeg"]),
        "geometry": geometry | geometry_overrides,
    }
    return cfg


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.command(name="reconstruct", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-dir",
    "input_dirs",
    metavar="DIR",
    multiple=True,
    required=True,
    help="папка с изображениями; подпапки = датасеты (как data/test), картинки в корне "
    "= один датасет с именем папки; можно передавать несколько раз",
)
@click.option("--output", required=True, metavar="FILE", help="куда записать итоговый JSON")
@click.option(
    "--config",
    default=None,
    metavar="YAML",
    help="yaml с дефолтами geometry/batch (параметры модели и кластеризации берутся из --checkpoint)",
)
@click.option(
    "--checkpoint",
    default=None,
    metavar="PT",
    help=f"чекпоинт модели (default: {DEFAULT_CHECKPOINT})",
)
@click.option(
    "--cache-dir",
    default=None,
    metavar="DIR",
    help="папка промежуточных результатов (default: <output>.cache)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="продолжить с последнего этапа из кэша, а не пересчитывать всё",
)
@click.option(
    "--output-csv",
    default=None,
    metavar="CSV",
    help="дополнительно записать submission.csv соревнования (format infer.py)",
)
@click.option("--device", default=None, help="cuda / cpu (default: cuda если есть)")
@click.option("--batch-size", type=int, default=None, help="батч DINOv2-эмбеддингов")
@click.option("--num-workers", type=int, default=None, help="воркеры DataLoader эмбеддингов")
@click.option("--cluster-distance", type=float, default=None, help="порог косинусного расстояния кластеризации")
@click.option("--min-cluster-size", type=int, default=None, help="кластеры меньше — выбросы")
@click.option("--linkage", default=None, help="связывание агломеративной кластеризации (default: из чекпоинта)")
@click.option("--neighbors", type=int, default=None, help="число ближайших картинок по эмбеддингу для матчинга")
@click.option("--max-keypoints", type=int, default=None, help="максимум ALIKED-ключей на картинку")
@click.option("--resize", type=int, default=None, help="длинная сторона при извлечении фич")
@click.option("--min-matches", type=int, default=None, help="минимум матчей в паре, чтобы её сохранить")
@click.option("--num-threads", type=int, default=None, help="потоки COLMAP-маппера (-1 — все)")
@click.option("--verbose", "-v", is_flag=True, help="DEBUG-логирование этапов")
@click.version_option(__version__, prog_name="reconstruct")
def reconstruct(**options: Any) -> int:
    """
    Сгруппировать изображения по сценам и восстановить позы камер; результат — JSON.

    Returns
    -------
    process exit code: 0 ok, 2 usage/cache error, 1 unexpected failure

    """
    _setup_logging(options["verbose"])
    cache_hint = options["cache_dir"] or f"{options['output']}.cache"
    try:
        cfg = _load_params(options)
        result = run(cfg, resume=options["resume"])
        write_json(result, Path(cfg["output"]))
        logger.info("result JSON: %s", cfg["output"])
        if options["output_csv"]:
            csv_path = Path(options["output_csv"])
            to_submission_csv(result).to_csv(csv_path, index=False)
            logger.info("submission CSV: %s", csv_path)
        return 0
    except (PipelineError, ContractError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 2
    except KeyboardInterrupt:
        logger.error("прервано пользователем; прогресс в кэше %s — продолжите с --resume", cache_hint)
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("unexpected failure")
        return 1


def main(argv: list[str] | None = None) -> int:
    """
    Entry point kept callable for tests: invoke the click command and map its exits.

    Click's usage errors (missing --input-dir/--output, bad types) exit 2, matching
    the argparse-era contract; the command itself returns 0/1/2/130.
    """
    try:
        exit_code = reconstruct.main(args=argv, prog_name="reconstruct", standalone_mode=False)
    except click.UsageError as usage_error:
        usage_error.show()
        return 2
    except click.exceptions.Exit as exit_exception:  # --help / --version
        return int(exit_exception.exit_code)
    return int(exit_code or 0)


def run_console() -> None:
    """Console-script wrapper: main() -> sys.exit."""
    sys.exit(main())


if __name__ == "__main__":
    run_console()
