"""
`reconstruct` CLI: images folder(s) in, strict JSON out.

    reconstruct --input-dir data/group_01/ --output result.json
    reconstruct --input-dir a/ --input-dir b/ --output result.json --verbose
    reconstruct --input-dir data/group_01/ --output result.json --resume

Exit codes: 0 ok (per-cluster geometry failures are reported in meta.errors),
2 usage/cache error, 1 unexpected failure.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sfm_rotations import __version__
from sfm_rotations.contracts import ContractError
from sfm_rotations.pipeline import PipelineError
from sfm_rotations.pipeline import run
from sfm_rotations.pipeline import to_submission_csv
from sfm_rotations.pipeline import write_json

logger = logging.getLogger("reconstruct")

DEFAULT_CONFIG = Path("configs/baseline_infer.yaml")


def build_parser() -> argparse.ArgumentParser:
    """
    Argument parser of the reconstruct CLI.

    Returns
    -------
    parser

    """
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description="Сгруппировать изображения по сценам и восстановить позы камер; результат — JSON.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        metavar="DIR",
        help="папка с изображениями; подпапки = датасеты (как data/test), картинки в корне "
        "= один датасет с именем папки; можно передавать несколько раз",
    )
    parser.add_argument("--output", required=True, metavar="FILE", help="куда записать итоговый JSON")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        metavar="YAML",
        help=f"yaml с параметрами по умолчанию (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PT",
        help="чекпоинт модели (default: checkpoint из --config)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="папка промежуточных результатов (default: <output>.cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="продолжить с последнего этапа из кэша, а не пересчитывать всё",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        metavar="CSV",
        help="дополнительно записать submission.csv соревнования (format infer.py)",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda если есть)")
    parser.add_argument("--batch-size", type=int, default=None, help="батч DINOv2-эмбеддингов")
    parser.add_argument("--num-workers", type=int, default=None, help="воркеры DataLoader эмбеддингов")
    parser.add_argument(
        "--cluster-distance",
        type=float,
        default=None,
        help="порог косинусного расстояния кластеризации",
    )
    parser.add_argument("--min-cluster-size", type=int, default=None, help="кластеры меньше — выбросы")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=None,
        help="число ближайших картинок по эмбеддингу для матчинга",
    )
    parser.add_argument("--max-keypoints", type=int, default=None, help="максимум ALIKED-ключей на картинку")
    parser.add_argument("--resize", type=int, default=None, help="длинная сторона при извлечении фич")
    parser.add_argument("--min-matches", type=int, default=None, help="минимум матчей в паре, чтобы её сохранить")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="потоки COLMAP-маппера (-1 — все)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-логирование этапов")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _load_params(args: argparse.Namespace) -> dict[str, Any]:
    """
    Merge defaults from --config with the explicit CLI flags into the pipeline cfg.

    Parameters
    ----------
    args:
        parsed CLI arguments.

    Returns
    -------
    cfg dict for pipeline.run

    """
    config_path = Path(args.config)
    defaults = OmegaConf.to_container(OmegaConf.load(config_path)) if config_path.is_file() else {}
    if not isinstance(defaults, dict):
        raise PipelineError(f"--config {config_path}: ожидался словарь")
    geometry = defaults.get("geometry", {})

    overrides = {
        key: getattr(args, key)
        for key in (
            "device",
            "batch_size",
            "num_workers",
            "cluster_distance",
            "min_cluster_size",
            "checkpoint",
        )
        if getattr(args, key) is not None
    }
    geometry_overrides = {
        key: getattr(args, key)
        for key in ("neighbors", "max_keypoints", "resize", "min_matches", "num_threads")
        if getattr(args, key) is not None
    }

    output = Path(args.output)
    cfg: dict[str, Any] = {
        "input_dirs": args.input_dir,
        "output": str(output),
        "cache_dir": args.cache_dir or str(output.parent / (output.name + ".cache")),
        "device": args.device,
        "checkpoint": overrides.pop("checkpoint", defaults.get("checkpoint", "checkpoints/baseline/best.pt")),
        "backbone": defaults.get("backbone", "facebook/dinov2-base"),
        "embedding_dim": defaults.get("embedding_dim", 256),
        "batch_size": overrides.pop("batch_size", defaults.get("batch_size", 32)),
        "num_workers": overrides.pop("num_workers", defaults.get("num_workers", 0)),
        "cluster_distance": overrides.pop("cluster_distance", defaults.get("cluster_distance", 0.3)),
        "min_cluster_size": overrides.pop("min_cluster_size", defaults.get("min_cluster_size", 3)),
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


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.

    Returns
    -------
    process exit code: 0 ok, 2 usage/cache error, 1 unexpected failure

    """
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cache_hint = args.cache_dir or f"{args.output}.cache"
    try:
        cfg = _load_params(args)
        result = run(cfg, resume=args.resume)
        write_json(result, Path(cfg["output"]))
        logger.info("result JSON: %s", cfg["output"])
        if args.output_csv:
            csv_path = Path(args.output_csv)
            to_submission_csv(result).to_csv(csv_path, index=False)
            logger.info("submission CSV: %s", csv_path)
        return 0
    except (PipelineError, ContractError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 2
    except KeyboardInterrupt:
        logger.error("прервано пользователем; прогресс в кэше %s — продолжите с --resume", cache_hint)
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("unexpected failure")
        return 1


def run_console() -> None:
    """Console-script wrapper: main() -> sys.exit."""
    sys.exit(main())


if __name__ == "__main__":
    run_console()
