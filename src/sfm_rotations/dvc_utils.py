"""DVC integration utilities using DVC Python API."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def pull_dvc_data(
    targets: list[str] | str | None = None,
    repo_root: str | Path = ".",
    required_paths: list[Path] | None = None,
) -> None:
    """
    Download dataset files via DVC Python API (Repo.pull).

    Parameters
    ----------
    targets:
        Target .dvc files or paths to pull. If None, pulls all DVC targets.
    repo_root:
        Path to the repository root containing .dvc/.
    required_paths:
        Optional list of local files/folders that must exist. If DVC pull fails
        and any of these paths is missing, an error is raised.

    """
    print(f"Checking and pulling data via DVC Python API (targets={targets})...")
    try:
        from dvc.repo import Repo

        repo = Repo(str(repo_root))
        if isinstance(targets, str):
            targets = [targets]
        result = repo.pull(targets=targets)
        stats = result.get("stats", result) if isinstance(result, dict) else result
        print(f"DVC pull completed successfully: {stats}")
    except Exception as e:
        print(f"Warning: DVC pull encountered an issue: {e}")
        if required_paths:
            missing = [p for p in required_paths if not p.exists()]
            if missing:
                raise RuntimeError(
                    f"Required data is missing ({missing}) and DVC pull failed: {e}. "
                    "Make sure your DVC remote and credentials are configured.",
                ) from e
            print("Existing local data found. Continuing with existing data.")
