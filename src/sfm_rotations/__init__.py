"""
sfm_rotations: scene clustering and camera pose reconstruction (IMC 2025).

OMP-совместимость macOS/arm64: torch и pycolmap статически линкуют каждый свой
libomp, вторая инициализация валит процесс (OMP Error #15). KMP_DUPLICATE_LIB_OK
читается libomp при загрузке второй копии, поэтому переменную нужно выставить до
импорта torch/pycolmap; __init__ пакета выполняется раньше любого submodule-импорта,
это единственное место, где такой shim надёжен для всех entry-точек (CLI, hydra-
скрипты, тесты).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.0.1"
