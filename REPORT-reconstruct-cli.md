# Отчёт: CLI `reconstruct` в sfm-rotations

Ветка: `feat/reconstruct-cli` (1 коммит `3193540`, main не тронут — откатан до origin/main)

______________________________________________________________________

## 1. Что за репозиторий (разбор)

**sfm-rotations** — решение Kaggle *Image Matching Challenge 2025*. Задача: по «свалке» фотографий
разных достопримечательностей сгруппировать их по сценам и для каждой группы восстановить позы камер.

### Пайплайн (вход → выход)

1. **Вход:** `data/test/<dataset>/*.png` (layout соревнования: датасеты → папки, внутри картинки;
   сцены в разметке test неизвестны — их надо найти).
1. **Эмбеддинги** (`infer.py` + `model.py`): DINOv2-base с SupCon-head (чекпоинт
   `checkpoints/baseline/best.pt`, DVC ~700 МБ) → L2-нормированный вектор [N, 256] на картинку.
   Модель дообучена в `train.py` (supervised contrastive loss, батч-сэмплер из `data.py`
   собирает батчи из 7 сцен + outliers).
1. **Кластеризация** (`metrics.cluster_embeddings`): agglomerative clustering, cosine distance,
   порог `cluster_distance=0.3`; кластеры меньше `min_cluster_size=3` → `outliers` (-1).
   Кластеризация строго **внутри датасета**.
1. **Позы** (`geometry.py`): для каждого кластера: shortlist пар поcosine-сходству эмбеддингов
   (top-`neighbors`=20) → ALIKED keypoint'ы (max 2048, resize 1024) → LightGlue матчи
   (мин. 15, геоверификация) → инкрементальный COLMAP-mapper (`pycolmap`) → для каждой картинки
   R (3×3) и t (3); незарегистрированные mapper’ом остаются в кластере с NaN.
1. **Выход:** `submission.csv` — `dataset,scene,image,rotation_matrix` (9 чисел `;`-split),
   `translation_vector` (3). У outliers — nan'ы.
1. **Метрики** (`evaluate.py`, `metrics.py`): imc/score (основная по протоколу), rot mAA, ARI —
   на замороженном val-сплите (`contracts/val_split.json`, протокол v2).
1. **Контракты** (`contracts.py` + `tests/contracts/`): жёсткие проверки формата CSV, метрик,
   протокола; тесты гоняются на tiny-синтетике без весов/GPU.

### Инфраструктура репо

- `uv` для зависимостей (Python 3.10), hydra-конфиги `configs/baseline_{train,infer}.yaml`.
- DVC: данные и веса (`dvc pull`, remote ssh://aboba@217.60.0.116).
- pre-commit: ruff check+format, pyright, yamllint, commitizen (conventional commits).

______________________________________________________________________

## 2. Что сделано: CLI `reconstruct`

### Интерфейс

```bash
reconstruct --input-dir data/group_01/ --output result.json --num-threads 1 [-v]
```

| Флаг                                                       | Значение                                                                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--input-dir DIR` (×N)                                     | папка с фото; **подпапки = датасеты** (layout data/test), картинки в корне = один датасет с именем папки; можно несколько |
| `--output FILE`                                            | итоговый JSON (обязательный)                                                                                              |
| `--resume`                                                 | продолжить с последнего сохранённого этапа из кэша                                                                        |
| `--cache-dir DIR`                                          | куда класть промежуточные результаты (default `<output>.cache`)                                                           |
| `--output-csv FILE`                                        | дополнительно submission.csv в формате соревнования                                                                       |
| `--config YAML`                                            | yaml с параметрами (default `configs/baseline_infer.yaml`)                                                                |
| `--checkpoint PT`                                          | чекпоинт модели                                                                                                           |
| `--device / --batch-size / --num-workers`                  | исполнение эмбеддингов                                                                                                    |
| `--cluster-distance / --min-cluster-size`                  | кластеризация                                                                                                             |
| `--neighbors / --max-keypoints / --resize / --min-matches` | матчинг/фичи                                                                                                              |
| `--num-threads N`                                          | потоки COLMAP-маппера (на mac: 1)                                                                                         |
| `-v/--verbose`                                             | DEBUG-лог; `--version`                                                                                                    |

Коды выхода: `0` ок, `2` ошибка использования/кэша, `1` непредвиденная, `130` Ctrl+C
(в сообщении путь к кэшу и подсказка про `--resume`).

### Этапы и кэш («поэтапное сохранение» из ТЗ)

```
discover → embeddings → clusters → poses (по кластерам) → result JSON
```

в `--cache-dir`:

- `fingerprint.json` — sha256 от (версия, список картинок относительно input-dir, backbone,
  embedding_dim, cluster_distance, min_cluster_size, geometry.\* кроме num_threads);
- `embeddings.npz` — [N,256];
- `clusters.json` — id кластеров;
- `cluster-<dataset>__<k>.json` — позы одного кластера (с fingerprint + списком картинок кластера).

Поведение:

- `--resume`: fingerprint совпал → готовые этапы читаются с диска (проверено: повторный прогон
  73 фото — **1.7 сек** против ~10 мин). Fingerprint не совпал → **ошибка** (resume честно
  отказывается продолжать про другие данные).
- без `--resume` (fresh run) устаревший кэш **сбрасывается** и всё считается заново.
- `num_threads`, `device`, `batch_size` не влияют на fingerprint — resume после смены CPU↔GPU работает.
- битый кластер (исключение геометрии) не роняет прогон: поз ы=NaN, ошибка в `meta.errors`,
  остальные кластеры считаются.

### Формат результата (`schema_version: 1`, `contracts.RESULT_SCHEMA`)

```json
{
  "schema_version": 1,
  "pipeline_version": "0.0.1",
  "meta": {"input_dirs": [...], "device": "cpu", "checkpoint": "...",
           "params": {...}, "cache_dir": "...", "cache_fingerprint": "...", "errors": []},
  "datasets": [{
      "name": "ETs", "num_images": 22, "num_clusters": 1,
      "num_outliers": 0, "num_registered": 22,
      "images": [{
          "image": "ETs/another_et_another_et001.png",
          "scene": "cluster0",
          "rotation_matrix": [[r11,r12,r13],[r21,...],[r31,...]],
          "translation_vector": [x, y, z]
      }]
  }]
}
```

Семантика повторяет submission.csv: у outliers поза `null`; у зарегистрированных — ортонормированная
R (det=+1, проверяется `is_rotation`); у незарегистрированных `null`. Перед записью JSON проходит
`validate_result()`, битый файл физически не может появиться (пишется tmp+atomic replace,
`allow_nan=False`). `to_submission_csv()` даёт CSV, который проходит существующий
`validate_submission` — тест это проверяет.

### Файлы

- `src/sfm_rotations/pipeline.py` (~700 стр.) — ядро: discover, fingerprint/кэш, build_result,
  validate_result, to_submission_csv, run() по этапам.
- `src/sfm_rotations/cli.py` — argparse, слияние `--config` + флагов, логирование, коды выхода;
  entry point `reconstruct` в pyproject (`[project.scripts]`).
- `contracts.py` — `RESULT_SCHEMA = 1`; `data.py` — `IMAGE_EXTENSIONS` (общий для infer и CLI);
  `__init__.py` — `__version__`.
- `tests/test_reconstruct_cli.py` — 8 тестов (см. §4).

______________________________________________________________________

## 3. Найденные и исправленные реальные баги (попали в коммит)

1. **torch 2.14 не грузил baseline-чекпоинт.** С torch≥2.6 `torch.load` по умолчанию
   `weights_only=True`, а в `best.pt` в `metrics` лежат numpy-скаляры — `UnpicklingError`.
   Фикс: `weights_only=False` в `infer.load_model` (чекпоинт создаётся этим же train.py
   и лежит в доверенном DVC-remote).
1. **Abort на старте CLI (macOS/arm64).** torch и pycolmap статически линкуют каждый свой libomp;
   при совместном импорте — `OMP: Error #15` и SIGABRT. Фикс: `KMP_DUPLICATE_LIB_OK=TRUE`
   до импорта pycolmap в `geometry.py`.
1. **pycolmap mapper SIGSEGV в многопоточном режиме** (mapper `num_threads=-1`) на большом
   кластере stairs (51 фото, arm64 build). С `--num-threads 1` всё считается чисто.
   Лечится на стороне запуска; `num_threads` поэтому исключён из fingerprint кэша.
   **Рекомендация для mac: всегда `--num-threads 1`.**
1. **DVC не принимал `checkpoints/baseline.dvc` как цель pull** — файл попадал под
   `.gitignore` (`checkpoints/`). Фикс: `checkpoints/*` + `!checkpoints/*.dvc`.

Локальные настройки (не в гите): `dvc remote modify sfm-dvc keyfile ~/.ssh/ml_server --local`
и `dvc config --local cache.dir ./.dvc/cache` — лежат в `.dvc/config.local`, DVC и git его игнорируют.
(Первоначально cache.dir попал в общий `.dvc/config` — вернула как было.)

______________________________________________________________________

## 4. Проверки

- `pytest tests/` → **66 passed, 4 skipped, 2 xfailed** (все прежние контрактные тесты зелёные).
  8 новых тестов: discover layout'ов; e2e CLI (2 датасета×5 фото, tiny-чекпоинт + фейковый
  реконструктор, проверка JSON/кэша/CSV через `validate_submission`); resume ничего не
  пересчитывает (compute/load_model/geometry заменены на «сбой»); инвалидация кэша
  (resume при других params = exit 2; fresh run чистит кэш); падение кластера → exit 0,
  NaN-позы + `meta.errors`; битые входы (нет папки / пусто / нет чекпоинта) = exit 2.
- `ruff check` + `ruff format --check` чисто (pre-commit конфиг не запускался: sandbox
  не даёт `~/.cache/pre-commit`).
- **Реальный прогон** (не тестовый): `data/test` (ETs 22 + stairs 51 фото), CPU:
  embeddings ~2 мин, кластеры мгновенно (всё в 1 кластер — корректно, внутри датасета одна сцена),
  позы ~8 мин: **ETs 22/22 registered, stairs 9/51** (42 фото mapper не зарегистрировал —
  это ожидаемое поведение, в JSON они `null`). `result.json` прошёл `validate_result`,
  экспорт CSV прошёл `validate_submission`. `--resume` — 1.7 сек.
- CLI-разведка: `reconstruct --version`, `--help`, коды выхода на битых входах.

______________________________________________________________________

## 5. Очистка и коммит

Удалено: `/tmp/rc-demo/`, `result.json`, `result.json.cache/`, демо-папка `data/group_01/`.
Осталось на диске (это и должно быть, git/DVC-игнорируемое): `.venv`, скачанные
`checkpoints/baseline/`, `data/test`, `data/train*`, `.dvc/cache`, `data/.gitignore`-покрытые.

Коммит `3193540` в **`feat/reconstruct-cli`** (main откатан к origin/main):
10 файлов, +1276/−4:

```
M  .gitignore                     # checkpoints/* + !*.dvc
M  pyproject.toml                 # [project.scripts] reconstruct
M  src/sfm_rotations/__init__.py  # __version__
M  src/sfm_rotations/contracts.py # RESULT_SCHEMA
M  src/sfm_rotations/data.py      # IMAGE_EXTENSIONS
M  src/sfm_rotations/geometry.py  # KMP_DUPLICATE_LIB_OK до pycolmap
M  src/sfm_rotations/infer.py     # weights_only=False, IMAGE_EXTENSIONS
A  src/sfm_rotations/cli.py
A  src/sfm_rotations/pipeline.py
A  tests/test_reconstruct_cli.py
```

## 6. Что дальше (не делала)

- Merge-решение за вами: из ветки — PR/MR.
- README/docs секцию про CLI не писала (скажите — добавлю).
- Оптимизации скорости: `--device cuda`, больше `--neighbors`, батчи эмбеддингов — по железу.
- Опционально: `dvc.yaml`-стейджи вместо ручного pull, `--log-file`, JSON-schema как файл-контракт
  для внешних систем.
