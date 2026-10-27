# Протокол и контракты

Зачем: чтобы при замене бейзлайна (модель, лосс, сэмплер, гиперпараметры) результаты оставались сравнимыми, а пайплайн — рабочим.

## Где что лежит

| Файл                                              | Что в нём                                                                                      |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [protocol.yaml](protocol.yaml)                    | Версия протокола, версия данных (md5 из DVC), параметры сплита, главная метрика, пороги метрик |
| [val_split.json](val_split.json)                  | Замороженный список val-сцен и val-выбросов                                                    |
| [contracts.py](../src/sfm_rotations/contracts.py) | Конвенции, интерфейсы и валидаторы: единственный источник правды в коде                        |
| [tests/contracts/](../tests/contracts)            | Проверяют, что код, данные и протокол соответствуют друг другу                                 |

## Что заморожено

Меняется только вместе с новой версией протокола.

- **Данные** — версия задаётся md5 из `data/*.dvc`, они записаны в `protocol.yaml`.
- **Валидация** — сцены и выбросы из `val_split.json`.
- **Метрики** — код `metrics.py`, пороги и главная метрика `imc/score`. Ожидаемые значения зафиксированы в `test_metrics.py`.
- **Конвенции**:
  - выбросы: `scene == "outliers"`, `scene_label == -1`, позы NaN;
  - сцена — пара `(dataset, scene)`;
  - поза world → camera: `x_cam = R @ x_world + t`, центр камеры `c = -Rᵀt`;
  - в CSV `R` записана построчно, компоненты `R` и `t` разделены `;`.
- **Формат сабмита** — колонки `dataset,scene,image,rotation_matrix,translation_vector`, у выбросов поза `nan`.

## Что обязан соблюдать любой бейзлайн

- Модель отдаёт `Predictions`: `embeddings [N, D]`, `rotations [N, 3, 3]` (настоящие повороты), `translations [N, 3]`. Как поворот параметризуется внутри (6D, кватернионы) — её дело.
- Чекпоинт устроен как `Checkpoint`: `protocol_version`, `model_config` (обязательно `image_size`), `postprocess` (`cluster_distance`, `min_cluster_size`), `model`, `epoch`, `metrics`. Внутри только тензоры и встроенные типы Python: файл должен грузиться через `torch.load(weights_only=True)`, это поведение по умолчанию с torch 2.6.
- Метрики эпохи содержат все `REQUIRED_METRIC_KEYS`; дополнительные ключи можно.
- Инференс пишет `submission.csv`, который проходит `validate_submission`.

## Что можно менять свободно

Backbone, головы, лоссы и их веса, сэмплер и состав батча, аугментации, оптимизатор, гиперпараметры. Сюда же входят `cluster_distance` и `min_cluster_size`: в соревновании кластеризацию делает участник, так что это часть решения, а не протокола.

Условие одно: `tests/contracts` зелёные.

## Правила версий

Версию протокола поднимают, когда меняются данные, сплит, код или пороги метрик, главная метрика или конвенции. Запуски разных версий несравнимы. Изменение бейзлайна версию не трогает.

Порядок смены протокола:

1. Обновить данные или метрики.
1. Поднять `version` в `protocol.yaml` и `PROTOCOL_VERSION` в `contracts.py` (тест сверяет их).
1. Переписать md5 в `protocol.yaml` из `data/*.dvc`.
1. Сделать `dvc pull` и пересоздать сплит: `python -m sfm_rotations.contracts`.
1. Если менялись метрики — обновить ожидаемые значения в `test_metrics.py`.
1. Перепрогнать бейзлайн, чтобы у новой версии была точка отсчёта.
1. В PR написать, почему поменялся протокол.

## Тесты

```bash
uv run pytest tests/contracts
```

- Тесты на настоящих данных пропускаются, пока нет `data/train_labels.csv`; после `dvc pull` они прогоняются.
- В CI ([tests.yml](../.github/workflows/tests.yml)) тесты идут на каждый PR, без данных и без скачивания модели.
- Известные расхождения помечены `xfail` с номером из таблицы ниже. `xfail` строгий: когда расхождение исправлено, тест начинает проходить и роняет прогон. Тогда снимите маркер и уберите строку из таблицы.

## Известные расхождения

| #   | Расхождение                                                                                                                                                                                                                                                         | Где                                                                                                                                                    | Тест                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Инференс кластеризует все изображения разом, а оценка — внутри каждого датасета. Метрика на валидации считается не той процедурой, которой строится сабмит. Заодно infer импортирует приватный `_normalize` из `metrics.py`.                                        | [infer.py:122](../src/sfm_rotations/infer.py#L122), [metrics.py:509](../src/sfm_rotations/metrics.py#L509)                                             | `test_pipeline::test_infer_clusters_within_dataset`                                                                                                 |
| 2   | Инференс пишет `predictions.json` (вложенные списки, номера кластеров), а не `submission.csv`. У выбросов есть позы вместо `nan`.                                                                                                                                   | [infer.py:130](../src/sfm_rotations/infer.py#L130)                                                                                                     | `test_pipeline::test_infer_writes_submission`                                                                                                       |
| 3   | В чекпоинте только `model`, `epoch`, `metrics`: нет `protocol_version`, `model_config`, `postprocess`. Поэтому `backbone`, `embedding_dim`, `cluster_distance`, `min_cluster_size` продублированы в двух yaml, а `image_size=224` захардкожен в `build_transforms`. | [train.py:304](../src/sfm_rotations/train.py#L304), [infer.py:77](../src/sfm_rotations/infer.py#L77), [data.py:329](../src/sfm_rotations/data.py#L329) | `test_pipeline::test_train_checkpoint_follows_contract`; пока не исправлено, `test_infer_config_matches_train_config` следит, чтобы дубли совпадали |
| 4   | Без чекпоинта infer печатает warning и работает на случайных головах. Путь по умолчанию `checkpoints/best.pt` не совпадает с выходом train `checkpoints/baseline`, спасает только захардкоженный fallback.                                                          | [infer.py:83](../src/sfm_rotations/infer.py#L83), [baseline_infer.yaml:3](../configs/baseline_infer.yaml#L3)                                           | `test_pipeline::test_infer_fails_without_checkpoint`, `test_pipeline::test_infer_default_checkpoint_is_train_output`                                |
| 5   | `train.py` не знает о протоколе: считает сплит сам из `seed` и `val_fraction` конфига вместо `val_split.json` и не пишет `protocol_version` в MLflow и metrics json. Поменяли `val_fraction` или данные — получили другую валидацию.                                | [train.py:210](../src/sfm_rotations/train.py#L210)                                                                                                     | `test_protocol::test_val_split_matches_baseline_split` (нужны данные) следит, что сплиты пока совпадают                                             |
| 6   | У модели нет `predict() -> Predictions`: 6D-представление разворачивают сами `train.py` и `infer.py`.                                                                                                                                                               | [train.py:149](../src/sfm_rotations/train.py#L149), [infer.py:110](../src/sfm_rotations/infer.py#L110)                                                 | `test_model::test_model_returns_predictions`                                                                                                        |
| 7   | README расходится с конфигами: `epochs` 10 против 2, `checkpoints` против `checkpoints/baseline`, совет про `--cluster-distance` со времён argparse.                                                                                                                | [README.md:211](../README.md#L211), [README.md:215](../README.md#L215), [README.md:307](../README.md#L307)                                             | —                                                                                                                                                   |
| 8   | В MLflow имена метрик санитизируются (`emb/recall@1` → `emb/recall_at_1`), а в json остаются исходными. При сравнении запусков у метрики два имени.                                                                                                                 | [train.py:321](../src/sfm_rotations/train.py#L321)                                                                                                     | —                                                                                                                                                   |

## Открытые вопросы

- `image_id`: в [overview.md](../docs/overview.md) сказано, что колонка из `sample_submission.csv` обязана быть в сабмите, но в примере формата её нет. `SUBMISSION_COLUMNS` пока без неё; уточнить по `sample_submission.csv`.
- Эталонные метрики бейзлайна на протоколе v1 появятся после первого полного прогона.
