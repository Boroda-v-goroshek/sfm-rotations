# SFM-Rotations

Structure from Motion Rotations

#### Использование uv

- Установка uv:

```
wget -qO- https://astral.sh/uv/install.sh | sh
Перезапустить терминал (или ввести одну из команд, которые предложит установщик)
uv venv --python 3.10
source .venv/bin/activate
```

- Далее, пакет ставим так (из корня репозитория):

```
uv sync --all-groups
pre-commit install
```

- Отдельные пакеты ставим так:

```
uv pip install <пакет>
```

#### В мейн не пушим

- Делаем свою ветку:

```
от текущей активной
git checkout -b <ветка>
```

- Пушим в remote:

```
git push origin <ветка, которую хотите запушить>
```

#### Запуск pre-commit

- Использование (из корня репоса):

```
pre-commit run -a
```

## Работа с DVC

### Первый запуск

```bash
# генерируете ключ и кидаете публичный в группу. админы добавят на сервер, и вы сможете пользоваться DVC
ssh-keygen -t rsa -b 4096 -f ~/.ssh/mlops
```

Сурсите окружение.
Настраиваете свой личный доступ. Эти настройки сохранятся в файле .dvc/config.local, который Git игнорирует.

```bash
# добавляете ключ в DVC
dvc remote modify --local sfm-dvc keyfile ~/.ssh/mlops
```

### Основы работы

В dvc храним данные для тренировки и модели:

Для хранения данных:

```
data/
- train/
  - набор картинок 1
  - набор картинок 2
  - ...
- test/
  - набор картинок 1
  - набор картинок 2
  - ...
- train_labels.csv (Список изображений в обучающих данных, включая эталонные данные)
- train_thresholds.csv (Список пороговых значений для оценки)
```

Для хранения моделей:

```
checkpoints/
- baseline/
  - набор файлов
- model_name/
  - набор файлов
```

### Пуш в dvc

1. Папки с моделями добавляем ПО ОТДЕЛЬНОСТИ и коммитим ТОЖЕ ОТДЕЛЬНО. То есть

```
dvc add checkpoints/baseline/
git add checkpoints/baseline.dvc
git commit -m "New weights for baseline. Changed lr"

dvc add checkpoints/baseline/
git add checkpoints/baseline.dvc
git commit -m "New weights for baseline. Changed batch size"

и т.д.
```

В противном случае(например, если добавить dvc add models) DVC видит всю папку как один артефакт и изменение весов для одного объекта повлечет за собой изменение вообще всей папки models, а это неудобно для версионирования

После того как всё добавили, делаете

```
dvc push
```

Затем проверяете что все .dvc закоммичены

И!! ОБЯЗАТЕЛЬНО

```
git push origin YOUR_BRANCH_NAME
```

И сливаете через пулл реквест с мейном, чтобы все могли пользоваться обновленными данными в DVC

## Работа с MlFlow

Все эксперименты, гиперпараметры, графики и метрики логируются на удалённый MLflow-сервер.
Для авторизации используется единый стандарт конфигурации через файл `.env` в корне проекта.

Файл `.env` хранится только локально на вашей машине.

### Настройка доступа (единоразово)

1. Создайте локальный файл `.env` из шаблона `.env.example`:

- **Linux / macOS**:
  ```bash
  cp .env.example .env
  ```
- **Windows (PowerShell)**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Windows (CMD)**:
  ```cmd
  copy .env.example .env
  ```

2. Откройте созданный `.env` и укажите выданные вам логин и пароль:

```env
MLFLOW_TRACKING_URI=https://217.60.0.116.sslip.io
MLFLOW_TRACKING_USERNAME=ваш_логин
MLFLOW_TRACKING_PASSWORD=ваш_пароль
```

3. (Опционально) Проверьте подключение СО СВОИМ ИМЕНЕМ В ПРОЕКТЕ!!!!!!:

```bash
python -c "from dotenv import load_dotenv; import mlflow; load_dotenv(); mlflow.set_experiment('test_sfm-rotations_YOURNAME'); print('MLflow OK')"
```

При запуске обучения скрипт автоматически считает `.env`, авторизуется и запишет:

- Использованные гиперпараметры;
- Версию кода (`git_commit_id`);
- Метрики и функции потерь по эпохам (интерактивные графики в веб-интерфейсе);
- Графики и лог обучения (в директорию `plots/` репозитория и в артефакты MLflow).

## Обучение

Конфигурация обучения управляется с помощью **[Hydra](https://hydra.cc/)**.
Все параметры по умолчанию заданы в файле [configs/train/main.yaml](configs/train/main.yaml).

#### Запуск обучения

```bash
# 1. Запуск с параметрами по умолчанию из configs/train/main.yaml:
python -m sfm_rotations.train

# 2. Запуск БЕЗ логирования в MLflow (для тестов и отладки пайплайна):
python -m sfm_rotations.train --no-logging

# 3. Переопределение любых параметров через аргументы командной строки:
python -m sfm_rotations.train epochs=5 lr=1e-4 batch_size=21

# 4. Переопределение списков и путей:
python -m sfm_rotations.train data_root=data checkpoints=checkpoints/exp1 "loss_weights=[1.0,0.0,0.0]"

# 5. Просмотр полной справки и доступных параметров:
python -m sfm_rotations.train --help
```

## Инференс

Модель отвечает только за группировку картинок, а позы считает геометрия ([geometry.py](src/sfm_rotations/geometry.py)):

1. DINOv2-эмбеддинги кластеризуются внутри каждого датасета / сцены.
1. Внутри кластера пары картинок отбираются по сходству эмбеддингов (`geometry.neighbors`).
1. На парах работают ALIKED и LightGlue, матчи проверяются геометрически.
1. Инкрементальный маппер COLMAP восстанавливает позы. Незарегистрированные картинки остаются в кластере с позой `nan`.

При запуске автоматически подтягиваются тестовые данные через **DVC API** (файл `data/test.dvc`).

#### Запуск инференса:

```bash
# 1. Запуск с параметрами по умолчанию (configs/inference/fast.yaml на основе main.yaml):
python -m sfm_rotations.infer

# 2. Запуск в режиме максимального качества (configs/inference/quality.yaml):
python -m sfm_rotations.infer --config-name=quality

# 3. Запуск на отдельной сцене (например, только stairs):
python -m sfm_rotations.infer test_dir=data/test/stairs

# 4. Запуск на нескольких сценах одновременно (общая папка data/test):
python -m sfm_rotations.infer test_dir=data/test

# 5. Запуск с автоматическим скачиванием данных через DVC (--dvc-pull):
python -m sfm_rotations.infer --dvc-pull

# 6. Указание своего чекпоинта и пути сохранения:
python -m sfm_rotations.infer checkpoint=checkpoints/dinov2-large/best.pt output_path=predictions.csv
```

#### Раздельный инференс и сохранение результатов

Скрипт инференса автоматически определяет структуру входной директории `test_dir`:

- **Поддержка одиночной сцены и набора сцен:**

  - Если передана папка с поддиректориями (например, `data/test`, где лежат `ETs/` и `stairs/`), инференс выполняется раздельно для каждой сцены: эмбеддинги, кластеризация и 3D-реконструкция COLMAP работают независимо по очереди, исключая смешивание разных сцен и снижая пиковое потребление памяти.
  - Если передана конкретная папка со снимками сцены (например, `data/test/stairs`), обрабатывается только она.

- **Именование и структура выходных файлов:**

  - Результаты автоматически сохраняются в папку с временной меткой: `results/YYYY-MM-DD_HH-MM-SS/` (настраивается в конфиге Hydra через `output_path: results/${now:%Y-%m-%d_%H-%M-%S}/...`).
  - **Для каждой сцены создаётся свой отдельный CSV-файл** с суффиксом `_<scene_name>`: например, `dinov2-large_ETs.csv` и `dinov2-large_stairs.csv` (общий объединённый файл не создаётся).

Каждый CSV-файл содержит колонки:

- `dataset`: название сцены / датасета (`ETs`, `stairs`);
- `scene`: предсказанный кластер сцены (`cluster0`, `cluster1` или `outliers`);
- `image`: относительное имя файла изображения;
- `rotation_matrix`: матрица поворота камеры (3x3, записанная через `;` в row-major порядке);
- `translation_vector`: вектор перемещения (3D, через `;`).

> **Что означает `nan` в позах:**
> Если для конкретного кадра COLMAP не смог надёжно восстановить 3D-позу (из-за смаза, недостатка текстур или экстремального ракурса), в `rotation_matrix` и `translation_vector` записываются значения `nan`. Это официальный стандарт Image Matching Challenge: кадр остаётся привязанным к своему кластеру, а отсутствие позы сигнализирует проверяющей системе о незарегистрированной камере.

## Метрики

Главной целевой метрикой соревнования Image Matching Challenge является **`imc/score`** — гармоническое среднее (аналог F1-меры), объединяющее качество группировки кадров и точность восстановления 3D-поз:

> **Score = 2 · (mAA · Clustering) / (mAA + Clustering)**

- **`imc/clustering`** (аналог Precision) — качество кластеризации: доля корректно отнесённых к своим сценам изображений и точность фильтрации посторонних кадров (выбросов).
- **`imc/mAA`** (*mean Average Accuracy*, аналог Recall) — геометрическая точность: средняя доля успешно зарегистрированных камер, чьи восстановленные пространственные позы укладываются в заданные пороги погрешности.

Именно по метрике `imc/score` на валидационном сплите автоматически выбирается лучший чекпоинт (`best.pt`).

#### Запуск валидации (Evaluate)

```bash
# 1. Быстрая оценка (configs/evaluate/fast.yaml на основе main.yaml):
python -m sfm_rotations.evaluate

# 2. Оценка в режиме максимального качества (configs/evaluate/quality.yaml):
python -m sfm_rotations.evaluate --config-name=quality
```
