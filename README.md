# SFM-Rotations

Structure from Motion Rotations

## Правила бойцовского клуба

## Я серьезно

#### Пользуемся uv и радуемся жизни

- Установка uv:

```
wget -qO- https://astral.sh/uv/install.sh | sh
Перезапустить терминал (или ввести одну из команд, которые предложит установщик)
uv venv --python 3.10
source .venv/bin/activate
```

- Далее, пакет ставим так (из root):

```
uv sync --all-groups
pre-commit install
```

- Отдельные пакеты ставим так:

```
uv pip install <пакет>
```

#### В мейн не пушим (разве что initial commit)

- Делаем свою ветку:

```
от текущей активной
git checkout -b <ветка>
```

- Пушим в remote:

```
git push origin <ветка, которую хотите запушить>
```

#### Гоняем pre-commit

- Использование (из root):

```
pre-commit run -a
```
