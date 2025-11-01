# Автоматизация аннотации генома

Базовый каркас проекта для исследований по автоматизации аннотации генома (Python + PyTorch + Hydra).

## Создание окружения

```bash
mamba env create -f genome-lab-cuda.yml # из собранного конфига
conda activate genome-lab
```

## Подключение в Jupyter / PyCharm

python -m ipykernel install --user --name genome-lab --display-name "Python (genome-lab)"

## Чистка кэша
```bash
mamba clean --all -y && rm -rf ~/.cache/pip
```

## Log
```bash
tensorboard --logdir lightning_logs
```

## Быстрый старт (WSL2 + conda/mamba)
```bash
conda activate genome-lab  # или ваше окружение
pip install -e .           # подключить проект в editable-режиме
pre-commit install         # (опционально) хуки
```

## Запуск проверки окружения
```bash
python -m genome_anno.utils.env_check
```

## Обучение (пример)
```bash
python -m genome_anno.dl.train trainer.max_epochs=1 data.window_size=1024
```
