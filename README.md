1) EDA производился на основе ipynb https://www.kaggle.com/code/ash316/eda-to-prediction-dietanic
   (я не скачивал ноутбук, копировал вставлял поштучно, разбирал, что было непонятно,
   исправлял устаревший код).

Для установки необходимого окружения:
conda env create -f environment.yml
conda activate titanic_ml

Запуск:
python main.py

Если запускается эксперимент повторно, то надо зайти в папку с его именем
и удалить _BEST.joblib файл, иначе будет ошибка, так как ожидается,
что _BEST файл будет только один.

## Pipeline

Проект запускается из `main.py` и управляется через `configs/config.py`.

Основной процесс:

1. Загружается `train.csv`.
2. Данные разбиваются на validation folds.
3. Для каждого fold preprocessing обучается только на train-части и применяется к train/validation.
4. Все `enabled` модели из config обучаются и оцениваются на folds.
5. Для каждого эксперимента модель заново обучается на всём `train.csv` и сохраняется как `.joblib` artifact.
6. Лучшая модель выбирается по максимальному mean score и минимальному std.
7. Если `inference.enabled=True`, создаётся submission для `test.csv`.

Artifact модели содержит не только модель, но и `preprocessing_state`, поэтому inference не делает `fit`, а только применяет сохранённые правила обработки данных и вызывает `predict`.

Результаты эксперимента сохраняются в:
trained_models/{experiment_name}/

Внутри папки эксперимента:
*.joblib          # обученные модели
*_BEST.joblib     # копия лучшей модели
fold_results.csv  # score каждой модели на каждом fold
summary.csv       # mean/std по моделям
best_model.csv    # выбранная лучшая модель
config.yaml       # snapshot config
submissions/      # submission-файлы

Важные настройки:
general.experiment_name   # имя папки эксперимента
modeling.models[].enabled # какие модели запускать
metric.name               # метрика выбора модели
inference.enabled         # создавать ли submission
inference.use_best_model  # использовать ли *_BEST.joblib

Структура проекта:
Final_project_Titanic/
├── configs/                         — все настройки эксперимента.
│   └── config.py                    — главный config: paths, preprocessing, models, logging, inference.
│
├── utils/
│   ├── train_validation_splitting.py — функции разбиения данных на folds.
│   ├── preprocessing.py             — fit/transform preprocessing и feature engineering.
│   ├── modeling.py                  — обучение моделей, CV, выбор лучшей модели, сохранение artifacts.
│   ├── inference.py                 — загрузка сохранённой модели и создание submission.
│   └── experiment_logging.py        — сохранение логов эксперимента: config, metrics, summary, best model.
│
├── data_raw/                        — сырые данные.
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── trained_models/                  — результаты экспериментов.
│   └── {experiment_name}/
│       ├── *.joblib                 — обученные модели, готовые к inference.
│       ├── *_BEST.joblib            — копия лучшей модели эксперимента.
│       ├── fold_results.csv         — метрика каждой модели на каждом fold.
│       ├── summary.csv              — mean/std score по моделям.
│       ├── best_model.csv           — информация о лучшей модели.
│       ├── artifacts.csv            — пути к сохранённым model artifacts.
│       ├── config.yaml              — snapshot config для воспроизводимости.
│       └── submissions/             — submission-файлы для Kaggle.
│
├── main.py                          — точка входа, запускается одной кнопкой.
├── EDA.ipynb                        — exploratory data analysis.
├── environment.yml                  — файл окружения conda.
│                                      conda env create -f environment.yml
├── requirements.txt                 — список Python-пакетов для pip.
│                                      pip install -r requirements.txt
└── README.md



Архитектура:

configs/config.py
  ↓
main.py
  ↓
загрузка train.csv
  ↓
train_validation_splitting.py
  разбиение данных на folds
  ↓
preprocessing.py
  fit preprocessing только на train-части fold
  transform train/validation
  ↓
modeling.py
  обучение всех enabled моделей
  оценка на folds
  выбор лучшей модели по mean/std score
  обучение enabled моделей на всём train.csv
  сохранение .joblib artifacts
  ↓
experiment_logging.py
  сохранение config.yaml
  сохранение fold_results.csv
  сохранение summary.csv
  сохранение best_model.csv
  ↓
inference.py
  если inference.enabled=True:
    загрузка *_BEST.joblib или выбранной модели
    transform test.csv через сохранённый preprocessing_state
    predict
    сохранение submission в папку эксперимента




