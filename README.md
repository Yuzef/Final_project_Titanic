1) EDA производился на основе ipynb https://www.kaggle.com/code/ash316/eda-to-prediction-dietanic
   (я не скачивал ноутбук, копировал вставлял поштучно, разбирал, что было непонятно,
   исправлял устаревший код).

2) Препроцессинг сделан в виде "конфигурируемых гипотез".

2) Наверное тестировать варианта препроцессинга есть смысл только на baseline модели,
   далее выбрать лучший вариант препроцессинга и использовать только его.


Структура проекта:
Final_project_Titanic/
├── configs/                  — все настройки эксперимента.
│   └── config.py 
├── utils/
│   ├── validation.py        — функции разбиения данных.
│   ├── preprocessing.py     — функции обработки данных.
│   └──
├── data_raw/                - сырые данные
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│   
├── outputs/
│   ├── 
│   ├── 
│   ├── 
│   └── 
├── main.py                  — точка входа, запускается одной кнопкой.
├── 
├── 
├── EDA.ipynb
├── environment.yml         - all dependencies. файл окружения для conda.
                              conda env create -f environment.yml
├── requirements.txt        - список Python-пакетов для pip.
                              pip install -r requirements.txt
└── README.md


Архитектура:

config.py
  ↓
main.py
  ↓
загрузка данных
  ↓
validation.py разбиение данных
  ↓
preprocessing.py
  ↓
выбор модели
  ↓
обучение / предсказание / сохранение результата




