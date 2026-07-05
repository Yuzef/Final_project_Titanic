import pandas as pd

def fill_embarked(df, cfg):
    """
    Заполняет пропуски в столбце Embarked наиболее часто встречающимся значением.
    В текущем датасете два пропуска заполняются значением S (порт посадки Southampton).

    Подробное исследование признака:
        EDA.ipynb.

    Args:
        df: Исходный DataFrame.
        cfg_embarked: Настройки предобработки столбца Embarked из конфигурации.

    Returns:
        DataFrame: DataFrame с заполненными пропусками в столбце Embarked.
    """
    df = df.copy()
    # Если в config будет enabled = False, то не будет использовать эту ф-цию в препроцессинге.
    if not cfg.enabled:
        return df
    
    if cfg.strategy == "most_frequent":
        most_frequent = df["Embarked"].mode().iloc[0]
        df["Embarked"] = df["Embarked"].fillna(most_frequent)
    else:
        raise ValueError(f"Unknown Embarked strategy: {cfg.strategy}")
    
    return df

def create_initial_column(df):
    """
    Создаёт столбец Initial на основе обращений в именах пассажиров.
    Извлекает обращения из столбца Name (например, Mr, Miss, Mrs) и объединяет редкие обращения
    в категорию Other.

    Используется при заполнении пропусков возраста по группам обращений.

    Args:
        df: Исходный DataFrame со столбцом Name.
    Returns:
        DataFrame: Копия исходного DataFrame с добавленным столбцом Initial.
    """
    df = df.copy()

    df['Initial'] = df['Name'].str.extract(r"([A-Za-z]+)\.")

    df['Initial'] = df['Initial'].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Other",
        "Countess": "Other",
        "Capt": "Other",
        "Col": "Other",
        "Don": "Other",
        "Dr": "Other",
        "Major": "Other",
        "Rev": "Other",
        "Sir": "Other",
        "Jonkheer": "Other",
        "Don": "Other",
        "Dona": "Other",
    })

    return df

def fill_age_column(df, strategy):
    """
    Заполняет пропуски в столбце Age средними значениями по группам в зависимости от их "титула".
    В текущем датасете титулы: ['Miss', 'Mrs', 'Other', 'Mr'].
    Также обрабатываются опечатки.

    Подробное исследование признака:
        EDA.ipynb.

    Args:
        df: Исходный DataFrame.
        cfg_embarked: Настройки предобработки столбца Age из конфигурации.

    Returns:
        DataFrame: DataFrame с заполненными пропусками в столбце Age.
    """
    df = df.copy() # Тут делать копию возможно излишне,
                   # потому что она делается ещё раз при create_initial_column

    if strategy == "mean_by_title":
        df = create_initial_column(df)

        age_means_by_initial = df.groupby("Initial")["Age"].transform("mean")
        df["Age"] = df["Age"].fillna(age_means_by_initial)

    return df
    


def preprocess(df, cfg):
    """
    Применяет полный набор шагов предобработки к исходным данным.
    Исходный DataFrame не изменяется: внутри функции создаётся его копия.

    Args:
        df: Исходный DataFrame с необработанными данными.
        cfg: Конфигурация параметров предобработки.

    Returns:
        DataFrame: Предобработанный набор данных, готовый к обучению модели.
    """
    df = df.copy() 

    df = fill_embarked(df, cfg.preprocessing.embarked)

    if cfg.preprocessing.age.enabled:
        df = fill_age_column(
            df,
            strategy = cfg.preprocessing.age.strategy,
        )

    return df