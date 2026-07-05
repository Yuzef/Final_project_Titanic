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

def create_age_band(df, cfg):
    """
    Создаёт num_of_bins возрастных групп на основе колонки Age.

    strategy = "equal_width" при num_bins = 5 (-> одинаковая ширина интервалов возраста):
        0: Age <= 16
        1: 16 < Age <= 32
        2: 32 < Age <= 48
        3: 48 < Age <= 64
        4: Age > 64
    
    strategy = "quantile" -> примерно одинаковое количество объектов в каждом бине.
    """
    df = df.copy()
    
    if not cfg.enabled:
        return df
    
    output_column = cfg.output_column

    if cfg.strategy == "equal_width":
        df[output_column] = pd.cut(
            df["Age"],
            bins=cfg.num_bins,
            labels = False,
            include_lowest=True,
        ).astype(int)

    elif cfg.strategy == "quantile":
        df[cfg.output_column] = pd.qcut(
            df["Age"],
            q=cfg.num_bins,
            # вернуть не сами интервалы, а номера групп
            labels=False,
            # Если какие-то границы совпали - duplicate, просто уменьши количество бинов.
            duplicates="drop",
        ).astype(int)

    else:
        raise ValueError(f"Unknown age binning strategy: {cfg.strategy}")
    
    if cfg.drop_original:
        df = df.drop(columns={"Age"})
    
    return df

def create_family_features(df, cfg):
    """
    Создаёт признаки Family_Size и Alone.

    Family_Size = Parch + SibSp
    Family_Size не включает самого пассажира.

    Alone:
        1: пассажир путешествует один
        0: пассажир путешествует с родственниками
    """
    df = df.copy()

    if not cfg.enabled:
        return df
    
    family_size_column = cfg.family_size_column
    alone_column = cfg.alone_column

    df[family_size_column] = df["Parch"] + df["SibSp"]

    df[alone_column] = 0
    df.loc[df[family_size_column] == 0, alone_column] = 1

    if cfg.drop_original:
        df = df.drop(columns=["Parch", "SibSp"])

    return df

def create_fare_range(df, cfg):
    """
    Создаёт признак Fare_Range на основе заданного количества квантилей Fare.

    strategy = "quantile":
        разбивает Fare на num_bins групп так, чтобы в каждой группе
        было примерно одинаковое количество пассажиров.

    При num_bins = 4 получаются квартильные группы:
    самые дешевые билеты, ниже среднего, выше среднего, самые дорогие.
    """
    df = df.copy()

    if not cfg.enabled:
        return df
    
    output_column = cfg.output_column

    if cfg.strategy == "quantile":
        df[output_column] = pd.qcut(
            df["Fare"],
            q=cfg.num_bins,
            labels=False,
            duplicates="drop",
        ).astype(int)
    
    else:
        raise ValueError(f"Unknown fate binning strategy: {cfg.strategy}")

    if cfg.drop_original:
        df = df.drop(columns=["Fare"])
    
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

    if cfg.preprocessing.age_binning.enabled: 
        df = create_age_band(df, cfg.preprocessing.age_binning)
    
    if cfg.preprocessing.family_features.enabled:
        df = create_family_features(df, cfg.preprocessing.family_features)
    
    if cfg.preprocessing.fare_binning.enabled:
        df = create_fare_range(df, cfg.preprocessing.fare_binning)

    return df