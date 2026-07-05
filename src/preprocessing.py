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

def create_initial_column(df, cfg):
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

    output_column = cfg.output_column

    df[output_column] = df["Name"].str.extract(r"([A-Za-z]+)\.")

    df[output_column] = df[output_column].replace({
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
        if "Initial" not in df.columns:
            raise ValueError("Initial column is required for age strategy mean_by_title")

        age_means_by_initial = df.groupby("Initial")["Age"].transform("mean")
        df["Age"] = df["Age"].fillna(age_means_by_initial)

    else:
        raise ValueError(f"Unknown age strategy: {strategy}")
        
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

def encode_categorical_columns(df, cfg):
    """
    Кодирует категориальные признаки:
    - mapping: ручное преобразование значений в числа ("Sex")
    - one_hot: one-hot encoding через pd.get_dummies ("Embarked", "Initials")
    """
    df = df.copy()

    if not cfg.enabled:
        return df
    
    if cfg.mapping.enabled:
        for column, mapping in cfg.mapping.columns.items():
            if column not in df.columns:
                raise ValueError(f"Column for mapping encoding not found: {column}")
            # Пр-ка есть ли в колонке значения,
            # для которых мы не написали правило кодирования в mapping
            # unknown_values = values_in_data - values_in_mapping
            unknown_values = set(df[column].dropna().unique()) - set(mapping.keys())
            if unknown_values:
                raise ValueError(
                    f"Unknown values in column {column}: {unknown_values}"
                )
            df[column] = df[column].map(mapping).astype(int)
    
    if cfg.one_hot.enabled:
        for column in cfg.one_hot.columns:
            if column not in df.columns:
                raise ValueError(f"Column for one-hot encoding not found: {column}")
        
        df = pd.get_dummies(
            df,
            columns=list(cfg.one_hot.columns), # OmegaConf ListConfig -> обычный Python list
            drop_first=cfg.one_hot.drop_first,
            dtype=int,
        )
    
    return df

def select_model_features(df, cfg):
    """
    Оставляет только признаки, которые должны пойти в модель:

    - явно указанные в use_columns
    - колонки с префиксами из include_prefixes
    """
    df = df.copy()

    feature_columns = list(cfg.use_columns)

    for prefix in cfg.include_prefixes:
        prefix_columns = [
            column
            for column in df.columns
            if column.startswith(prefix)
        ]
        feature_columns.extend(prefix_columns)
    
    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns after preprocessing: {missing_columns}")
    
    return df[feature_columns]

def preprocess(df, cfg):
    """
    Применяет полный набор шагов предобработки к исходным данным:
        1. fill Embarked
        2. fill Age
        3. create Age_band
        4. create Family_Size / Alone
        5. create Fare_Range
        6. encode categories
        7. select_model_features
    Исходный DataFrame не изменяется: внутри функции создаётся его копия.

    Args:
        df: Исходный DataFrame с необработанными данными.
        cfg: Конфигурация параметров предобработки.

    Returns:
        DataFrame: Предобработанный и отобранный набор данных, готовый к обучению модели.
    """
    df = df.copy() 

    if cfg.preprocessing.embarked.enabled:
        df = fill_embarked(df, cfg.preprocessing.embarked)
    
    if cfg.preprocessing.initial.enabled:
        df = create_initial_column(df, cfg.preprocessing.initial)

    if cfg.preprocessing.age.enabled:
        df = fill_age_column(
            df, strategy = cfg.preprocessing.age.strategy,
        )

    if cfg.preprocessing.age_binning.enabled: 
        df = create_age_band(df, cfg.preprocessing.age_binning)
    
    if cfg.preprocessing.family_features.enabled:
        df = create_family_features(df, cfg.preprocessing.family_features)
    
    if cfg.preprocessing.fare_binning.enabled:
        df = create_fare_range(df, cfg.preprocessing.fare_binning)
    
    if cfg.preprocessing.categorical_encoding.enabled:
        df = encode_categorical_columns(
            df, cfg.preprocessing.categorical_encoding
            )

    df = select_model_features(df, cfg.preprocessing.features)

    return df

