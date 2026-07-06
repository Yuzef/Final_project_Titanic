import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def fill_embarked(df, cfg, state=None):
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
        fill_value = state["embarked_fill_value"]
        df["Embarked"] = df["Embarked"].fillna(fill_value)
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

def fill_age_column(df, strategy, state):
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

        age_fill_values = df["Initial"].map(state["age_means_by_initial"])
        df["Age"] = df["Age"].fillna(age_fill_values)

    else:
        raise ValueError(f"Unknown age strategy: {strategy}")

    return df

def create_age_band(df, cfg, state):
    """
    Создаёт num_of_bins возрастных групп на основе колонки Age (сохранённых age_bins).

    Границы age_bins должны быть вычислены в fit_preprocessing() только на train
    и затем переиспользованы для train/validation/test.

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

    if cfg.strategy in ["equal_width", "quantile"]:
        df[output_column] = pd.cut(
            df["Age"],
            bins=state["age_bins"],
            labels = False,
            include_lowest=True,
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

def fill_fare_column(df, cfg, state):
    """
    Заполняет пропуски в Fare значением, вычисленным на train.
    """
    df = df.copy()

    if not cfg.enabled:
        return df

    if cfg.strategy == "median":
        df["Fare"] = df["Fare"].fillna(state["fare_fill_value"])
    else:
        raise ValueError(f"Unknown fare strategy: {cfg.strategy}")

    return df

def create_fare_range(df, cfg, state=None):
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
        df[output_column] = pd.cut(
            df["Fare"],
            bins=state["fare_bins"],
            labels=False,
            include_lowest=True,
        ).astype(int)
    
    else:
        raise ValueError(f"Unknown fate binning strategy: {cfg.strategy}")

    if cfg.drop_original:
        df = df.drop(columns=["Fare"])
    
    return df

def encode_categorical_columns(df, cfg, state):
    """
    Кодирует категориальные признаки:
    - mapping: ручное преобразование значений в числа ("Sex")
    - one_hot: one-hot encoding через sklearn ("Embarked", "Initials")
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

    # Исходные категориальные колонки заменяются числовыми бинарными признаками,
    # которые модель может использовать для обучения.
    if cfg.one_hot.enabled:
        one_hot_columns = state["one_hot_columns"]
        encoder = state["one_hot_encoder"]

        # содержит числа 0 и 1
        encoded_values = encoder.transform(df[one_hot_columns])
        encoded_columns = encoder.get_feature_names_out(one_hot_columns)

        # Преобразуем NumPy-массив обратно в pandas DataFrame.
        encoded_df = pd.DataFrame(
            encoded_values,
            columns=encoded_columns,
            # сохраняет исходные индексы строк.
            index=df.index,
        )
    # Удаляем исходные категориальные колонки.
    df = df.drop(columns=one_hot_columns)
    # означает: приклеить encoded_df справа от df.
    df = pd.concat([df, encoded_df], axis=1)
    
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

# -----------------------------------------------------------------------------------------

def fit_preprocessing(df, cfg):
    """
    Вычисляет и сохраняет параметры предобработки на ОБУЧАЮЩИХ данных.

    Эта функция нужна, чтобы правила preprocessing были найдены ТОЛЬКО на train
    и затем ОДИНАКОВО ПРИМЕНЯЛИСЬ к validation/test. Например, здесь считаются
    значения для заполнения пропусков, границы бинов и другие параметры,
    КОТОРЫЕ НЕ ДОЛЖНЫ ПЕРЕСЧИТЫВАТЬСЯ ЗАНОВО НА НОВЫХ ДАННЫХ.

    Args:
        df: Обучающий DataFrame, на котором вычисляются параметры preprocessing.
        cfg: Конфигурация preprocessing.

    Returns:
        dict: Словарь с сохранёнными параметрами preprocessing.
    """
    fit_df = df.copy()

    state = {}

    if cfg.preprocessing.embarked.enabled:
        if cfg.preprocessing.embarked.strategy == "most_frequent":
            state["embarked_fill_value"] = df["Embarked"].mode().iloc[0]
            fit_df["Embarked"] = fit_df["Embarked"].fillna(state["embarked_fill_value"])
    
    if cfg.preprocessing.initial.enabled:
        fit_df = create_initial_column(fit_df, cfg.preprocessing.initial)

    # Фиксируем OHE с помощью sklearn
    if cfg.preprocessing.categorical_encoding.one_hot.enabled:
        one_hot_columns = list(cfg.preprocessing.categorical_encoding.one_hot.columns)

        encoder = OneHotEncoder(
            # вернуть обычный NumPy-массив,
            # на больших данных с большим числом категорий
            # лучше часто оставлять значение по умолчанию.
            sparse_output=False,
            # Для неизвестного значения encoder просто поставит нули
            # во всех one-hot колонках этого признака.
            handle_unknown="ignore",
            dtype=int,
        )

        encoder.fit(fit_df[one_hot_columns])

        state["one_hot_encoder"] = encoder
        state["one_hot_columns"] = one_hot_columns

    if cfg.preprocessing.fare.enabled:
        if cfg.preprocessing.fare.strategy == "median":
            state["fare_fill_value"] = fit_df["Fare"].median()
        else:
            raise ValueError(f"Unknown fare strategy: {cfg.preprocessing.fare.strategy}")

    if cfg.preprocessing.fare_binning.enabled:
        if cfg.preprocessing.fare_binning.strategy == "quantile":
            # Будем сохранять только границы интервалов.
            _, fare_bins = pd.qcut(
                df["Fare"],
                q=cfg.preprocessing.fare_binning.num_bins,
                labels=False,
                retbins=True, # Когда True будет возвращаться два значения:
                # fare_groups — номер квантильной группы для каждого пассажира - не интересует;
                # fare_bins — реальные границы этих групп.
                duplicates="drop",
            )
            # На случай если в тесте будет пассажиры превышающий интервал (дороже/дешевле билет),
            # меньше чем бесплатно "0" он заплатить не может.
            # Пример: fare_bins = [0, 7.9, 14.5, 31.0, inf]
            fare_bins[0] = 0
            fare_bins[-1] = float("inf")
            state["fare_bins"] = fare_bins
    
    if cfg.preprocessing.age.enabled:
        if cfg.preprocessing.age.strategy == "mean_by_title":
            age_means = (
                fit_df.groupby(cfg.preprocessing.initial.output_column)["Age"]
                .mean()
                .to_dict()
            )
        state["age_means_by_initial"] = age_means
        # Заполнили df средними значениями, их же тоже сохранили в state.
        fit_df["Age"] = fit_df["Age"].fillna(
            fit_df[cfg.preprocessing.initial.output_column].map(age_means)
        )

    # Теперь разбираемся с фиксацией bins для age.
    if cfg.preprocessing.age_binning.enabled:
        if cfg.preprocessing.age_binning.strategy == "equal_width":
            _, age_bins = pd.cut(
                fit_df["Age"],
                bins=cfg.preprocessing.age_binning.num_bins,
                labels=False,
                retbins=True,
                include_lowest=True,
            )
        elif cfg.preprocessing.age_binning.strategy == "quantile":
            _, age_bins = pd.qcut(
            fit_df["Age"],
            q=cfg.preprocessing.age_binning.num_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        else:
            raise ValueError(
                f"Unknown age binning strategy: {cfg.preprocessing.age_binning.strategy}"
            ) 

        age_bins[0] = 0
        age_bins[-1] = float("inf")
        state["age_bins"] = age_bins


    return state


def preprocess(df, cfg, state):
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
        df = fill_embarked(df, cfg.preprocessing.embarked, state)
    
    if cfg.preprocessing.initial.enabled:
        df = create_initial_column(df, cfg.preprocessing.initial)

    if cfg.preprocessing.age.enabled:
        df = fill_age_column(
            df, strategy = cfg.preprocessing.age.strategy, state = state
        )

    if cfg.preprocessing.age_binning.enabled: 
        df = create_age_band(df, cfg.preprocessing.age_binning, state)
    
    if cfg.preprocessing.family_features.enabled:
        df = create_family_features(df, cfg.preprocessing.family_features)
    
    if cfg.preprocessing.fare.enabled:
        df = fill_fare_column(df, cfg.preprocessing.fare, state)

    if cfg.preprocessing.fare_binning.enabled:
        df = create_fare_range(df, cfg.preprocessing.fare_binning, state)
    
    if cfg.preprocessing.categorical_encoding.enabled:
        df = encode_categorical_columns(
            df, cfg.preprocessing.categorical_encoding, state
            )

    df = select_model_features(df, cfg.preprocessing.features)

    return df

