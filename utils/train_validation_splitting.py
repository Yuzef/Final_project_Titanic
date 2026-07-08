from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from utils.preprocessing import fit_preprocessing, preprocess

def create_family_groups(df, cfg):
    """
    Создаёт идентификаторы семей для StratifiedGroupKFold.

    Группа нужна, чтобы пассажиры из одной семьи попадали целиком либо
    в train, либо в validation. Это снижает риск data leakage.

    Семья определяется как:

        surname + "_" + family_size

    где:
        surname берётся из Name до первой запятой;
        family_size = SibSp + Parch + 1.

    Если solo_as_unique_group=True, пассажиры без родственников получают
    уникальную группу вида:

        solo_<PassengerId>

    Это нужно, чтобы не объединять одиночек с одинаковой фамилией.
    """
    family_cfg = cfg.validation.family

    surname = (
        df[family_cfg.surname_column]
        .str.extract(r"^([^,]+),", expand=False)
        .str.strip()
    )

    sibsp_column = family_cfg.sibsp_column
   
    family_size = (
        df[sibsp_column]
        + df[family_cfg.parch_column]
        + 1
    )

    groups = surname + "_" + family_size.astype(str)

    if family_cfg.solo_as_unique_group:
        solo_mask = family_size == 1

        groups.loc[solo_mask] = (
            "solo_"
            + df.loc[solo_mask, family_cfg.passenger_id_column].astype(str)
        )

    return groups


def make_validation_splits(df, cfg):
    """
    Создаёт train/validation split'ы по стратегии из config.

    Поддерживает:
        - stratified_kfold
        - stratified_group_kfold

    Returns:
        splits: список пар (train_idx, valid_idx)
        groups: семейные группы или None, если используется обычный StratifiedKFold
    """
    y = df[cfg.validation.target_column]

    if cfg.validation.strategy == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=cfg.validation.n_splits,
            shuffle=cfg.validation.shuffle,
            random_state=cfg.general.seed,
        )

        splits = list(splitter.split(df, y))
        groups = None

    elif cfg.validation.strategy == "stratified_group_kfold":
        if cfg.validation.group_by == "family":
            groups = create_family_groups(df, cfg)
        else:
            raise ValueError(f"Unknown group_by: {cfg.validation.group_by}")

        splitter = StratifiedGroupKFold(
            n_splits=cfg.validation.n_splits,
            shuffle=cfg.validation.shuffle,
            random_state=cfg.general.seed,
        )

        splits = list(splitter.split(df, y, groups))

    else:
        raise ValueError(
            f"Unknown validation strategy: {cfg.validation.strategy}"
        )

    return splits, groups

def iter_preprocessed_folds(df, cfg):
    """
    Генератор готовых fold'ов после CV-разбиения и preprocessing.

    На каждом fold:
        1. создаёт train_raw и valid_raw;
        2. проверяет отсутствие family leakage, если используются groups;
        3. fit_preprocessing выполняет только на train_raw;
        4. preprocess применяет к train_raw и valid_raw;
        5. возвращает данные, готовые для обучения модели.

    Важно:
        функция использует yield, поэтому fold'ы не хранятся все сразу в памяти.
        Каждый fold готовится и отдаётся по одному.
    """
    splits, groups = make_validation_splits(df, cfg)

    for fold, (train_idx, valid_idx) in enumerate(splits):
        # Заново делает нумерацию индексов, чтобы не было каши.
        train_raw = df.iloc[train_idx].reset_index(drop=True)
        valid_raw = df.iloc[valid_idx].reset_index(drop=True)

        family_overlap = None

        if groups is not None:
            train_groups = groups.iloc[train_idx]
            valid_groups = groups.iloc[valid_idx]

            overlap = set(train_groups) & set(valid_groups)
            family_overlap = len(overlap)

            if family_overlap != 0:
                raise ValueError(
                    "Data leak: family appears in both train and validation."
                )
        
        state = fit_preprocessing(train_raw, cfg)

        X_train = preprocess(train_raw, cfg, state)
        X_valid = preprocess(valid_raw, cfg, state)

        y_train = train_raw[cfg.validation.target_column]
        y_valid = valid_raw[cfg.validation.target_column]

        yield {
            "fold": fold,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "X_train": X_train,
            "X_valid": X_valid,
            "y_train": y_train,
            "y_valid": y_valid,
            "preprocessing_state": state,
            "family_overlap": family_overlap,
        }

def print_fold_summary(fold_data):
    """
    Печатает краткую диагностику текущего fold'а:
        - номер fold'а;
        - проверку family leakage;
        - размеры X_train / X_valid;
        - распределение target в train / validation.
    """
    print(f"\nFold {fold_data['fold']}")

    if fold_data["family_overlap"] is None:
        print(f"Family overlap check: skipped (No division by families).")
    else:
        print(f"Family overlap: {fold_data['family_overlap']}")
        
    print("X_train:", fold_data["X_train"].shape)
    print("X_valid:", fold_data["X_valid"].shape)
    print("y_train distribution:")
    # Показывает долю выживших / погибших.
     # Пр-ка как отработал Stratified - split
     # должен сохранить похожее распределение классов
    # в каждом fold’е.
    print(fold_data["y_train"].value_counts(normalize=True))
    print("y_valid distribution:")
    print(fold_data["y_valid"].value_counts(normalize=True))