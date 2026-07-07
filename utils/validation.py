from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


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

    if "sibsp_column" in family_cfg:
        sibsp_column = family_cfg.sibsp_column
    else:
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