import pandas as pd

from configs.config import config
from utils.preprocessing import preprocess, fit_preprocessing
from utils.validation import make_validation_splits


def main():
    cfg = config

    df = pd.read_csv(cfg.paths.train_csv)

# Универсальный цикл по fold’ам.
# Он работает и для обычного StratifiedKFold, и для StratifiedGroupKFold.
    splits, groups = make_validation_splits(df, cfg)

    # Распаковка folds
    for fold, (train_idx, valid_idx) in enumerate(splits):
        # Каждый элемент внутри splits — это пара:
        # splits[0] = ([0, 1, 2, 4], [3, 5])
        # fold = 0, 1, 2, 3, 4, num_of_splits
        # (train_idx, valid_idx) = ([0, 1, 2, 4], [3, 5])
        print(f"\nFold {fold}")
        # .reset_index(drop=True) делает индексы чистыми и последовательными
        train_raw = df.iloc[train_idx].reset_index(drop=True)
        valid_raw = df.iloc[valid_idx].reset_index(drop=True)

    if groups is not None:
        # Разбиваем по семейным индексам.
        train_groups = groups.iloc[train_idx]
        valid_groups = groups.iloc[valid_idx]

        # Пр-ка на family leakage.
        overlap = set(train_groups) & set(valid_groups)
        print("Family overlap:", len(overlap)) # должно быть "0"

        assert len(overlap) == 0, (
            "Data leak: family appears in both train and validation"
        )
# ------------------Конец разбиения на train и validation folds.-------------------------

    state = fit_preprocessing(train_raw, cfg)

    X_train = preprocess(train_raw, cfg, state)
    X_valid = preprocess(valid_raw, cfg, state)

    y_train = train_raw[cfg.validation.target_column]
    y_valid = valid_raw[cfg.validation.target_column]

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("y_train distribution:")
    print(y_train.value_counts(normalize=True))
    print("y_valid distribution:")
    print(y_valid.value_counts(normalize=True))

if __name__ == "__main__":
    main()




# Проверка препроцессинга.
# df = pd.read_csv(config.paths.train_csv)

# # Fit preprocessing only on train, then reuse this state for validation/test.
# state = fit_preprocessing(df, config)
# processed_df = preprocess(df, config, state)


# print(df.isna().sum())
# print("---------------------")
# print(processed_df.isna().sum())
# print("---------------------")
# print(processed_df.shape)
# print(processed_df.columns.tolist())





