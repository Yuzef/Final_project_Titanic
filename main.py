import pandas as pd

from configs.config import config
from utils.preprocessing import preprocess, fit_preprocessing
from utils.train_validation_splitting import iter_preprocessed_folds, print_fold_summary


def main():
    cfg = config

    df = pd.read_csv(cfg.paths.train_csv)

    for fold_data in iter_preprocessed_folds(df, cfg):
        print_fold_summary(fold_data)



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





