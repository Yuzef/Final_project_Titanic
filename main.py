# import config


import pandas as pd

from config import config
from src.preprocessing import preprocess

# Проверка препроцессинга.
df = pd.read_csv(config.paths.train_csv)
processed_df = preprocess(df, config)

print(df.isna().sum())
print("---------------------")
print(processed_df.isna().sum())
print("---------------------")
print(processed_df.shape)
print(processed_df.columns.tolist())

# def main():
#     cfg = config
#     train_df = load_train_data(cfg)
#     train_df = preprocess(train_df, cfg)

#     model = build_model(cfg)
#     train_model(model, train_df, cfg)






