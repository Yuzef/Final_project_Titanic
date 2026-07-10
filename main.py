import pandas as pd

from configs.config import config
from utils.modeling import run_modeling
from utils.train_validation_splitting import iter_preprocessed_folds, print_fold_summary


def main():
    cfg = config

    train_df = pd.read_csv(cfg.paths.train_csv)

    results_df, summary, artifact_paths, best_model_info = run_modeling(
        df=train_df,
        cfg=cfg,
        folds_iterator=iter_preprocessed_folds,
    )

    print("\nFold results:")
    print(results_df)

    print("\nSummary")
    print(summary)

    print("\nSaved artifacts:")
    for model_name, artifact_path in artifact_paths.items():
        print(f"Model name: {model_name}, Path: {artifact_path}")



if __name__ == "__main__":
    main()










