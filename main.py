import pandas as pd

from configs.config import config
from utils.modeling import run_modeling
from utils.train_validation_splitting import iter_preprocessed_folds
from utils.experiment_logging import save_experiment_logs
from utils.inference import create_submission_from_artifact

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

    saved_log_paths = save_experiment_logs(
        results_df=results_df,
        summary=summary,
        artifact_paths=artifact_paths,
        best_model_info=best_model_info,
        cfg=cfg,
    )

    if cfg.inference.enabled:
        test_df = pd.read_csv(cfg.paths.test_csv)

        submission, submission_path = create_submission_from_artifact(
            test_df=test_df,
            cfg=cfg,
       )

        print("\nSubmission saved:")
        print(submission_path)
        print(submission.head())    


if __name__ == "__main__":
    main()










