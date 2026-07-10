from pathlib import Path

import joblib
import pandas as pd

from utils.preprocessing import preprocess

def create_submission_from_artifact(test_df, cfg):
    """
    Загружает сохранённый _BEST artifact модели автоматически
    или указанный в config вручную и создаёт submission-файл.
    """
    if cfg.inference.use_best_model:
        artifact_files = list(
            (
                Path(cfg.paths.trained_models)
                / cfg.general.experiment_name
            ).glob("*_BEST.joblib")
        )

        if len(artifact_files) != 1:
            raise ValueError(
                f"Expected exactly one BEST artifact, found {len(artifact_files)}"
            )

        artifact_path = artifact_files[0]
    else:    
        artifact_path = (
            Path(cfg.paths.trained_models)
            / cfg.general.experiment_name
            / f"{cfg.inference.model_name}.joblib"
        )

    artifact = joblib.load(artifact_path)

    model = artifact["model"]
    preprocessing_state = artifact["preprocessing_state"]

    X_test = preprocess(test_df, cfg, preprocessing_state)

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        cfg.inference.id_column: test_df[cfg.inference.id_column],
        cfg.inference.prediction_column: predictions.astype(int)
    })

    submission.to_csv(cfg.inference.submission_path, index=False)

    return submission, cfg.inference.submission_path





