import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib

from utils.preprocessing import fit_preprocessing, preprocess

def get_enabled_models(cfg):
    """
    Возвращает только те модели из config, у которых enabled=True.
    """

    return [
        model_cfg
        for model_cfg in cfg.modeling.models
        if model_cfg.enabled
    ]

def build_model(model_cfg, cfg):
    """
    Создаёт sklearn-модель по настройкам из config.
    Пока поддерживаем только LogisticRegression.
    """
    if model_cfg.type == "logistic_regression":
        params = dict(model_cfg.params)
        params["random_state"] = cfg.general.seed

        model = LogisticRegression(**params)
        if cfg.modeling.scale_features:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")

    return model

def evaluate_model(model, X_valid, y_valid, cfg):
    """
    Считает метрики на validation-части fold'а.
    """
    metric_name = cfg.metric.name

    if metric_name == "accuracy":
        predictions = model.predict(X_valid)
        # accuracy_score - уже реализована в sklearn
        score = accuracy_score(y_valid, predictions)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return {
        "metric": metric_name,
        "score": score,
    }

def run_modeling(df, cfg, folds_iterator):
    """
    Запускает все enabled-модели на всех fold'ах.

    Важно:
    fold'ы создаются один раз, и внутри каждого fold'а прогоняются все модели.
    Так мы не пересчитываем preprocessing лишний раз для каждой модели.
    """
    results = []
    enabled_models = get_enabled_models(cfg)

    for fold_data in folds_iterator(df, cfg):
        for model_cfg in enabled_models:
            model = build_model(model_cfg, cfg)

            model.fit(
                fold_data["X_train"],
                fold_data["y_train"],
            )

            metrics = evaluate_model(
                model,
                fold_data["X_valid"],
                fold_data["y_valid"],
            )

            result = {
                "model_name": model_cfg.name,
                "model_type": model_cfg.type,
                "fold": fold_data["fold"],
                "params": dict(model_cfg.params),
                **metrics,
            }

            results.append(result)
    
    results_df = pd.DataFrame(results)

    summary = {
        results_df
        # std - покажет разброс между folds.
        .groupby("model_name")["score"].agg(["mean", "std"])
    }

    return results_df, summary


def save_model_artifact(artifact, cfg, model_name):
    """
    Сохраняет обученную модель и preprocessing_state через joblib.
    """
    








