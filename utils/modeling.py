import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib

from utils.preprocessing import fit_preprocessing, preprocess

import shutil

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
    """
    if model_cfg.type == "logistic_regression":
        params = dict(model_cfg.params)
        params["random_state"] = cfg.general.seed

        model = LogisticRegression(**params)
    
    elif model_cfg.type == "knn":
        params = dict(model_cfg.params)

        model = KNeighborsClassifier(**params)
    
    elif model_cfg.type == "random_forest":
        params = dict(model_cfg.params)
        params["random_state"] = cfg.general.seed
        # "Если не указан уже n_jobs, то возьми из cfg.modeling.n_jobs."
        # Есть возможность поставить какой-то другой n_jobs для конкретной модели. 
        params.setdefault("n_jobs", cfg.modeling.n_jobs)

        model = RandomForestClassifier(**params)

    elif model_cfg.type == "catboost":
        params = dict(model_cfg.params)
        params["random_seed"] = cfg.general.seed
        params.setdefault("thread_count", cfg.modeling.n_jobs)
        # Чтобы CatBoost не печатал длинный лог обучения.
        params.setdefault("verbose", False)
        # Чтобы CatBoost не создавал лишние служебные файлы.
        params.setdefault("allow_writing_files", False)

        model = CatBoostClassifier(**params)
    
    elif model_cfg.type == "lightgbm":
        params = dict(model_cfg.params)

        params["random_state"] = cfg.general.seed
        params.setdefault("n_jobs", cfg.modeling.n_jobs)
        # Убирает лишний лог LightGBM.
        params.setdefault("verbosity", -1)

        model = LGBMClassifier(**params)

    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")
    
    # Для KNN особенно важно!
    if cfg.modeling.scale_features:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )

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

def save_model_artifact(artifact, cfg, model_name):
    """
    Сохраняет artifact обученного эксперимента в .joblib файл.

    Artifact содержит всё, что нужно для последующего inference:
    обученную модель, preprocessing_state, параметры модели
    и служебную информацию об эксперименте.
    """

    output_dir = Path(
        Path(cfg.paths.trained_models)
        /cfg.general.experiment_name
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / f"{model_name}.joblib"

    joblib.dump(artifact, artifact_path)

    return artifact_path

def train_and_save_full_model(df, cfg, model_cfg):
    """
    Обучает выбранную модель на всём train.csv и сохраняет готовый artifact.

    Эта функция используется после cross-validation:
    fold'ы нужны для оценки качества, а здесь модель заново обучается на всех
    размеченных данных, чтобы сохранённый artifact можно было напрямую
    использовать для inference.
    """
    preprocessing_state = fit_preprocessing(df, cfg)

    X_train = preprocess(df, cfg, preprocessing_state)
    y_train = df[cfg.validation.target_column]

    model = build_model(model_cfg, cfg)
    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "preprocessing_state": preprocessing_state,
        "model_name": model_cfg.name,
        "model_type": model_cfg.type,
        "model_params": dict(model_cfg.params),
        "feature_columns": list(X_train.columns),
        "metric_name": cfg.metric.name,
        "seed": cfg.general.seed,
    }
    
    artifact_path = save_model_artifact(
        artifact = artifact,
        cfg = cfg,
        model_name = model_cfg.name,
    )

    return artifact_path


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
                cfg,
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

    summary = (
        results_df
        # std - покажет разброс между folds.
        .groupby("model_name")["score"]
        .agg(["mean", "std"])
    )

    artifact_paths = {}

    for model_cfg in enabled_models:
        artifact_path = train_and_save_full_model(
            df=df,
            cfg=cfg,
            model_cfg=model_cfg,
        )
        artifact_paths[model_cfg.name] = artifact_path
    
    best_model_info = select_best_model(summary)

    best_artifact_path = save_best_model_copy(
        artifact_paths=artifact_paths,
        best_model_info=best_model_info,
    )

    best_model_info["best_artifact_path"] = best_artifact_path

    return results_df, summary, artifact_paths, best_model_info

def select_best_model(summary):
    """
    Выбирает лучшую модель по summary.

    Критерий:
    1. максимальный mean score;
    2. при равенстве mean — минимальный std.
    """
    sorted_summary = summary.sort_values(
        by=["mean", "std"],
        ascending=[False, True],
    )

    best_model_name = sorted_summary.index[0]
    best_row = sorted_summary.loc[best_model_name]

    return {
        "model_name": best_model_name,
        "mean_score":best_row["mean"],
        "std_score": best_row["std"],
    }

def save_best_model_copy(artifact_paths, best_model_info):
    """
    Создаёт копию лучшего artifact с суффиксом _BEST перед .joblib.
    """
    best_model_name = best_model_info["model_name"]
    source_path = artifact_paths[best_model_name]

    # Метод with_name() заменяет только имя файла, оставляя папку.
    best_model_path = source_path.with_name(
        # stem — имя файла без расширения
        f"{source_path.stem}_BEST{source_path.suffix}"
    )

    shutil.copy2(source_path, best_model_path)

    return best_model_path








