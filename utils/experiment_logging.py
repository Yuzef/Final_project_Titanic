from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path


def save_experiment_logs(results_df, summary, artifact_paths, best_model_info, cfg):
    """
    Сохраняет config, fold-метрики, summary, пути к artifacts
    и информацию о лучшей модели эксперимента.
    """
    if not cfg.logging.enabled:
        return None
    
    experiment_dir = (
        Path(cfg.paths.trained_models)
        / cfg.general.experiment_name
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    if cfg.logging.save_config:
        saved_config_path = experiment_dir / "config.yaml"
        OmegaConf.save(config=cfg, f=saved_config_path)
        saved_paths["config"] = saved_config_path
    
    if cfg.logging.save_fold_results:
        fold_results_path = experiment_dir / "fold_results.csv"
        results_df.to_csv(fold_results_path, index=False)
        saved_paths["fold_result_path"] = fold_results_path

    if cfg.logging.save_summary:
        summary_path = experiment_dir / "summary.csv"
        # groupby("model_name") делает model_name индексом,
        # поэтому возвращаем его в обычную колонку перед сохранением.
        summary.reset_index().to_csv(summary_path, index=False)
        saved_paths["summary"] = summary_path
    
    if cfg.logging.save_artifact_paths:
        artifact_rows = [
            {
                "model_name": model_name,
                "artifact_path": str(artifact_path),
            }
            for model_name, artifact_path in artifact_paths.items()
        ]

        artifact_paths_df = pd.DataFrame(artifact_rows)

        artifact_paths_file = experiment_dir / "artifacts.csv"
        artifact_paths_df.to_csv(artifact_paths_file, index=False)
        saved_paths["artifact_paths"] = artifact_paths_file
    
    if cfg.logging.save_best_model:
        best_model_path = experiment_dir / "best_model.csv"

        best_model_df = pd.DataFrame([{
            "model_name": best_model_info["model_name"],
            "mean_score": best_model_info["mean_score"],
            "std_score": best_model_info["std_score"],
            "best_artifact_path": str(best_model_info["best_artifact_path"]),
        }])

        best_model_df.to_csv(best_model_path, index=False)
        saved_paths["best_model"] = best_model_path
    
    return saved_paths








