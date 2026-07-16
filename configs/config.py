from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "catboost_st_gr_kfold_bootstrap_search_300_lr005_d4_l2_10",
        "seed": 0xFACED,
        "num_classes": 2 
    },
    "paths": {
        "train_csv": "data_raw/train.csv",
        "test_csv": "data_raw/test.csv",
        "trained_models": "trained_models"
    },
    "validation": {
        "enabled": True,
        "strategy": "stratified_group_kfold",  # "stratified_kfold" "stratified_group_kfold"
        "n_splits": 5,
        "shuffle": True,
        "target_column": "Survived",
        "group_by": "family",
        "family": {
            "surname_column": "Name",
            "sibsp_column": "SibSp",
            "parch_column": "Parch",
            "passenger_id_column": "PassengerId",
            "solo_as_unique_group": True,
            # Если человек путешествует один,
            # лучше считать его отдельной “семьёй”,
            # иначе можно случайно объединить незнакомых людей
            # с одинаковой фамилией.
        }
    },
    "preprocessing": {
        "embarked": {
            "enabled": True,
            "strategy": "most_frequent",
        },
        "age": {
            "enabled": True,
            "strategy": "mean_by_title"
        },
        "initial": {
            "enabled": True,
            "output_column": "Initial",
        },
        "age_binning": {
            "enabled": True,
            "strategy": "equal_width",   # "quantile"
            "output_column": "Age_band",
            "num_bins": 5,
            "drop_original": True,
        },
        "categorical_encoding": {
            "enabled": True,
            "mapping": {
                "enabled": True,
                "columns": {
                    "Sex": {
                        "male": 0,
                        "female": 1,
                    },
                },
            },
            "one_hot": {
                "enabled": True,
                "columns": ["Embarked", "Initial"],
                "drop_first": True,   # для LogisticRegression можно попробовать поставить True
            },
        },
        "family_features": {
            "enabled": True,
            "family_size_column": "Family_Size",
            "alone_column": "Alone",
            "drop_original": True,      # удалять ли потом SibSp и Parch
                                        # (KNN, LogRef - True, деревья - False.)
        },
        "fare": { # если в тесте будет пропуск, то заменяем его значением median из train.
            "enabled": True,
            "strategy": "median",
        },
        "fare_binning": {
            "enabled": True,
            "strategy": "quantile",
            "output_column": "Fare_Range",
            "num_bins": 4,
            "drop_original": True,
        },
        "features": {
            "given_columns": [ # что изначально дали?
                "PassengerId",
                "Survived",
                "Pclass",
                "Name",
                "Sex",
                "Age",     # Исходная колонка до FE
                "SibSp",   # Исходная колонка
                "Parch",   # Исходная колонка
                "Ticket",  # Удаляем
                "Fare",    # Исходная колонка до FE
                "Cabin",   # Удаляем
                "Embarked",
            ],
            "use_columns": [    # что используем?
                "Pclass",
                "Sex",
                # "Age",
                "Age_band",     # feature engineering
                # "SibSp",
                # "Parch",
                "Fare_Range",   # feature engineering
                "Family_Size",  # feature engineering
                "Alone",        # feature engineering
            ],
            "include_prefixes": [ # columns after one-hot-encoding
            "Embarked_",
            "Initial_",
            ],

        },
        "save_processed": False 
    },
    "modeling": {
        "enabled": True,
        # Приведение всех числовых признаков к одному масштабу.
        # StandardScaler(): x_scaled = (x - mean) / std
        # LogReg - True, KNN - True, RandomForest - False, Boosting - False
        "scale_features": False,
        # Использовать все доступные ядра процессора "-1".
        #n_jobs=6 — использовать ровно 6 ядер.
        "n_jobs": 6,

        "models": [
            # ---------------- CatBoost: bootstrap_type search --------------------------
            {
                "name": "catboost_300_lr_005_d4_l2_10_bootstrap_bayesian",
                "enabled": True,
                "type": "catboost",
                "params": {
                    "iterations": 300,
                    "learning_rate": 0.05,
                    "depth": 4,
                    "l2_leaf_reg": 10,
                    "loss_function": "Logloss",
                    "eval_metric": "Accuracy",
                    "bootstrap_type": "Bayesian",
                },
            },
        ]
    },
    "metric": {
        "name": "accuracy"
    },
    "inference": {
        "enabled": True,
        "model_name": "logreg_l2", # если выбрать inference вручную
                                # по названию .joblib файла.
        "use_best_model": True, # выберет _BEST .joblib 
        "id_column": "PassengerId",
        "prediction_column": "Survived",
        "submission_dir": "submissions",
    },
    "logging": {
        "enabled": True,
        "save_config": True,
        "save_fold_results": True,
        "save_summary": True,
        "save_artifact_paths": True, 
        "save_best_model": True,
        "save_readable_report": True,
    },


    
    # "training": {
    #     "num_epochs": 10,
    #     "early_stopping_epochs": 5,
    #     "Ir": 1e-4,
    #     "mixed_precision": True,
    #     "device": "cuda",
    #     "save_best": True,
    #     "save_last": False,
    #     "drop_last": True
    # },
    # "dataloader_params": {
    #     "batch_size": 64,
    #     "num_workers": 8,
    #     "pin_memory": False,
    #     "persistent_workers": True,
    #     "shuffle": True,
    #     "drop_last": True,
    # },
    # "loss": {
    #     "params": {
            
    #     }
    # }
}


config = OmegaConf.create(config_dict)


