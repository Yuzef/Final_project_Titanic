from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "33_xgboost_st_gr_kfold_min_child_weight_search_50_lr005_d3",
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
        "strategy": "stratified_group_kfold",  # "stratified_kfold", 
                                               # "stratified_group_kfold"
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
                "enabled": True, # False for pure catboost native categorical.
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
                "drop_first": True,   # для LogisticRegression можно попробовать
                                      # поставить True
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
                # "Age",        # drop original ?
                "Age_band",     # feature engineering
                # "SibSp",      # drop original ?
                # "Parch",      # drop original ?
                # "Fare",       # drop original ?
                "Fare_Range",   # feature engineering
                "Family_Size",  # feature engineering
                "Alone",        # feature engineering
                # "Embarked",   # for catboost
                # "Initial",    # for catboost
            ],
            "include_prefixes": [ # columns after one-hot-encoding
                "Embarked_",
                "Initial_",
            ],
            # "cat_features": [
            #     "Sex", "Embarked", "Initial",
            #     ]

        },
    },
    "modeling": {
        "enabled": True,
        # Приведение всех числовых признаков к одному масштабу.
        # StandardScaler(): x_scaled = (x - mean) / std
        # LogReg - True, KNN - True, RandomForest - False, Boosting - False
        "scale_features": True, # True для DL, False для бустингов.
        # Использовать все доступные ядра процессора "-1".
        #n_jobs=6 — использовать ровно 6 ядер.
        "n_jobs": 6,

        "models": [
            {
                "name": "dnn_mlp_h16_relu",
                "enabled": True,
                "type": "pytorch_mlp",
                "params": {
                    "hidden_dim": 16,
                    "activation": "relu",
                },
            },
        ]
    },
    "dl": {
        "enabled": True,
    
        "training": {
            "num_epochs": 10,
            "device": "auto", # код сам выберет cuda / mps / cpu
# Автоматический выбор устройства
# if torch.cuda.is_available():
# 	device = torch.device("cuda")
# elif torch.backends.mps.is_available():
# 	device = torch.device("mps")
# else:
# 	device = torch.device("cpu")
	
# print(f" Используется устройство: {device}")
            "mixed_precision": True,
            "verbose": False,
            "early_stopping_epochs": 5,
            "lr": 1e-4,
            
        },
        "dataloader_params": {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "shuffle": True,
            # Если последний batch получился неполным, он будет отброшен.
            "drop_last": False, # Ставлю False, т.к. dataset маленький.
        },
        "optimizer": {
            "name": "adam",
            "params": {
                "lr": "${dl.training.lr}",
                # Регуляризация.
                "weight_decay": 0.0001,
            },
        },
        "loss": {
            "name": "cross_entropy",
            "params": {
            }
        },
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
}

config = OmegaConf.create(config_dict)


# modeling.models
#   что обучаем

# dl.training
#   сколько и где обучаем

# dl.dataloader_params
#   как подаем данные батчами

# dl.optimizer
#   как обновляем веса

# dl.loss
#   какую функцию ошибки используем

