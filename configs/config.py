from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "knn_v2_stratified_group_kfold",
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
        "scale_features": True,
        # Использовать все доступные ядра процессора "-1".
        #n_jobs=6 — использовать ровно 6 ядра.
        "n_jobs": 6,

        "models": [
            # -------------- LogReg Baseline ----------------
            {
                "name": "logistic_regression_no_regularization",
                "enabled": False,
                "type": "logistic_regression",
                "params": {
                    "penalty": None,
                    "solver": "lbfgs",
                    "max_iter": 5000,
                },
            }, 
            {
                "name": "logreg_l1",
                "enabled": False,
                "type": "logistic_regression",
                "params": {
                    "penalty": "l1",
                    # C - обратная сила регуляризации.
                    # = 1/λ
                    "C": 0.1,
                    "solver": "liblinear",
                    "max_iter": 5000,
                },

            },
            {
                "name": "logreg_l2",
                "enabled": False,
                "type": "logistic_regression",
                "params": {
                    "penalty": "l2",
                    "C": 10.0, # 0.1 , 10
                    "solver": "lbfgs",
                    "max_iter": 5000,
                },

            },
            {
                "name": "logreg_elasticnet",
                "enabled": False,
                "type": "logistic_regression",
                "params": {
                    "penalty": "elasticnet",
                    "C": 10.0, # 0.1 , 10
                    "solver": "saga",
                    # 0 <= l1_ratio <= 1 
                    # определяет баланс между L1 и L2.
                    "l1_ratio": 0.5,
                    "max_iter": 5000,
                },

            },
            # --------------- KNN -----------------------
            {
                "name": "knn_k3_uniform_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 3,
                    # Соседи имеют разный вес: чем сосед ближе,
                    # тем сильнее он влияет на прогноз.
                    "weights": "uniform",
                    # Cпособ, которым KNN понимает, кто “ближайший сосед”.
                    # euclidean: sqrt(3² + 4²) = 5
                    # manhattan: |3| + |4| = 7
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k3_distance_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 3,
                    # Соседи имеют разный вес: чем сосед ближе,
                    # тем сильнее он влияет на прогноз.
                    "weights": "distance",
                    # Cпособ, которым KNN понимает, кто “ближайший сосед”.
                    # euclidean: sqrt(3² + 4²) = 5
                    # manhattan: |3| + |4| = 7
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k5_uniform_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k5_distance_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 5,
                    "weights": "distance",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k5_distance_manhattan",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 5,
                    "weights": "distance",
                    "metric": "manhattan",
                }

            },
            {
                "name": "knn_k7_distance_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 7,
                    "weights": "distance",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k7_distance_manhattan",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 7,
                    "weights": "distance",
                    "metric": "manhattan",
                }

            },
            {
                "name": "knn_k7_uniform_manhattan",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 7,
                    "weights": "uniform",
                    "metric": "manhattan",
                }

            },
            {
                "name": "knn_k7_uniform_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 7,
                    "weights": "uniform",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k11_uniform_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 11,
                    "weights": "uniform",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k11_distance_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 11,
                    "weights": "distance",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k11_uniform_manhattan",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 11,
                    "weights": "uniform",
                    "metric": "manhattan",
                }

            },
            {
                "name": "knn_k13_distance_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 13,
                    "weights": "distance",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k13_uniform_euclidean",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 13,
                    "weights": "uniform",
                    "metric": "euclidean",
                }

            },
            {
                "name": "knn_k13_uniform_manhattan",
                "enabled": False,
                "type": "knn",
                "params": {
                    "n_neighbors": 13,
                    "weights": "uniform",
                    "metric": "manhattan",
                }

            },
            # ---------------- Random Forest --------------------------
            {
                "name": "rf_100_depth_none",
                "enabled": True,
                "type": "random_forest",
                "scale_features": False, # ?????????
                "params": {
                    # Сколько деревьев в лесу.
                    "n_estimators": 100,
                    "max_depth": None,
                    # Минимальное количество объектов в узле, необходимое,
                    # чтобы дерево могло попробовать разделить этот узел.
                    "min_samples_split": 2,
                    # Минимум объектов в листе.
                    "min_samples_leaf": 1,
                    # Сколько признаков смотреть при каждом разбиении.
                    # При каждом разбиении узла дерево рассматривает не все признаки,
                    # а случайно выбранные √N признаков,
                    # где N — общее количество признаков.
                    "max_features": "sqrt",
                    "bootstrap": True,
                }
            },
            {
                "name": "rf_200_depth_5",
                "enabled": True,
                "type": "random_forest",
                "scale_features": False,
                "params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "n_jobs": -1,
                }
            },
            {
                "name": "rf_300_depth_7_leaf_2",
                "enabled": True,
                "type": "random_forest",
                "scale_features": False,
                "params": {
                    "n_estimators": 300,
                    "max_depth": 7,
                    "min_samples_split": 4,
                   "min_samples_leaf": 2,
                   "max_features": "sqrt",
                   "bootstrap": True,
                   "n_jobs": -1,
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


