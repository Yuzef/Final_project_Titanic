from omegaconf import OmegaConf

config_dict = {
    'general': {
        "experiment_name": "Titanic",
        "seed": 0xFACED,
        "num_classes": 2 
    },
    "paths": {
        "train_csv": "train.csv",
        "test_csv": "test.csv",
        "output_dir": "outputs"
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
                "drop_first": False,   # для LogisticRegression можно попробовать поставить True
            },
        },
        "family_features": {
            "enabled": True,
            "family_size_column": "Family_Size",
            "alone_column": "Alone",
            "drop_original": False,      # удалять ли потом SibSp и Parch
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
                "Age_band",     # feature engineering
                "SibSp",
                "Parch",
                "Fare_Range",   # feature engineering
                "Family_Size",  # feature engineering
                "Alone",        # feature engineering
            ],
            "include_prefixes": [ # columns after one-hot-encoding
                "Embarked_",
                "Initial_",
            ],

        },
        "save_processed": False   # Потом поставлю True
    },

    "training": {
        "num_epochs": 10,
        "early_stopping_epochs": 5,
        "Ir": 1e-4,
        "mixed_precision": True,
        "device": "cuda",
        "save_best": True,
        "save_last": False,
        "drop_last": True
    },
    "dataloader_params": {
        "batch_size": 64,
        "num_workers": 8,
        "pin_memory": False,
        "persistent_workers": True,
        "shuffle": True,
        "drop_last": True,
    },
    "loss": {
        "params": {
            
        }
    }
}


config = OmegaConf.create(config_dict)


