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
        "categorical_encoding": {
            "enabled": True,
            "columns": ["Sex", "Embarked"]
        },
        "features": {
            "use_columns": [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare", # Здесь вот что-то не так, мы делали другую колонку из этой.
                "Embarked"
            ]
        },
        "save_processed": False # Потом поставлю True
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


