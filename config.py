from omegaconf import OmegaConf

config_dict = {
    "paths": {
        "train.csv": "train.csv",
        "test.csv": "test.csv"
    }
}

config = OmegaConf.create(config_dict)


