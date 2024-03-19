from feature_creation import fc
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

import lightning.pytorch as pl


import pandas as pd

from tft import MyTemporalFusionTransformer


def build_training_data(config):
    df = pd.read_csv(config["training_data_path"])
    df.rename(columns={"period": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df = fc(df, config["time_windows"], config["large_time_windows"])

    training_cutoff = round(df["time_idx"].quantile(config["training_cutoff_quantile"]))
    time_varying_unknown_reals = [x for x in df.columns if "rolling" in x] + [config["target"]]
    time_varying_known_reals = (
        [x for x in df.columns if x.startswith("sin") or x.startswith("cos")]
        + [x for x in df.columns if "shifted" in x]
        + config["time_varying_known_real_additions"]
    )

    return df, training_cutoff, time_varying_unknown_reals, time_varying_known_reals


def build_time_series_ds(config):
    df, training_cutoff, time_varying_unknown_reals, time_varying_known_reals = build_training_data(config)

    ds_train = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=config["target"],
        group_ids=config["group_ids"],
        max_encoder_length=config["max_encoder_len"],
        min_encoder_length= config['min_encoder_len'],
        min_prediction_length=1,
        max_prediction_length=config["max_pred_len"],
        static_categoricals=config["static_categoricals"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["fueltype"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    ds_val = TimeSeriesDataSet.from_dataset(
        ds_train, df, predict=True, stop_randomization=True
    )  # allow_missing_timesteps=True)

    train_dataloader = ds_train.to_dataloader(train=True, batch_size=config["batch_size"])
    val_dataloader = ds_val.to_dataloader(train=False, batch_size=config["val_batch_size"])

    return train_dataloader, val_dataloader, ds_train


def build_tft(config):
    _, _, time_varying_unknown_reals, time_varying_known_reals = build_training_data(config)

    trainer = pl.Trainer(**config["trainer_params"])

    tft = MyTemporalFusionTransformer(  # refactor this to config.py, then try to break it down to a BaseModelWithCovariates -> BaseModel ....... probbaly put this above build time series dataset
        max_encoder_length=config["max_encoder_len"],
        static_categoricals=config["static_categoricals"],
        static_reals=config["static_reals"],
        time_varying_reals_encoder=time_varying_known_reals + time_varying_unknown_reals,
        time_varying_reals_decoder=time_varying_known_reals,
        x_reals=config["static_reals"] + time_varying_known_reals + time_varying_unknown_reals,
        x_categoricals=config["group_ids"],
        **config["tft_params"]
    )

    return trainer, tft
