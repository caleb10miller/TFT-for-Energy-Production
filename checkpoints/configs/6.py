

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss, MAE, MAPE, RMSE, SMAPE
from pytorch_forecasting.data import GroupNormalizer

import torch.nn as nn


checkpoint_callback = ModelCheckpoint(
    dirpath="tft/checkpoints/6",
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=1,
    verbose=False,
    monitor="val_loss",
    mode="min",
    save_weights_only=False,  # Set to False to save the optimizer state as well # default
)


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")


def get_config():
    return {
        # file paths
        "training_data_path": "/home/luke/projects/jupyterlab/Notebooks/tg/tft/data/power_consumption_by_fuel_type.csv",
        # general and feature creation params
        "target": "value",
        "group_ids": ["fueltype"],
        "static_categoricals": ["fueltype"],
        "static_reals": ["encoder_length", "value_center", "value_scale"],
        "time_windows": [
            2,
            4,
            12,
            24,
            48,
        ],  # can change
        "large_time_windows": [168, 730, 8760],  # can change # DO NOT DO LESS THAN PRED LEN, MESSES THINGS UP
        "max_pred_len": 168,
        "max_encoder_len": 168 * 8,
        "min_encoder_len": 168 * 2,
        "training_cutoff_quantile": 0.9,
        "batch_size": 64,  # can change to accommodate for larger models
        "val_batch_size": 8,
        # trainer_params
        "trainer_params": {
            "accelerator": "gpu",
            "max_epochs": 100,
            "enable_model_summary": True,
            "gradient_clip_val": 0.014,
            "limit_train_batches": 100,
            "callbacks": [checkpoint_callback, lr_logger, early_stop_callback],
            "logger": logger,
        },
        # tft params
        "tft_params": {
            "hidden_size": 32,  # can change
            "lstm_layers": 2,  # can change
            "dropout": 0.5,  # can change
            "output_size": 7,  # number of quantiles to map to
            "loss": QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            "attention_head_size": 4,  # can change
            "learning_rate": 0.003,  # can change
            "hidden_continuous_size": 16,  # can change
            "hidden_continuous_sizes": {},
            "log_interval": 10,
            "log_val_interval": 10,
            "log_gradient_flow": False,
            "optimizer": "Ranger",
            "reduce_on_plateau_patience": 4,
            "time_varying_categoricals_encoder": [],
            "time_varying_categoricals_decoder": [],
            "categorical_groups": {},
            "embedding_sizes": {"fueltype": (8, 5)},  # where does 5 come from
            "embedding_paddings": [],
            "embedding_labels": {
                "fueltype": {"COL": 0, "NG": 1, "NUC": 2, "OIL": 3, "OTH": 4, "SUN": 5, "WAT": 6, "WND": 7}
            },
            "monotone_constaints": {},
            "share_single_variable_networks": False,
            "causal_attention": True,
            "logging_metrics": nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
            "output_transformer": GroupNormalizer(
                method="standard",
                groups=["fueltype"],
                center=True,
                scale_by_group=False,
                transformation="softplus",
                method_kwargs={},
            ),
        },
        # other
        "time_varying_known_real_additions": ["year"],
    }
