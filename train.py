from model import build_tft, build_time_series_ds
from config import get_config


def train_model(config):
    train_dataloader, val_dataloader,_ = build_time_series_ds(config)

    trainer, tft = build_tft(config)

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # include this to resume training, comment out to train with new params
        # figure out way to link config and different checkpoints?
        ckpt_path="tft/checkpoints/6/epoch=4-val_loss=787.53.ckpt",
    )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
