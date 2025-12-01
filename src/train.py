import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor
from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger
from src.models import SimpleDetector
from src.data.LSA64 import LSA64DataModule

MODEL = SimpleDetector()
config = {
    "exp_group": "v2",
    "max_epochs": 300,
    "checkpoint": {
        "monitor": "val_acc"
    },
    "batch_size": 32
}

root_dir = Path(__file__).parent.parent
run = wandb.init(project="SignTranslator", group=config["exp_group"])
checkpoint_callback = ModelCheckpoint(
    monitor=config["checkpoint"]["monitor"],
    mode="max",
    save_top_k=1,
    dirpath=root_dir / "checkpoints" / config["exp_group"] / run.id,
    filename="best_params",
    save_last=False
)
wandb_logger = WandbLogger(
    save_dir=root_dir / "wandb" / config["exp_group"] / run.id
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
trainer = L.Trainer(
    max_epochs=config["max_epochs"],
    callbacks=[RichProgressBar(), ModelCheckpoint(), lr_monitor]
)

dm = LSA64DataModule(root_dir / "data" / "LSA64", batch_size=config["batch_size"])
trainer.fit(MODEL, )