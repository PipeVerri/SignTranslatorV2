import lightning as L
import torch
from lightning.pytorch.cli import ReduceLROnPlateau
from torchmetrics.classification import Accuracy
from models import SimpleRNN
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm

class SimpleDetector(L.LightningModule):
    def __init__(self, hidden_layers=5, hidden_width=144, lr=1e-4, weight_decay=1e-4, lr_reduction_factor=0.5, patience=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleRNN(hidden_layers=hidden_layers, output_dim=65, hidden_dim=hidden_width)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=65)
        self.validation_accuracy = Accuracy(task="multiclass", num_classes=65)

    def training_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        loss = nn.functional.cross_entropy(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.train_accuracy(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, y = batch
        y_pred = self.model(x, lengths)
        self.validation_accuracy(y_pred, y)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.validation_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.validation_accuracy.reset()

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.reset()

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True, on_epoch=False, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.hparams.lr_reduction_factor, patience=self.hparams.patience, monitor="train_acc")
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "train_acc"
        }