import os
import sys

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

# to import utils module
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)
from utils.utils import optimizer_selector, SMARTLoss, load_obj


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = hparams["model_name"]
        self.lr = hparams["lr"]
        self.drop_out = hparams["drop_out"]
        self.warmup_ratio = hparams["warmup_step"]
        self.use_r_drop = hparams["R_drop"]
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)["logits"]
        return x

    def training_step(self, batch, batch_idx):

        if self.use_r_drop:
            x, y = batch
            logits1 = self(x)
            logits2 = self(x)

            # R-drop for regression task
            loss_r = self.loss_func(logits1, logits2)
            loss = self.loss_func(logits1, y.float()) + self.loss_func(
                logits2, y.float()
            )
            loss = loss + loss_r

            self.log("train_loss", loss)
            return loss

        else:
            x, y = batch
            logits = self(x)
            loss = self.loss_func(logits, y.float())
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())

        self.log("val_loss", loss)
        pearson_corr = torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()
        )
        self.log("val_pearson", pearson_corr)

        return {"val_loss": loss, "val_pearson_corr": pearson_corr}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        test_pearson_corr = torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()
        )
        self.log("test_pearson", test_pearson_corr)
        return test_pearson_corr

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_ratio
            * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]