import argparse
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

import transformers
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

# wandb logger for lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

# preprocessing
from preprocessing import Preprocessing
import time
import wandb

#from utils import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf
#from pytorch_lightning.callbacks import Callback

from utils import optimizer_selector
from load import load_obj

class Model2(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfgs.model.model_name
        self.lr = cfgs.train.learning_rate
        self.drop_out = cfgs.train.drop_out
        self.warmup_ratio = cfgs.train.warmup_ratio

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=768,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out)

        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2,
                            dropout=self.drop_out, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(self.drop_out)
        self.classifier = nn.Linear(768*2,1)

        try:  
            self.loss_func = load_obj(cfgs.train.loss_function)()
        except:
            self.loss_func = torch.nn.SmoothL1Loss() # L1Loss -> SmoothL1Loss

    def forward(self, x):
        x = self.plm(x)['logits']

        hidden,(last_hidden, last_cell) = self.lstm(x)
        concat_hidden = torch.cat(last_hidden[0], last_hidden[1],dim=1)
        logits = self.classifier(self.dropout(concat_hidden))
        return logits

    def training_step(self, batch, batch_idx):

        if cfg.train.R_drop:
            x, y = batch
            logits1 = self(x)
            logits2 = self(x)

            # R-drop for regression task
            loss_r = self.loss_func(logits1, logits2)
            loss = self.loss_func(logits1, y.float()) + self.loss_func(logits2, y.float())
            loss = loss + cfg.train.R_drop_alpha*loss_r

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
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())

        self.log("val_loss", loss)
        self.log("val_pearson", pearson_corr)

        return {'val_loss':loss, 'val_pearson_corr':pearson_corr}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        test_pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", test_pearson_corr)
        return test_pearson_corr

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        optimizer = optimizer_selector(cfg.train.optimizer, self.parameters(), lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio*self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]