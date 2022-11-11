import os, sys
import argparse

import pandas as pd
import numpy as np
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything

dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)
from data_loader.data_loader import Dataloader
from model.model import PredictModel
from utils.utils import soft_voting, weighted_voting


def main(hparams):

    # Load dataloader & model
    dataloader = Dataloader(hparams)
    output = pd.read_csv("../data/sample_submission.csv")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hparams['epochs'],
        log_every_n_steps=1,
    )

    # Inference part
    model = Model.load_from_checkpoint(
        checkpoint_path=f"./saved/{hparams['saved_name']}.ckpt"
    )
    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # Save as csv file
    output["target"] = predictions
    output.to_csv("./saved/output.csv", index=False)

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="klue/roberta-small", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--train_path", default="/opt/ml/data/train.csv")
    parser.add_argument("--dev_path", default="/opt/ml/data/dev.csv")
    parser.add_argument("--test_path", default="/opt/ml/data/dev.csv")
    parser.add_argument("--predict_path", default="/opt/ml/data/test.csv")
    parser.add_argument("--R_drop", default=False)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--loss_fct", default="L1Loss")
    parser.add_argument("--drop_out", default=0.1)
    parser.add_argument("--warmup_step", default=0)
    parser.add_argument("--preprocessing", default=False)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--saved_name", default="test_model", type=str)
    parser.add_argument('--seed', default=2022, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # receive arguments
    args = arguments()

    # seed everything
    seed_everything(args.seed)

    hparams = {
            "lr": args.lr,
            "bs": args.batch_size,
            "epochs": args.epochs,
            "precision": args.precision,
            "R_drop": args.R_drop,
            "warmup_step": args.warmup_step,
            "drop_out": args.drop_out,
            "optimizer": args.optimizer,
            "model_name": args.model_name,
            "train_path": args.train_path,
            "dev_path": args.dev_path,
            "test_path": args.test_path,
            "predict_path": args.predict_path,
            'saved_name':args.saved_name
        }

    # main
    main(hparams)
