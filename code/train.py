import argparse
import yaml

import wandb
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# Pl modules
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

# Dataloader & Model
from data_loader.data_loader import Dataloader
from model.model import Model


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
    parser.add_argument("--predict_path", default="/opt/ml/data/dev.csv")
    parser.add_argument("--R_drop", default=False)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--loss_fct", default="L1Loss")
    parser.add_argument("--drop_out", default=0.1)
    parser.add_argument("--warmup_step", default=0)
    parser.add_argument("--preprocessing", default=False)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--saved_name", default="test_model", type=str)
    parser.add_argument("--seed", default=2022, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arguments()

    # seed everything
    seed_everything(args.seed)

    with open("config.yaml") as file:

        config = yaml.load(file, Loader=yaml.FullLoader)

        run = wandb.init(config=config)

        hparams = {
            "lr": wandb.config.lr,
            "bs": wandb.config.batch_size,
            "epochs": wandb.config.epochs,
            "precision": wandb.config.precision,
            "R_drop": wandb.config.R_drop,
            "warmup_step": wandb.config.warmup_step,
            "drop_out": wandb.config.drop_out,
            "optimizer": args.optimizer,
            "model_name": args.model_name,
            "train_path": args.train_path,
            "dev_path": args.dev_path,
            "test_path": args.test_path,
            "predict_path": args.predict_path,
        }

        dataloader = Dataloader(hparams)
        model = Model(hparams)

        wandb_logger = WandbLogger(project="sangmun_test_warmup")
        wandb.watch(model)

        # checkpoint config
        checkpoint_callback = ModelCheckpoint(
            dirpath="saved/",
            filename=f"{args.saved_name}",
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
        )

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=hparams["epochs"],
            logger=wandb_logger,
            log_every_n_steps=1,
            precision=hparams["precision"],
            callbacks=[checkpoint_callback, lr_monitor],
        )

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        wandb.finish()
