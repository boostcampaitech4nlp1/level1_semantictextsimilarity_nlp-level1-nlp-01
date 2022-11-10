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
dir = os.path.abspath(os.path.join(dir, os.pardir,os.pardir))
sys.path.append(dir)
from data_loader.data_loader import Dataloader
from model.model import PredictModel
from utils.utils import soft_voting, weighted_voting


def main(cfg):
    # Load dataloader & model
    dataloader = Dataloader(cfg)
    output = pd.read_csv('../data/sample_submission.csv')

    # Pred using a single model
    if not cfg.inference.ensemble:

        trainer = pl.Trainer(gpus=cfg.train.gpus, 
                            max_epochs=cfg.train.max_epoch,
                            log_every_n_steps=cfg.train.logging_step)

        # Inference part
        model = PredictModel.load_from_checkpoint(checkpoint_path=f'./saved/{cfg.model.saved_name}.ckpt')
        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        
        # Save as csv file
        output['target'] = predictions
        output.to_csv('output.csv', index=False)
    
    # Pred using ensemble
    else:
        if not cfg.inference.weighted_ensemble: #  soft voting
            length = len(output)

            # make void tensor to store each model's predictions
            tmp_sum = torch.zeros((length,),dtype=torch.float32)

            for each in cfg.inference.ensemble:
                trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step)

                # Inference part
                if each.endswith('.ckpt'):
                    model = PredictModel.load_from_checkpoint(checkpoint_path=f'models/{each}')
                else:
                    model = torch.load('models/' + each)
                each_pred = trainer.predict(model=model, datamodule=dataloader)
                each_pred = torch.cat(each_pred)
                tmp_sum += each_pred

            # Divide total_sum by the number of models
            tmp_sum = tmp_sum / len(cfg.inference.ensemble)
            predictions = list(round(float(i), 1) for i in tmp_sum)

            # Save as csv file
            output['target'] = predictions
            output.to_csv('output.csv', index=False)

        else: #Weighted voting ensemble
            trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step)
            weights = cfg.inference.weighted_ensemble
            model = Model.load_from_checkpoint(checkpoint_path=f'./saved/{name}')
            vote_predictions = weighted_voting(model, cfg.inference.ensemble, weights, cfg)
            
            # Save as csv file
            output['target'] = vote_predictions
            output.to_csv('output.csv', index=False)


if __name__ == '__main__':

    # receive arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    parser = argparse.ArgumentParser()

    # seed everything
    seed_everything(cfg.train.seed)

    # main
    main(cfg)