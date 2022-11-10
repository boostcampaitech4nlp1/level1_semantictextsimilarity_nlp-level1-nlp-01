import argparse

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


def main(cfg):
    # Not use kFold
    if not cfg.train.k_fold:
        # Load dataloader & model
        dataloader = Dataloader(cfg)
        model = Model(cfg)

        # wandb logger
        wandb_logger = WandbLogger(name=cfg.model.saved_name, project=cfg.repo.project_name, entity=cfg.repo.entity)
        wandb.watch(model)

        # checkpoint config
        checkpoint_callback = ModelCheckpoint(dirpath="saved/",
                                            filename=f'{cfg.model.saved_name}',
                                            save_top_k=1, 
                                            monitor="val_pearson",
                                            mode='max')   

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')                                    

        # Train & Test
        trainer = pl.Trainer(gpus=cfg.train.gpus, 
                            max_epochs=cfg.train.max_epoch,
                            log_every_n_steps=cfg.train.logging_step,
                            precision=cfg.train.precision,
                            logger=wandb_logger,
                            callbacks=[checkpoint_callback, lr_monitor])

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

    # Use KFold
    else:
        results = []
        nums_folds = cfg.train.k_fold

        for k in range(1,nums_folds+1):

            # checkpoint config
            checkpoint_callback = ModelCheckpoint(dirpath="saved/",
                                                filename=f'{cfg.model.saved_name}_{str(k)}th_fold',
                                                save_top_k=1, 
                                                monitor="val_pearson",
                                                mode='max')

            model = Model(cfg)
            dataloader = Dataloader(cfg,k-1)
            dataloader.prepare_data()
            dataloader.setup()

            # wandb logger
            wandb_logger = WandbLogger(name=f'{cfg.model.saved_name}_{str(k)}th_fold', project=cfg.repo.project_name)
            wandb.watch(model)

            # Learning rate monitor
            lr_monitor = LearningRateMonitor(logging_interval='step')

            # Train & Test
            trainer = pl.Trainer(gpus=cfg.train.gpus, 
                                max_epochs=cfg.train.max_epoch,
                                log_every_n_steps=cfg.train.logging_step,
                                precision=cfg.train.precision,
                                logger=wandb_logger,
                                callbacks=[checkpoint_callback, lr_monitor])

            trainer.fit(model=model, datamodule=dataloader)
            test_pearson_corr = trainer.test(model=model, datamodule=dataloader)

            results.append(float(test_pearson_corr[0]['test_pearson']))
            wandb.finish()
            
        # Just for final mean KF_score logging
        wandb.init(project=cfg.repo.project_name, entity=cfg.repo.entity)
        wandb.run.name = f'{cfg.model.saved_name}_{str(nums_folds)}_fold_mean'
        KF_mean_score = sum(results) / nums_folds
        wandb.log({"test_pearson_corr": KF_mean_score})
        wandb.finish()


if __name__ == '__main__':
    # receive arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    # seed everything
    seed_everything(cfg.train.seed)

    # main 
    main(cfg)