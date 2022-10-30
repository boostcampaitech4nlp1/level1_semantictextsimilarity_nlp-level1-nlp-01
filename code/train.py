import argparse
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

# wandb logger for lightning
from pytorch_lightning.loggers import WandbLogger

# preprocessing
from preprocessing import Preprocessing
import time
import wandb

#from utils import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf
#from pytorch_lightning.callbacks import Callback

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, cfg,idx=None):
        super().__init__()
        self.model_name = cfg.model.model_name
        self.batch_size = cfg.train.batch_size
        self.shuffle = cfg.data.shuffle

        self.train_path = cfg.path.train_path
        self.dev_path = cfg.path.dev_path
        self.test_path = cfg.path.test_path
        self.predict_path = cfg.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=128)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.prepro_spell_check = Preprocessing()
        self.use_prepro = cfg.data.use_prepro
        self.k_fold = cfg.train.k_fold
        self.k = idx

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)
        
        if self.use_prepro:
        # 맞춤법 교정 및 이모지 제거
            start = time.time()
            data[self.text_columns[0]] = data[self.text_columns[0]].apply(lambda x: self.prepro_spell_check.preprocessing(x))
            data[self.text_columns[1]] = data[self.text_columns[1]].apply(lambda x: self.prepro_spell_check.preprocessing(x))
            end = time.time()
            print(f"---------- Spell Check Time taken {end - start:.5f} sec ----------")

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':

            if self.k_fold:
                # Split train data only K-times
                train_data = pd.read_csv(self.train_path)
                kf = KFold(n_splits = self.k_fold, shuffle=self.shuffle, random_state=cfg.train.seed)
                all_splits = [k for k in kf.split(train_data)]

                train_indexes, val_indexes = all_splits[self.k]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

                train_inputs, train_targets = self.preprocessing(train_data.loc[train_indexes])
                val_inputs, val_targets = self.preprocessing(train_data.loc[val_indexes])

                self.train_dataset = Dataset(train_inputs,train_targets)
                self.val_dataset = Dataset(val_inputs, val_targets)

            else:
                train_data = pd.read_csv(self.train_path)
                val_data = pd.read_csv(self.dev_path)

                train_inputs, train_targets = self.preprocessing(train_data)
                val_inputs, val_targets = self.preprocessing(val_data)

                self.train_dataset = Dataset(train_inputs, train_targets)
                self.val_dataset = Dataset(val_inputs, val_targets)

        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfgs.model.model_name
        self.lr = cfgs.train.learning_rate
        self.drop_out = cfgs.train.drop_out
        self.warmup_ratio = cfgs.train.warmup_ratio

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out)

        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
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
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio*self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]

if __name__ == '__main__':

    # receive arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    # seed everything
    seed_everything(cfg.train.seed)

    print(cfg)

    if not cfg.train.k_fold:

        # wandb init & setting & config
        wandb.init(project=cfg.repo.project_name, entity=cfg.repo.entity)
        wandb.run.name = cfg.model.saved_name
        wandb.config = {
            "learning_rate": cfg.train.learning_rate,
            "epochs": cfg.train.max_epoch,
            "batch_size": cfg.train.batch_size,
        }

        # Load dataloader & model
        dataloader = Dataloader(cfg)
        model = Model(cfg)


        wandb_logger = WandbLogger(project="sangmun_test2")
        wandb.watch(model)

        # Train & Test
        trainer = pl.Trainer(gpus=cfg.train.gpus, 
                            max_epochs=cfg.train.max_epoch,
                            log_every_n_steps=cfg.train.logging_step,
                            precision=cfg.train.precision,
                            logger=wandb_logger)

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # save model in the models category
        torch.save(model.state_dict(), f'models/{cfg.model.saved_name}.pt')
    else:

        results = []
        nums_folds = cfg.train.k_fold

        for k in range(1,nums_folds+1):

            # wandb init & setting & config
            wandb.init(project=cfg.repo.project_name, entity=cfg.repo.entity)
            wandb.run.name = f'{cfg.model.saved_name}_{str(k)}th_fold'
            wandb.config = {
                "learning_rate": cfg.train.learning_rate,
                "epochs": cfg.train.max_epoch,
                "batch_size": cfg.train.batch_size,
            }

            model = Model(cfg)
            dataloader = Dataloader(cfg,k)
            dataloader.prepare_data()
            dataloader.setup()

            trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step, precision=cfg.train.precision)
            trainer.fit(model=model, datamodule=dataloader)
            test_pearson_corr = trainer.test(model=model, datamodule=dataloader)
            wandb.log({"test_pearson_corr": test_pearson_corr[0]['test_pearson']})

            results.append(float(test_pearson_corr[0]['test_pearson']))
            wandb.finish()

            # Model save code need to added

        # Just for final mean KF_score logging
        wandb.init(project=cfg.repo.project_name, entity=cfg.repo.entity)
        wandb.run.name = f'{cfg.model.saved_name}_{str(nums_folds)}_fold_mean'
        wandb.config = {
                "learning_rate": cfg.train.learning_rate,
                "epochs": cfg.train.max_epoch,
                "batch_size": cfg.train.batch_size,
        }

        KF_mean_score = sum(results) / nums_folds
        wandb.log({"test_pearson_corr": KF_mean_score})
        wandb.finish()


# if __name__ == '__main__':
#     # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
#     # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
#     # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', default='klue/roberta-base', type=str)
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--max_epoch', default=5, type=int)
#     parser.add_argument('--shuffle', default=True)

#     parser.add_argument('--learning_rate', default=1e-5, type=float)
#     parser.add_argument('--train_path', default='../data/train.csv')
#     parser.add_argument('--dev_path', default='../data/dev.csv')
#     parser.add_argument('--test_path', default='../data/dev.csv')
#     parser.add_argument('--predict_path', default='../data/test.csv')
#     parser.add_argument('--loss_function', default='L1Loss')
    
#     parser.add_argument('--preprocessing', default=False)
#     parser.add_argument('--precision', default=32, type=int)
#     parser.add_argument('--dropout', default=0.1, type=float)
#     args = parser.parse_args()

#     # check hyperparameter arguments
#     print(args)

#     # seed everything
#     seed_everything(2022)

#     # wandb init
#     wandb.init(project="sangmun_test", entity="nlp_level1_team1")

#     # wandb.run.name setting
#     run_name = 'roberta_base_' + str(args.max_epoch) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.learning_rate) + '_' + str(args.precision) + '_' + str(args.preprocessing)
#     wandb.run.name = run_name

#     wandb.config = {
#     "learning_rate": args.learning_rate,
#     "epochs": args.max_epoch,
#     "batch_size": int(args.batch_size),
#     }

#     # dataloader와 model을 생성합니다.cls
#     dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
#                             args.test_path, args.predict_path)
#     # num_workers = 4, 

#     model = Model(args.model_name, args.learning_rate, args.dropout)
    
#     # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요 # precision : [32bit(default), 16bit]
#     trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1, precision=args.precision)
#     # WandbLogger 사용 시:
#     # trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger, detect_anomaly=True)
#     # Train part
#     trainer.fit(model=model, datamodule=dataloader)
#     test_pearson_corr = trainer.test(model=model, datamodule=dataloader)
#     wandb.log({"test_pearson_corr": test_pearson_corr[0]['test_pearson']})

#     # save model in the models category
#     torch.save(model, 'models/' + run_name + '.pt')
