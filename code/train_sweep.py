import argparse
import yaml
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

import transformers
import torch
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

from utils import optimizer_selector, loss_fct_selector
from load import load_obj

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
    def __init__(self, hparams):
        super().__init__()
        self.model_name = hparams['model_name']
        self.batch_size = hparams['bs']
        self.shuffle = True

        self.train_path = hparams['train_path']
        self.dev_path = hparams['dev_path']
        self.test_path = hparams['test_path']
        self.predict_path = hparams['predict_path']

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=128)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.prepro_spell_check = Preprocessing()
        self.use_prepro = False
        self.k_fold = 0
        self.k = 0

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
        
        if args.preprocessing:
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
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = hparams['model_name']
        self.lr = hparams['lr']
        self.drop_out = 0.1 # hparams['drop_out']
        self.warmup_ratio = hparams['warmup_step']
        self.use_r_drop = False # hparams['R_drop']
        self.optimizer = hparams['optimizer']

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = loss_fct_selector(hparams['loss_fct'])

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):

        if self.use_r_drop:
            x, y = batch
            logits1 = self(x)
            logits2 = self(x)

            # R-drop for regression task
            loss_r = self.loss_func(logits1, logits2)
            loss = self.loss_func(logits1, y.float()) + self.loss_func(logits2, y.float())
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

        optimizer = optimizer_selector(self.optimizer, self.parameters(), lr=self.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_ratio*self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {'scheduler':scheduler, 'interval':'step', 'frequency':1}

        return [optimizer], [scheduler]

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
        logits = self.classifier(self.dropout(hidden))
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
        optimizer = optimizer_selector(self.optimizer, self.parameters(), lr=self.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_ratio*self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {'scheduler':scheduler, 'interval':'step', 'frequency':1}

        return [optimizer], [scheduler]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--shuffle', default=True)

    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='/opt/ml/data/train_sez_preprocessed.csv')
    parser.add_argument('--dev_path', default='/opt/ml/data/dev_preprocessed.csv')
    parser.add_argument('--test_path', default='/opt/ml/data/dev_preprocessed.csv')
    parser.add_argument('--predict_path', default='/opt/ml/data/dev_preprocessed.csv')

    parser.add_argument('--R_drop', default=False)
    parser.add_argument('--optimizer',default='AdamW')
    parser.add_argument('--loss_fct', default='L1Loss')
    parser.add_argument('--drop_out', default=0.1)
    parser.add_argument('--warmup_step', default=0)
    
    parser.add_argument('--preprocessing', default=False)
    parser.add_argument('--precision', default=16, type=int)

    args = parser.parse_args()

    # # check hyperparameter arguments
    print(args)

    with open('/opt/ml/code/config/sweep_config.yaml') as file:

        config = yaml.load(file, Loader=yaml.FullLoader)

        run = wandb.init(config=config)

        hparams = {
            'model_name':args.model_name,
            'lr':wandb.config.learning_rate,
            'bs':wandb.config.batch_size,
            'epochs':wandb.config.max_epoch,
            'precision':wandb.config.precision,
            #'R_drop':wandb.config.R_drop,
            'loss_fct':wandb.config.loss_fct,
            'optimizer':wandb.config.optimizer,
            'warmup_step':wandb.config.warmup_step,
            #'drop_out':wandb.config.drop_out,

            'train_path':args.train_path,
            'dev_path':args.dev_path,
            'test_path':args.test_path,
            'predict_path':args.predict_path
        }

        dataloader = Dataloader(hparams)
        model = Model(hparams)

        wandb_logger = WandbLogger(project="sangmun_test_warmup")
        wandb.watch(model)

        # checkpoint config
        checkpoint_callback = ModelCheckpoint(dirpath="models/",
                                            filename = str(hparams['warmup_step']) + '_' + str(hparams['optimizer']) + '_' + str(hparams['lr']),
                                            #filename=f'{hparams['optmizer']}_{hparams['lr']}',
                                            save_top_k=1, 
                                            monitor="val_pearson",
                                            mode='max')

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(gpus=1,
                            max_epochs=hparams['epochs'],
                            logger=wandb_logger,
                            log_every_n_steps=1,
                            precision=hparams['precision'],
                            callbacks=[checkpoint_callback, lr_monitor])

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        wandb.finish()