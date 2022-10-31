import argparse
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import time
from preprocessing import Preprocessing
from omegaconf import OmegaConf
#from utils import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np

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
    def __init__(self, cfgs):
        super().__init__()
        self.model_name = cfgs.model.model_name
        self.batch_size = cfgs.train.batch_size
        self.shuffle = cfgs.data.shuffle
        self.max_length = cfgs.data.max_length

        self.train_path = cfgs.path.train_path
        self.dev_path = cfgs.path.dev_path
        self.test_path = cfgs.path.test_path
        self.predict_path = cfgs.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_length)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.prepro_spell_check = Preprocessing()
        self.use_prepro = cfgs.data.use_prepro

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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=cfgs.data.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self,cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfgs.model.model_name
        self.lr = cfgs.model.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
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

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

def soft_voting(model_names, trainer, dataloader):
    models = torch.nn.ModuleList()
    for name in model_names:
        models.append(torch.load(f'./models/{name}.pt'))

    predictions = []
    for model in models:
        predict = trainer.predict(model=model, datamodule=dataloader)
        predict = list(float(i) for i in torch.cat(predict))
        predictions.append(predict)

    vote_predictions = np.sum(np.array(predictions), axis=0)/len(predictions)
    vote_predictions = torch.from_numpy(vote_predictions)
    vote_predictions = list(round(float(i), 1) for i in vote_predictions)
    
    return vote_predictions

def weighted_voting(model_names, weights, trainer, dataloader):
    models = torch.nn.ModuleList()
    for name in model_names:
        models.append(torch.load(f'./models/{name}.pt'))

    predictions = []
    for idx,model in enumerate(models):
        predict = trainer.predict(model=model, datamodule=dataloader)
        predict = list(float(i)*weights[idx] for i in torch.cat(predict))
        predictions.append(predict)

    vote_predictions = np.sum(np.array(predictions), axis=0)/sum(weights)
    vote_predictions = torch.from_numpy(vote_predictions)
    vote_predictions = list(round(float(i), 1) for i in vote_predictions)
    
    return vote_predictions
    
    
if __name__ == '__main__':


    # receive arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    parser = argparse.ArgumentParser()

    # seed everything
    seed_everything(cfg.train.seed)

    # parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    # parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--max_epoch', default=30, type=int)
    # parser.add_argument('--shuffle', default=True)
    # parser.add_argument('--learning_rate', default=1e-5, type=float)
    # parser.add_argument('--train_path', default='../data/train.csv')
    # parser.add_argument('--dev_path', default='../data/dev.csv')
    # parser.add_argument('--test_path', default='../data/dev.csv')
    # parser.add_argument('--predict_path', default='../data/test.csv')
    # args = parser.parse_args()

    # Load dataloader & model
    dataloader = Dataloader(cfg)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step)

        # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model_names = ['tunib50_BS_32_LR_5e-06', 'tunib50_BS_32_LR_1e-05','tunib30_BS_32_LR_1e-05','tunib30_BS_16_LR_1e-05','tunib_32_BS_30_ep_1e-05_bt_eda']
    weights = [1,1,1,1,2]
    
    predictions = weighted_voting(model_names, weights, trainer, dataloader)
    #model = torch.load('./models/tunib50_BS_32_LR_5e-06.pt')
    #predictions = trainer.predict(model=model, datamodule=dataloader)
    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    #predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    #output = pd.read_csv('../data/dev.csv')
    output['target'] = predictions
    output.to_csv('tunib_weighted_voting5_test.csv', index=False)

    # Inference part
    #model = torch.load(f'./models/{cfg.model.saved_name}.pt')
    #predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    #predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    #output = pd.read_csv('../data/dev_preprocessed.csv')
    #output['target'] = predictions
    #output.to_csv('output_dev.csv', index=False)

