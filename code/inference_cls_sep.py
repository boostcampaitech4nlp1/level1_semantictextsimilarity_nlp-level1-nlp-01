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
    def __init__(self, cfgs):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfgs.model.model_name
        self.lr = cfgs.train.learning_rate
        self.drop_out = cfgs.train.drop_out
        self.warmup_ratio = cfgs.train.warmup_ratio
        self.dense = torch.nn.Linear(2048, 768)
        #self.dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(self.drop_out)
        self.output = torch.nn.Linear(768, 1)

        #self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            #num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out)

        try:  
            self.loss_func = load_obj(cfgs.train.loss_function)()
        except:
            self.loss_func = torch.nn.SmoothL1Loss() # L1Loss -> SmoothL1Loss

    def forward(self, input):
        # print(type(input), input.size()) #torch.Size([4, 256])
        
        sep = 3
        if('roberta' in self.model_name):
            sep = 2
            
        sep_idx = [item.tolist().index(sep) for item in input]    
        #print(sep_idx)
        
        output = self.plm(input_ids=input,
            attention_mask=None,
            token_type_ids=None)
        
        features = output[0] # last_hidden_state
        #print(features[:, 0, :].size())  # torch.Size([4, 768])
        
        sep_hidden = []
        for i,item in enumerate(features):
            sep_hidden.append(features[i,sep_idx[i],:])
        
        sep_hidden = torch.stack(sep_hidden, dim=0)
        #print(sep_hidden.size()) # torch.Size([4, 768])
        
        x = torch.concat([features[:, 0, :],sep_hidden], dim=1)
        #x = features[:, 0, :] + sep_hidden
        #print(x.size())
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)

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


if __name__ == '__main__':

    # receive arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    #parser.add_argument('--mode', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    mode = ''
    if(mode == 'dev'):
        cfg.path.predict_path = '../data/dev_preprocessed.csv'
    parser = argparse.ArgumentParser()

    # # seed everything
    seed_everything(cfg.train.seed)

    # Load dataloader & model
    dataloader = Dataloader(cfg)

    # Pred using a single model
    if not cfg.inference.ensemble:

        trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step)

        # Inference part
        model = Model.load_from_checkpoint(checkpoint_path=f'./models/roberta-large_new2_32_BS_30_ep_swap.ckpt')
        #model = torch.load(f'./models/tunib_32_BS_30_ep_1e-05_bt_eda.pt')
        #print(cfg.model.saved_name)

        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('../data/sample_submission.csv')
        if(mode == 'dev'):
            output = pd.read_csv('../data/dev.csv')
        output['target'] = predictions
        output.to_csv(f'roberta-large_new_32_BS_30_ep_swap_ez_test_output.csv', index=False)
    
    # Pred using ensemble
    else:

        output = pd.read_csv('../data/sample_submission.csv')
        length = len(output)

        # make void tensor to store each model's predictions
        tmp_sum = torch.zeros((length,),dtype=torch.float32)

        for each in cfg.inference.ensemble:

            trainer = pl.Trainer(gpus=cfg.train.gpus, max_epochs=cfg.train.max_epoch, log_every_n_steps=cfg.train.logging_step)

            # Inference part
            model = Model.load_from_checkpoint(checkpoint_path=f'models/{each}.ckpt')
            each_pred = trainer.predict(model=model, datamodule=dataloader)
            each_pred = torch.cat(each_pred)
            tmp_sum += each_pred

        # divide total_sum by the number of models
        tmp_sum = tmp_sum / len(cfg.inference.ensemble)
        predictions = list(round(float(i), 1) for i in tmp_sum)

        output['target'] = predictions
        output.to_csv('output.csv', index=False)