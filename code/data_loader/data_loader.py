import re

import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import transformers
import torch
import pytorch_lightning as pl

# package for Preprocessing
from emoji import core
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize


class Preprocessing():

    def __init__(self):
        pass

    def preprocessing(self, sentence):
        # remove emojis
        sentence = core.replace_emoji(sentence, replace='')

        # spell_check
        try:
            spell_checked = spell_checker.check(sentence).as_dict()['checked']
        except:
            spell_checked = sentence

        # emoticon_normalize가 'ㅋㅋㅋ' 연쇄 앞뒤의 음절을 지우지 않도록 해당 연쇄 앞뒤에 공백 추가
        p = re.compile('[ㄱ-ㅎ]{2,}')
        pattern_list = p.findall(spell_checked)
        if pattern_list:
            for pattern in pattern_list:
                spell_checked = spell_checked.replace(pattern, ' ' + pattern + ' ')
        
        # '맞앜ㅋㅋ' -> '맞아 ㅋㅋㅋ' 와 같이 정규화합니다.   
        normalized = emoticon_normalize(spell_checked, num_repeats=2)

        return normalized.strip()

        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.prepro_spell_check = Preprocessing()
        self.use_prepro = False

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True,  max_length=128, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)
        
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':

            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)

            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
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