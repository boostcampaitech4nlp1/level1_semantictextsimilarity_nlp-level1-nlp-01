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
        inputs = pd.DataFrame(inputs)
        self.inputs = inputs['input_ids'].tolist()
        self.attention_mask = inputs['attention_mask'].tolist()
        self.token_type_ids = inputs['token_type_ids'].tolist()
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_mask[idx]), torch.tensor(self.token_type_ids[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_mask[idx]), torch.tensor(self.token_type_ids[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):

    def __init__(self, cfg,idx=None):
        super().__init__()
        self.model_name = cfg.model.model_name
        self.batch_size = cfg.train.batch_size
        self.shuffle = cfg.data.shuffle
        self.max_length = cfg.data.max_length

        self.train_path = cfg.path.train_path
        self.dev_path = cfg.path.dev_path
        self.test_path = cfg.path.test_path
        self.predict_path = cfg.path.predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_length)
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
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(item['sentence_1'], item['sentence_2'], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
            data.append(outputs)
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)
        
        if self.use_prepro:
            start = time.time()
            data[self.text_columns[0]] = data[self.text_columns[0]].apply(lambda x: self.prepro_spell_check.preprocessing(x))
            data[self.text_columns[1]] = data[self.text_columns[1]].apply(lambda x: self.prepro_spell_check.preprocessing(x))
            end = time.time()
            print(f"---------- Spell Check Time taken {end - start:.5f} sec ----------")

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        inputs = self.tokenizing(data)
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':

            if self.k_fold:
                # 학습 데이터를 k개로 나눕니다.
                train_data = pd.read_csv(self.train_path)
                dev_data = pd.read_csv(self.dev_path)
                total_data = pd.concat([train_data,dev_data]).reset_index(drop=True)

                kf = KFold(n_splits = self.k_fold, shuffle=self.shuffle, random_state=cfg.train.seed)
                all_splits = [k for k in kf.split(total_data)]

                train_indexes, val_indexes = all_splits[self.k]
                train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

                train_inputs, train_targets = self.preprocessing(total_data.loc[train_indexes])
                val_inputs, val_targets = self.preprocessing(total_data.loc[val_indexes])

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
            
    def collate_fn(self, batch):
        label_list = []
        data_list = {'input_ids':[], 'attention_mask':[], 'token_type_ids':[]}
        pad = 0
        maxl = 0
        for _input, _att_mask, _tok_type_ids, _label in batch:
            if pad not in _att_mask.tolist():
                maxl = self.max_length
            elif _att_mask.tolist().index(pad) > maxl: 
                maxl = _att_mask.tolist().index(pad)
            
            data_list['input_ids'].append(_input)
            data_list['attention_mask'].append(_att_mask)
            data_list['token_type_ids'].append(_tok_type_ids)
            label_list.append(_label)
            
        for k in data_list:
            _data = data_list[k]
            for i,v in enumerate(_data):
                _data[i] = v[:maxl]
        
        feature_list = []
        for i in range(len(data_list['input_ids'])):
            tmp = torch.stack([data_list['input_ids'][i], data_list['attention_mask'][i], data_list['token_type_ids'][i]], dim=0).cpu()
            feature_list.append(tmp)
        
        return torch.stack(feature_list, dim=0).long(), torch.tensor(label_list).long()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)