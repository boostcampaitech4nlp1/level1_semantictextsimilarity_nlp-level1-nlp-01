import os
import sys

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

# to import utils module
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir,os.pardir))
sys.path.append(dir)
from utils.utils import optimizer_selector, SMARTLoss, load_obj


class Model(pl.LightningModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate
        self.drop_out = cfg.train.drop_out
        self.warmup_ratio = cfg.train.warmup_ratio
        self.cfg = cfg

        if(self.cfg.train.cls_sep.lower() == "none"):
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels=1,
                hidden_dropout_prob=self.drop_out,
                attention_probs_dropout_prob=self.drop_out)
        else:
            input_size = 1536
            if('roberta' in self.model_name):
                input_size = 2048
            if(self.cfg.train.cls_sep.lower() == "add"):
                input_size = input_size/2
            
            self.dense = torch.nn.Linear(input_size, 768)
            self.dropout = torch.nn.Dropout(self.drop_out)
            self.output = torch.nn.Linear(768, 1)

            self.plm = transformers.AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                hidden_dropout_prob=None,
                attention_probs_dropout_prob=None)

        self.loss_func = load_obj(cfg.train.loss_function)()
        self.optimizer = optimizer_selector(cfg.train.optimizer, self.parameters(), lr=self.lr)

        # smart_loss
        if cfg.train.smart_loss:
            def eval_fn(embed):
                outputs = self.plm.roberta(inputs_embeds=embed, attention_mask=None)
                pooled = outputs[0] 
                logits = self.plm.classifier(pooled) 
                return logits 
            def reg_loss_fn(p, q):
                return ((p-q)**2).mean()

            self.eval_fn = eval_fn
            self.regularizer = SMARTLoss(eval_fn=eval_fn, loss_fn=reg_loss_fn)

    def forward(self, data):
        input_ids = data[:,0,:].long()
        attention_mask = data[:,1,:].long()
        token_type_ids = data[:,2,:].long()
        
        if(self.cfg.train.cls_sep.lower() == "none"):
            x = self.plm(input_ids)['logits']
            
        else:
            sep = 1
            sep_idx = [item.tolist().index(sep)-1 for item in token_type_ids]  
            
            output = self.plm(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

            features = output[0]
            sep_hidden = []
            for i,item in enumerate(features):
                sep_hidden.append(features[i,sep_idx[i],:])
            
            sep_hidden = torch.stack(sep_hidden, dim=0)
            
            x = torch.concat([features[:, 0, :],sep_hidden], dim=1)
            if(self.cfg.train.cls_sep.lower() == "add"):
                x = features[:, 0, :] + sep_hidden
                
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.output(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        if self.cfg.train.R_drop:
            x, y = batch
            logits1 = self(x)
            logits2 = self(x)

            # R-drop for regression task
            loss_r = self.loss_func(logits1, logits2)
            loss = self.loss_func(logits1, y.float()) + self.loss_func(logits2, y.float())
            loss = loss + cfg.train.R_drop_alpha*loss_r

            self.log("train_loss", loss)
            return loss

        elif self.cfg.train.smart_loss:
            x, y = batch
            logits = self(x)
            # Apply loss
            loss = self.loss_func(logits, y.float())
            embed = self.plm.roberta.embeddings.word_embeddings(x)
            state = self.eval_fn(embed)
            smart_loss = self.regularizer(embed, state)
            smart_loss = loss + 0.02 * smart_loss
            return smart_loss
        
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

        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze().float())
        self.log("val_pearson", pearson_corr)

        return {'val_loss':loss, 'val_pearson_corr':pearson_corr}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        test_pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze().float())
        self.log("test_pearson", test_pearson_corr)

        return test_pearson_corr

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_ratio*self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {'scheduler':scheduler, 'interval':'step', 'frequency':1}

        return [self.optimizer], [scheduler]


class PredictModel(pl.LightningModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate
        self.drop_out = cfg.train.drop_out
        self.warmup_ratio = cfg.train.warmup_ratio
        self.cfg = cfg

        if(self.cfg.train.cls_sep.lower() == "none"):
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels=1,
                hidden_dropout_prob=self.drop_out,
                attention_probs_dropout_prob=self.drop_out)
        else:
            input_size = 1536
            if('roberta' in self.model_name):
                input_size = 2048
            if(self.cfg.train.cls_sep.lower() == "add"):
                input_size = input_size/2
            
            self.dense = torch.nn.Linear(input_size, 768)
            self.dropout = torch.nn.Dropout(self.drop_out)
            self.output = torch.nn.Linear(768, 1)

            self.plm = transformers.AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                hidden_dropout_prob=None,
                attention_probs_dropout_prob=None)

        try:  
            self.loss_func = load_obj(cfg.train.loss_function)()
        except:
            self.loss_func = torch.nn.SmoothL1Loss() # L1Loss -> SmoothL1Loss

    def forward(self, data):

        input_ids = data[0].long()
        attention_mask = data[1].long()
        token_type_ids = data[2].long()
        
        if(self.cfg.train.cls_sep.lower() == "none"):
            x = self.plm(input_ids)['logits']
            
        else:
            sep = 1
            sep_idx = [item.tolist().index(sep)-1 for item in token_type_ids]  
            
            output = self.plm(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

            features = output[0]
            sep_hidden = []
            for i,item in enumerate(features):
                sep_hidden.append(features[i,sep_idx[i],:])
            
            sep_hidden = torch.stack(sep_hidden, dim=0)
            
            x = torch.concat([features[:, 0, :],sep_hidden], dim=1)
            if(self.cfg.train.cls_sep.lower() == "add"):
                x = features[:, 0, :] + sep_hidden
                
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.output(x)['logits']

        return x

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()