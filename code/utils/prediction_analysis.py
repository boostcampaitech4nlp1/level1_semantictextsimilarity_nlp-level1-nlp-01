import os, math

import pandas as pd
import torch
import torchmetrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
args = parser.parse_args()
pred_file = args.pred_file

df = pd.read_csv(pred_file)

# Get total pearson correlation
pearson_corr = torchmetrics.functional.pearson_corrcoef(torch.tensor(df['target']), torch.tensor(df['label']))
print(f'\n##### total pearson_corr:{pearson_corr:.4f}#####\n')


# Grouping by label
# 0~1: 0, 1~2: 1, 2~3: 2, 3~4: 3, 4~5: 4
def grouping(x):
    if x ==5:
        return 4
    else:
        return math.trunc(x)
df['group'] = df['label'].apply(grouping)


# Get pearson correlation by group
def get_group_corr(x):
    return torchmetrics.functional.pearson_corrcoef(torch.tensor(x['target'].values), torch.tensor(x['label'].values))

print(f"##### The number of label per each group: #####\n{df.groupby('group').apply(len)}\n")
print("#"*40)
print(f"\n##### pearson_corr by group: #####\n{df.groupby('group').apply(get_group_corr)}")

