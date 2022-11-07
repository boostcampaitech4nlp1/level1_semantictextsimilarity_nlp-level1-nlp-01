# 개발새발(🐕🐾🐥🐾)조 level1_semantictextsimilarity_nlp-level1-nlp-01

# Role
- PM: 이상문 <br>
- Data: 신혜진 <br>
- Research: 김해원 <br>
- Code review: 양봉석, 임성근 <br>
<br>
<br>

# Model

|Model|huggingface_model_name|github|
|:---|:---|:---|
| KLUE-RoBERTa-large | klue/roberta-large | https://github.com/KLUE-benchmark/KLUE |
| TUNiB-Electra-ko-base | tunib/electra-ko-base | https://github.com/tunib-ai/tunib-electra |
| KoELECTRA-base-v3 | monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA/blob/master/README_EN.md |<br>
<br>

# Dataset
Boostcamp 4기 내부 대회용 STS 데이터셋을 사용합니다. (비공개) <br>
<br>

## 전처리
데이터셋 전처리는 다음과 같이 진행합니다.
```
python code/preprocessing.py
```
이 모듈을 통해 이모지 제거, 맞춤법 교정을, text normalizing을 수행할 수 있습니다. 맞춤법 교정은 py-hanspell(https://github.com/ssut/py-hanspell)을 사용하고, text normaliziing은 soynlp(https://github.com/lovit/soynlp)를 사용합니다.
<br>
<br>

## Data augmentation
```
## TODO: augmentation 코드 넣기
```
<br>
<br>

# Train
학습은 다음과 같이 진행합니다.
<br>

## 1. config 파일 만들기
`./code/config/base_config.yaml` 파일의 양식과 같이 학습 데이터, 사용할 모델, hyperparameter, wandb log 주소 등을 지정해 준 `my_config.yaml` 파일을 만들어줍니다.
<br>

## 2. 학습 수행
터미널 창에 `python train.py --my_config`을 입력하여 학습을 진행합니다. config 파일에 입력된 설정으로 학습이 진행됩니다. 학습이 모두 완료되면 dev set에 대한 pearson correlation 점수가 가장 높은 모델이 `./code/models/` 에 저장됩니다.
<br>
<br>

# Inference
1. 학습 때 사용했던 config 파일을 `inference.py`에 인자로 넣어서 추론을 진행합니다.
`python code/inference.py --config my_config`
2. config 파일의 `Inference` 세팅을 이용하면 여러 모델끼리의 ensemble도 수행해볼 수 있습니다.