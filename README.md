# Level1_STS_01
A solution for STS Competition in the 4nd BoostCamp AI Tech by **개발새발(🐕🐾🐥🐾) (1조)**  


## Role
- PM : 이상문 <br>
- Data : 신혜진 <br>
- Research : 김해원 <br>
- Code review : 양봉석, 임성근 <br>


## Content
- [Competition Abstract](#competition-abstract)
- [Model](#model)
- [Preprocessing](#preprocessing)
- [Data augmentation](#data-augmentation)
- [Train](#train)
- [Inference](#inference)
- [Result](#result)


## Competition Abstract  
- **의미 유사도 판별(Semantic Text Similarity, STS)** 이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크이다.
- STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 AI모델을 구축한다. 
- 0과 5사이의 유사도 점수를 예측하는 것을 목적으로 한다.
- 총 데이터 개수 : 10,974 문장 쌍
  - Train : 9,324
  - Test : 1,100
  - Dev : 550
  
## Project Tree
<pre>
<code>
level1_semantictextsimilarity_nlp-level1-nlp-01
├── README.md
├── code   
│   ├── config
│   │   └── base_config.yaml
│   ├── data_loader
│   │   └── data_loader.py
│   ├── model
│   │   └── model.py
│   ├── utils
│   │   ├── prediction_analysis.py
│   │   └── utils.py
│   ├── saved 
│   │   └── ... # model & submission.csv saved here
│   ├── train.py
│   ├── inference.py
│   └── requirements.txt
├── data
│   ├── DA.png
│   ├── EDA.ipynb
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── preprocessed 
│   │   └── ... # Store preprocessed data
└── install.sh
</code>
</pre>

## Model

|Model|huggingface_model_name|github|
|:---|:---|:---|
| KLUE-RoBERTa-large | klue/roberta-large | https://github.com/KLUE-benchmark/KLUE |
| TUNiB-Electra-ko-base | tunib/electra-ko-base | https://github.com/tunib-ai/tunib-electra |
| KoELECTRA-base-v3 | monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA/blob/master/README_EN.md |<br>


## Preprocessing
데이터셋 전처리는 다음과 같이 진행합니다.
```
python code/preprocessing.py
```
이 모듈을 통해 이모지 제거, 맞춤법 교정을, text normalizing을 수행할 수 있습니다. 맞춤법 교정은 py-hanspell(https://github.com/ssut/py-hanspell)을 사용하고, text normaliziing은 soynlp(https://github.com/lovit/soynlp)를 사용합니다.


## Data augmentation
```
## TODO: augmentation 코드 넣기
```  


## Train
학습은 다음과 같이 진행합니다.  

**1. config 파일 만들기**  
`./code/config/base_config.yaml` 파일의 양식과 같이 학습 데이터, 사용할 모델, hyperparameter, wandb log 주소 등을 지정해 준 `my_config.yaml` 파일을 만들어줍니다. 

**2. 학습 수행**  
터미널 창에 `python train.py --my_config`을 입력하여 학습을 진행합니다. config 파일에 입력된 설정으로 학습이 진행됩니다. 학습이 모두 완료되면 dev set에 대한 pearson correlation 점수가 가장 높은 모델이 `./code/models/` 에 저장됩니다.  


## Inference
1. 학습 때 사용했던 config 파일을 `inference.py`에 인자로 넣어서 추론을 진행합니다.  
`python code/inference.py --config my_config`
2. config 파일의 `Inference` 세팅을 이용하면 여러 모델끼리의 ensemble도 수행해볼 수 있습니다.  


## Result
||Pearson|Rank|
|:---|:---|:---|
|Public|0.9271|7|
|**Private**|**0.9337**|**5**|
