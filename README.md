# Level1_STS_01
A solution for STS Competition in the 4nd BoostCamp AI Tech by **개발새발(🐕🐾🐥🐾) (1조)**  
</br>


## Role
- **이상문** (PM) : 프로젝트 관리 및 전체적인 방향 설정, 모델 결과 모니터링 및 앙상블 담당
- **김해원** (Research) : Task에 적용가능한 기법 리서치 담당, 다양한 Optimizer, Loss function 실험
- **신혜진** (Data) : EDA 및 Data agumetation 담당, 모델 출력 결과 분석
- **임성근** (Code reviewer) : 코드 리뷰 및 베이스라인 코드 커스터마이징, AutoML 환경(Sweep) 구축 담당
- **양봉석** (Code reviewer) : 팀 리포지토리 코드 리뷰 및 다양한 모델 실험 담당
</br>


## Content
- [Competition Abstract](#competition-abstract)
- [Project Tree](#project-tree)
- [Model](#model)
- [Preprocessing](#preprocessing)
- [Data augmentation](#data-augmentation)
- [Train](#train)
- [Inference](#inference)
- [Result](#result)
</br>


## Competition Abstract  
- **의미 유사도 판별(Semantic Text Similarity, STS)** 이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크이다.
- STS 데이터셋을 활용해 두 문장의 유사도를 측정하는 AI모델을 구축한다. 
- 0과 5사이의 유사도 점수를 예측하는 것을 목적으로 한다.
- 총 데이터 개수 : 10,974 문장 쌍
  - Train : 9,324
  - Test : 1,100
  - Dev : 550
</br>


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
</br>


## Model

|Model|huggingface_model_name|github|
|:---|:---|:---|
| KLUE-RoBERTa-large | klue/roberta-large | https://github.com/KLUE-benchmark/KLUE |
| TUNiB-Electra-ko-base | tunib/electra-ko-base | https://github.com/tunib-ai/tunib-electra |
| KoELECTRA-base-v3 | monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA/blob/master/README_EN.md |
</br>


## Preprocessing
데이터셋 전처리는 다음과 같이 진행합니다.
```
python code/preprocessing.py
```
이 모듈을 통해 이모지 제거, 맞춤법 및 띄어쓰기 교정, text normalizing을 수행할 수 있습니다. 
- 맞춤법 및 띄어쓰기 교정 : [py-hanspell](https://github.com/ssut/py-hanspell)
- text normaliziing : [soynlp](https://github.com/lovit/soynlp)
</br>


## Data augmentation
- **Swap** : sentence_1과 sentence_2를 바꿔서 데이터 증강
- **Backtranslation** : sentence_1과 sentence_2 중 하나만 또는 둘 다 Backtranslation 수행
- **EDA** : Synonym Replacement, Random Insertion, Random Swap, Random Deletion 을 모두 수행한 후 길이가 가장 긴 것을 선택  
</br>


## Train
학습은 다음과 같이 진행합니다.  

**1. config 파일 만들기**  
`./code/config/base_config.yaml` 파일의 양식과 같이 학습 데이터, 사용할 모델, hyperparameter, wandb log 주소 등을 지정하여 `my_config.yaml` 파일을 만들어줍니다.  

**2. 학습 수행**  
터미널 창에 `python train.py --my_config`을 입력하여 학습을 진행합니다. config 파일에 입력된 설정으로 학습이 진행됩니다. 학습이 모두 완료되면 dev set에 대한 pearson correlation 점수가 가장 높은 모델이 `./code/models/` 에 저장됩니다.  
</br>


## Inference
- 학습 때 사용했던 config 파일을 `inference.py`에 인자로 넣어서 추론을 진행합니다.  
  - `python code/inference.py --config my_config`
- config 파일의 `Inference` 세팅을 이용하면 여러 모델끼리의 ensemble도 수행해볼 수 있습니다.  
</br>


## Result
||Pearson|Rank|
|:---|:---|:---|
|Public|0.9271|7|
|**Private**|**0.9337**|**5**|

## Wrap-up Report
![NLP-01조_Wrap_up_리포트.pdf](https://github.com/boostcampaitech4nlp1/level1_semantictextsimilarity_nlp-level1-nlp-01/files/9994735/NLP-01._Wrap_up_.pdf)

