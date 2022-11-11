# Level1_STS_01
A solution for STS Competition in the 4nd BoostCamp AI Tech by **개발새발(🐕🐾🐥🐾) (1조)**  


## Role
- PM : 이상문 <br>
- Data : 신혜진 <br>
- Research : 김해원 <br>
- Code review : 양봉석, 임성근 <br>

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
│   ├── config.yaml # wandb sweep config file
│   ├── data_loader
│   │   └── data_loader.py
│   ├── model
│   │   └── model.py
│   └── utils
│   |    ├── prediction_analysis.py
│   |    └── utils.py
│   ├── saved
│   |    └── # model & output.csv saved here
│   ├── inference.py
│   ├── requirements.txt
│   ├── saved
│   └── train.py
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
Main branch 내용과 동일


## Data augmentation
Main branch 내용과 동일


## Train
학습은 다음과 같이 진행합니다.
wandb official document : https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code
1. config.yaml 파일 수정
2. 반복하고자 하는 시행횟수 지정
```
NUM=5 # 5회 반복 지정
```
3. sweep initialize
```
wandb sweep --project your_project_name config.yaml
```
4. train 시작
```
wandb agent --count $NUM your_name/your_project_name/sweepID # 발급되는 sweepID는 그때 마다 다름
```


## Inference
1. 학습 때 사용했던 argument를 지정하여 inference.py 실행해줍니다.
`python code/inference.py --batch_size 16 ....`


## Result
Main branch 내용과 
