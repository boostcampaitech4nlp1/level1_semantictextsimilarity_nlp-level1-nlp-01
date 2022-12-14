# Level1_STS_01
A solution for STS Competition in the 4nd BoostCamp AI Tech by **κ°λ°μλ°(ππΎπ₯πΎ) (1μ‘°)**  
</br>


## Role
- **μ΄μλ¬Έ** (PM) : νλ‘μ νΈ κ΄λ¦¬ λ° μ μ²΄μ μΈ λ°©ν₯ μ€μ , λͺ¨λΈ κ²°κ³Ό λͺ¨λν°λ§ λ° μμλΈ λ΄λΉ
- **κΉν΄μ** (Research) : Taskμ μ μ©κ°λ₯ν κΈ°λ² λ¦¬μμΉ λ΄λΉ, λ€μν Optimizer, Loss function μ€ν
- **μ νμ§** (Data) : EDA λ° Data agumetation λ΄λΉ, λͺ¨λΈ μΆλ ₯ κ²°κ³Ό λΆμ
- **μμ±κ·Ό** (Code reviewer) : μ½λ λ¦¬λ·° λ° λ² μ΄μ€λΌμΈ μ½λ μ»€μ€ν°λ§μ΄μ§, AutoML νκ²½(Sweep) κ΅¬μΆ λ΄λΉ
- **μλ΄μ** (Code reviewer) : ν λ¦¬ν¬μ§ν λ¦¬ μ½λ λ¦¬λ·° λ° λ€μν λͺ¨λΈ μ€ν λ΄λΉ
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
- **μλ―Έ μ μ¬λ νλ³(Semantic Text Similarity, STS)** μ΄λ λ λ¬Έμ₯μ΄ μλ―Έμ μΌλ‘ μΌλ§λ μ μ¬νμ§λ₯Ό μμΉννλ μμ°μ΄μ²λ¦¬ νμ€ν¬μ΄λ€.
- STS λ°μ΄ν°μμ νμ©ν΄ λ λ¬Έμ₯μ μ μ¬λλ₯Ό μΈ‘μ νλ AIλͺ¨λΈμ κ΅¬μΆνλ€. 
- 0κ³Ό 5μ¬μ΄μ μ μ¬λ μ μλ₯Ό μμΈ‘νλ κ²μ λͺ©μ μΌλ‘ νλ€.
- μ΄ λ°μ΄ν° κ°μ : 10,974 λ¬Έμ₯ μ
  - Train : 9,324
  - Test : 1,100
  - Dev : 550
</br>


## Project Tree
<pre>
<code>
level1_semantictextsimilarity_nlp-level1-nlp-01
βββ README.md
βββ code   
β   βββ config
β   β   βββ base_config.yaml
β   βββ data_loader
β   β   βββ data_loader.py
β   βββ model
β   β   βββ model.py
β   βββ utils
β   β   βββ prediction_analysis.py
β   β   βββ utils.py
β   βββ saved 
β   β   βββ ... # model & submission.csv saved here
β   βββ train.py
β   βββ inference.py
β   βββ requirements.txt
βββ data
β   βββ DA.png
β   βββ EDA.ipynb
β   βββ train.csv
β   βββ dev.csv
β   βββ test.csv
β   βββ sample_submission.csv
β   βββ preprocessed 
β   β   βββ ... # Store preprocessed data
βββ install.sh
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
λ°μ΄ν°μ μ μ²λ¦¬λ λ€μκ³Ό κ°μ΄ μ§νν©λλ€.
```
python code/preprocessing.py
```
μ΄ λͺ¨λμ ν΅ν΄ μ΄λͺ¨μ§ μ κ±°, λ§μΆ€λ² λ° λμ΄μ°κΈ° κ΅μ , text normalizingμ μνν  μ μμ΅λλ€. 
- λ§μΆ€λ² λ° λμ΄μ°κΈ° κ΅μ  : [py-hanspell](https://github.com/ssut/py-hanspell)
- text normaliziing : [soynlp](https://github.com/lovit/soynlp)
</br>


## Data augmentation
- **Swap** : sentence_1κ³Ό sentence_2λ₯Ό λ°κΏμ λ°μ΄ν° μ¦κ°
- **Backtranslation** : sentence_1κ³Ό sentence_2 μ€ νλλ§ λλ λ λ€ Backtranslation μν
- **EDA** : Synonym Replacement, Random Insertion, Random Swap, Random Deletion μ λͺ¨λ μνν ν κΈΈμ΄κ° κ°μ₯ κΈ΄ κ²μ μ ν  
</br>


## Train
νμ΅μ λ€μκ³Ό κ°μ΄ μ§νν©λλ€.  

**1. config νμΌ λ§λ€κΈ°**  
`./code/config/base_config.yaml` νμΌμ μμκ³Ό κ°μ΄ νμ΅ λ°μ΄ν°, μ¬μ©ν  λͺ¨λΈ, hyperparameter, wandb log μ£Όμ λ±μ μ§μ νμ¬ `my_config.yaml` νμΌμ λ§λ€μ΄μ€λλ€.  

**2. νμ΅ μν**  
ν°λ―Έλ μ°½μ `python train.py --my_config`μ μλ ₯νμ¬ νμ΅μ μ§νν©λλ€. config νμΌμ μλ ₯λ μ€μ μΌλ‘ νμ΅μ΄ μ§νλ©λλ€. νμ΅μ΄ λͺ¨λ μλ£λλ©΄ dev setμ λν pearson correlation μ μκ° κ°μ₯ λμ λͺ¨λΈμ΄ `./code/models/` μ μ μ₯λ©λλ€.  
</br>


## Inference
- νμ΅ λ μ¬μ©νλ config νμΌμ `inference.py`μ μΈμλ‘ λ£μ΄μ μΆλ‘ μ μ§νν©λλ€.  
  - `python code/inference.py --config my_config`
- config νμΌμ `Inference` μΈνμ μ΄μ©νλ©΄ μ¬λ¬ λͺ¨λΈλΌλ¦¬μ ensembleλ μνν΄λ³Ό μ μμ΅λλ€.  
</br>


## Result
||Pearson|Rank|
|:---|:---|:---|
|Public|0.9271|7|
|**Private**|**0.9337**|**5**|

## Wrap-up Report
![NLP-01μ‘°_Wrap_up_λ¦¬ν¬νΈ(μλ‘λμ©).pdf](https://github.com/boostcampaitech4nlp1/level1_semantictextsimilarity_nlp-level1-nlp-01/files/9994739/NLP-01._Wrap_up_.pdf)

