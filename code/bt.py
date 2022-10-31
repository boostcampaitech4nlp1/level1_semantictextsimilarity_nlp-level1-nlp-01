import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from selenium import webdriver
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.common.by import By
from tqdm import trange, tqdm
import argparse
from emoji import core
import csv

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--single-process')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-setuid-sandbox')
driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)

def translation(sentence, sk='ko', tk='en'):
    """ 
    한국어(Korean) : ko, 영어(english) : en,  일본어(Japanese) : ja, 중국어(Chinese(Simplified)) : zh-CN
    """
    sentence = core.replace_emoji(sentence, replace='')
    driver.get(f'https://papago.naver.com/?sk={sk}&tk={tk}')
    driver.implicitly_wait(7)

    input_box = driver.find_element(By.CSS_SELECTOR, '#sourceEditArea textarea')
    input_box.clear(); input_box.clear();      
    input_box.send_keys(sentence)
    
    time.sleep(1)
    
    driver.find_element(By.CSS_SELECTOR, '#btnTranslate').click()

    time.sleep(1)
    
    result = str(driver.find_element(By.CSS_SELECTOR, "#txtTarget").text)

    input_box.clear(); input_box.clear();  
    
    return result


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../data/train.csv')
    args = parser.parse_args()

    
    train_df = pd.read_csv('../data/train.csv')
    count_train_df = train_df.copy()
    z = dict(train_df['label'].value_counts())
    count_train_df['count'] = train_df['label'].map(z)
    print(z)
    augment_df = train_df[(count_train_df['count']>=270) & (count_train_df['label']!=0)]
    #augment_df = train_df[train_df['label']!=0]
    print(len(augment_df))
    augment_df.head()

    sentence1 = augment_df['sentence_1'].copy()
    sentence1_new = []
    sentence2 = augment_df['sentence_2'].copy()
    sentence2_new = []
    
    f = open('bt1_single.csv', 'w')
    writer = csv.writer(f)
    
    for s in tqdm(sentence1):
        #print('*')
        eng = translation(s, 'ko', 'en')
        kor = translation(eng, 'en', 'ko')
        print(s,'|',kor)
        sentence1_new.append(kor)
        writer.writerow(kor)
    
    f.close()
    #pd.DataFrame(sentence1_new).to_csv('bt1_single.csv')
    
    f2 = open('bt2_single.csv', 'w')
    writer2 = csv.writer(f2)
    for s in tqdm(sentence2):
        #print('*')
        eng = translation(s, 'ko', 'en')
        kor = translation(eng, 'en', 'ko')
        print(s,'|',kor)
        sentence2_new.append(kor)
        writer2.writerow(kor)
    f2.close()
    #pd.DataFrame(sentence2_new).to_csv('bt2_single.csv')
    
    backtranslate_1_df = augment_df.copy()
    backtranslate_1_df['sentence_1'] = sentence1_new
    backtranslate_1_df['sentence_2'] = sentence2_new
    backtranslate_1_df.to_csv('bt_remain.csv')
    
