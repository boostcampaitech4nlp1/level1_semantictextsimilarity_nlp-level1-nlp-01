import re
from emoji import core
# from quickspacer import Spacer
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize
import pandas as pd

class Preprocessing():

    def __init__(self):
        # self.spacer = Spacer()
        pass

    def preprocessing(self, sentence):

        # remove emojis
        sentence = core.replace_emoji(sentence, replace='')

        # spacing # tensorflow vs lightning 버전 충돌 문제로 잠시 보류.
        # spaced = self.spacer.space([sentence])[0]

        # spell_check
        try:
            spell_checked = spell_checker.check(sentence).as_dict()['checked']
        except:
            # 에러 발생시 맞춤법 교정은 생략하고 다음 단계로 넘김.
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

# 학습 전에 전처리를 미리 하고 싶다면 python preprocessing.py를 실행한다.
if __name__ == '__main__':
    prepro = Preprocessing()
    train_df = pd.read_csv('../data/train.csv')
    dev_df = pd.read_csv('../data/dev.csv')
    test_df = pd.read_csv('../data/test.csv')

    train_df['sentence_1'] = train_df['sentence_1'].apply(lambda x: prepro.preprocessing(x))
    train_df['sentence_2'] = train_df['sentence_2'].apply(lambda x: prepro.preprocessing(x))
    train_df.to_csv('../data/train_preprocessed.csv')

    dev_df['sentence_1'] = dev_df['sentence_1'].apply(lambda x: prepro.preprocessing(x))
    dev_df['sentence_2'] = dev_df['sentence_2'].apply(lambda x: prepro.preprocessing(x))
    dev_df.to_csv('../data/dev_preprocessed.csv')

    # test_df['sentence_1'] = test_df['sentence_1'].apply(lambda x: prepro.preprocessing(x))
    # test_df['sentence_2'] = test_df['sentence_2'].apply(lambda x: prepro.preprocessing(x))
    # test.to_csv('../data/test_preprocessed.csv')