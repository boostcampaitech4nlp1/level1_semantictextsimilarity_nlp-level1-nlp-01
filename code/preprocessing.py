import re
from emoji import core
# from quickspacer import Spacer
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize


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

        # emoticon_normalize가 'ㅋㅋㅋ' 연쇄 뒤의 음절을 지우지 않도록 해당 연쇄 뒤에 공백 추가
        p = re.compile('[ㄱ-ㅎ]{1,}')
        pattern_list = p.findall(spell_checked)
        if pattern_list:
            for pattern in pattern_list:
                spell_checked = spell_checked.replace(pattern, pattern + ' ')
        
        # '맞앜ㅋㅋ' -> '맞아 ㅋㅋㅋ' 와 같이 정규화합니다.   
        normalized = emoticon_normalize(spell_checked, num_repeats=2)

        return normalized.strip()