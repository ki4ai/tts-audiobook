import numpy as np
import re
import pickle as pkl

import torch
import torch.utils.data as data

from text import text_to_sequence, cmudict
from config_dict import emo_dict, spk_dict

english_word = {'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이', 'f': '에프', 'g': '쥐', 'h': '에이치',
               'i': '아이', 'j': '제이', 'k': '케이', 'l': '엘', 'n': '엔', 'm': '엠', 'o': '오', 'p': '피',
               'q': '큐', 'r':'얼', 's': '에스', 't': '티', 'u':'유', 'v':'브이', 'w':'더블유', 'x': '엑스',
               'y': '와이', 'z': '지'}
english_word_keys = english_word.keys()

lexicon_pickle = 'ssml/dictionary/lexicon.pickle'
sub_pickle = 'ssml/dictionary/sub.pickle'
num_sub_pickle = 'ssml/dictionary/num_sub.pickle'

# lexicon: english to korean
# lexicon 기본 규칙: 참고 사전 <국립국어원 표준국어대사전>
with open(lexicon_pickle, 'rb') as handle:
    lexicon = pkl.load(handle)
# sub 기본 규칙
with open(sub_pickle, 'rb') as handle:
    sub = pkl.load(handle)
# num_sub 기본 규칙
with open(num_sub_pickle, 'rb') as handle:
    num_sub = pkl.load(handle)


def read1to999(n):
    units = [''] + list('십백천')
    nums = '일이삼사오육칠팔구'
    result = []
    i = 0
    while n > 0:
        n, r = divmod(n, 10)
        if r > 0:
            if units[i] == '':
                result.append(nums[r - 1] + units[i])
            else:
                if r == 1:
                    result.append(units[i])
                else:
                    result.append(nums[r - 1] + units[i])
        i += 1
    return ''.join(result[::-1])


def readNumM(n):
    result = ''
    if n >= 1000000000000:
        r, n = divmod(n, 10000000000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '조'
        else:
            result += tmp + "조"
    if n >= 100000000:
        r, n = divmod(n, 100000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '억'
        else:
            result += tmp + "억"
    if n >= 10000:
        r, n = divmod(n, 10000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '만'
        else:
            result += tmp + "만"
    result += read1to999(n)
    return result


def readNumK(intNum):
    """
    한글로 숫자 읽기
    """
    korean_num = {"1": "한", "2": "두", "3": "세", "4": "네", "5": "다섯", "6": "여섯", "7": "일곱",
                  '8': "여덟", "9": "아홉", "10": "열", "20": "스물", "30": "서른", "40": "마흔", "50": "쉰",
                  "60": "예순", "70": "일흔", "80": "여든", "90": "아흔"}
    tmp_list = list(korean_num.keys())
    num_list = list()
    for num in tmp_list:
        num_list.append(int(num))
    num_list.sort(reverse=True)
    result = ""
    for num in num_list:
        if intNum >= num:
            intNum -= num
            result += korean_num[str(num)]
    return result


def removeS(word):
    return word.replace('\\', '').replace('"', '').lower()


class Dataset(data.Dataset):
    def __init__(self, sentences, use_lexicon, text_cleaners=['korean_english_cleaners'], cmudict_path='data/cmu_dictionary', default_speaker=6, default_emotion=0, default_langauge=0):
        # SSML list
        self.ssml = [[i, s] for i, s in enumerate(sentences)]
        self.use_lexicon = use_lexicon

        self.emo_dict = emo_dict
        self.spk_dict = spk_dict
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.text_cleaners = text_cleaners
        
        self.default_speaker = default_speaker
        self.default_emotion = default_emotion
        self.default_langauge = default_langauge

    def __len__(self):
        return len(self.ssml)
    
    def get_text(self, text, language_id=None):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, dictionary=self.cmudict, language=language_id))
            
        return text_norm
    
    def ssml_parse(self, ssml):# default style
        sentence_num = ssml[0]
        # 문장 앞 뒤 띄어쓰기 지우기
        txt = ssml[1].text.strip()
        content = ssml[1].find_all()
        speaker_id = self.default_speaker
        emotion_id = self.default_emotion
        langauge_id = self.default_langauge

        for c in content:
            # 문장별 파라미터: 1. emotion, 2. voice
            if c.name == 'voice':
                if removeS(c.attrs['name']) == 'etri_w':
                    speaker_id = self.spk_dict['etri_w']
                elif removeS(c.attrs['name']) == 'etri_m':
                    speaker_id = self.spk_dict['etri_m']
                else:
                    speaker_id = int(removeS(c.attrs['name']))
            elif c.name == 'emotion':
                emotion_id = self.emo_dict[removeS(c.attrs['class'][0])]
            # 문장 처리(현재 return 값): 1. say-as 처리, 2. sub 처리
            elif c.name == 'say-as' and self.use_lexicon:
                check = removeS(c.attrs['interpret-as'])
                strNum_list = re.findall('\d+', str(c.contents[0]))
                tmp = c.contents[0]

                for strNum in strNum_list:
                    tmpNum = ""
                    intNum = int(strNum)
                    if check == "한문-분리":
                        for s in strNum:
                            # 한글자씩 읽기 (0 == 공)
                            mandarin_num = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육",
                                            "7": "칠", "8": "팔", "9": "구"}
                            tmpNum += mandarin_num[s]
                    elif check == "한문":
                        # 숫자 한문 읽기
                        tmpNum = readNumM(intNum)
                    else:  # check == "한글"
                        # 100이상 한문 읽기 + 이하 한글 읽기
                        tmpNum = readNumM(intNum // 100 * 100) + readNumK(intNum % 100)

                    tmp = str(c.contents[0]).replace(str(strNum), str(tmpNum))

                english = re.sub('[^a-zA-Z]', '', str(c.contents[0]))
                if english != '':

                    for key, value in num_sub.items():
                        if key in english:
                            tmp.replace(english, value)

                txt = txt.replace(str(c.contents[0]), str(tmp))

            elif c.name == 'sub':
                tmp = removeS(c.attrs['alias'])
                txt = txt.replace(c.contents[0], tmp)

        # lexicon 처리를 마지막으로
        if self.use_lexicon:
            langauge_id = 1
            
            word_list = txt.split(' ')
            for k, word in enumerate(word_list):
                english = re.sub('[^a-zA-Z]', '', word)
                if english != '':
                    for key, value in lexicon.items():
                        if key.lower() == english.lower():
                            word_list[k] = word_list[k].replace(english, value)

                    for key, value in sub.items():
                        if key.lower() == english.lower():
                            word_list[k] = word_list[k].replace(english, value)

            txt = word_list[0]
            for word in word_list[1:]:
                txt += ' ' + word

            # 영어 단어 전체 교체
            txt = txt.lower()
            for key, value in english_word.items():
                txt = txt.replace(key, value)
                
        text_index = self.get_text(txt, langauge_id)
                
        return txt, sentence_num, text_index, speaker_id, emotion_id

    def __getitem__(self, index):
        return self.ssml_parse(self.ssml[index])


def collate_fn(data):
    n_batch = len(data)
    data.sort(key=lambda x: len(x['txt']), reverse=True)

    txt_len = torch.tensor([len(x['txt']) for x in data])
    max_txt_len = max(txt_len)

    origin_txt = []
    txt = torch.zeros(n_batch, max_txt_len).long()

    gender = torch.zeros(n_batch).long()
    age = torch.zeros(n_batch).long()
    emotion = torch.zeros(n_batch).long()

    attributes_which = torch.zeros(n_batch, max_txt_len).int()
    attributes_how = torch.zeros(n_batch, max_txt_len).long()

    emb = torch.zeros((n_batch, 256))

    filename = []

    for ii, item in enumerate(data):
        origin_txt.append(item['origin_txt'])
        txt[ii, :len(item['txt'])] = torch.tensor(item['txt']).long()

        attributes_which[ii, :len(item['txt'])] = torch.tensor(item['attributes'][0]).int()
        attributes_how[ii, :len(item['txt'])] = torch.tensor(item['attributes'][1]).long()

        gender[ii]  = item['style']['gender']
        age[ii]     = item['style']['age']
        emotion[ii] = item['style']['emotion']
        filename.append(item['filename'])

    return origin_txt, txt, txt_len, attributes_which, attributes_how, gender, age, emotion, emb, filename
