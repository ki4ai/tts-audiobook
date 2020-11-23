
from ssml.preprocessing import txt_preprocessing, emotionTTS_preprocessing, ssml_preprocessing
from ssml.postprocessing import Dataset as ssml_postprocessing
from bs4 import BeautifulSoup


def sentence_to_ssml(sentence, korean_normalization):
    txt = sentence.replace('"', '')
    txt_split = txt.replace('!', '.').replace('?', '.').split('.')
    txt_split_post = list()
    for t in txt_split:
        if t != '':
            try:
                txt_split_post.append(t + txt[txt.find(t) + len(t)])
            except:
                txt_split_post.append(t)

    whole_ssml = list()
    for t in txt_split_post:
        if korean_normalization == 'True':
            ssml = txt_preprocessing(t)
        else:
            ssml = t
        # 추후 기초 화자 설정 후 수정 예정
        ssml = emotionTTS_preprocessing(ssml, 'etri_w', 'neutral')
        whole_ssml.append(ssml)

    # make & print SSML
    ssml = ssml_preprocessing(whole_ssml, korean_normalization)
    return ssml


def ssml_to_tts_input(ssml):
    ssml = ssml.replace('\\t', '')
    bs = BeautifulSoup(ssml, 'html.parser')

    # 사전 정의에 따른 읽어오기
    dictionary = bs.find('lexicon')
    # lookup을 통해 사용여부 확인
    use_lexicon = 0
    if bs.find('lookup'):
        use_lexicon = 1
    # 데이터 준비: 문장 처리
    sentences = bs.find_all('s')
    
    data_loader = ssml_postprocessing(sentences, use_lexicon)
    
    return data_loader