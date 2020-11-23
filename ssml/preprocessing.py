
import re
import pickle as pkl

korean_end_word = ['개', '대', '돈', '마리', '벌', '살', '손', '자루', '죽', '채', '켤레', '쾌', '시', '과', '포기', '번째', '살', '척']
num_sub_pickle = 'ssml/dictionary/num_sub.pickle'

with open(num_sub_pickle, 'rb') as handle:
    num_sub = pkl.load(handle)

def txt_preprocessing(txt):
    word_list = txt.split(' ')
    ssml_list = txt.split(' ')

    for k, word in enumerate(word_list):
        strNum_list = re.findall('\d+', word)
        if strNum_list:
            # num_sub 처리
            for key, value in num_sub.items():
                if key in word:
                    tmp_change = "<sub alias=\"" + value + "\">" + key + "</sub>"
                    ssml_list[k] = ssml_list[k].replace(key, tmp_change)
                    not_checked = 0
            # say-as 발음 교체
            seperated_num = 0
            if '-' in word:
                seperated_num = 1

            dot_index = 0
            if '.' in word:
                if word[-1] != '.':
                    tmp_change = "<sub alias=\"" + '점' + "\">" + '.' + "</sub>"
                    ssml_list[k] = ssml_list[k].replace('.', tmp_change)
                    dot_index =  word.index('.')
            if ',' in word:
                if word[-1] != ',':
                    tmp_change = "<sub alias=\"" + '' + "\">" + ',' + "</sub>"
                    ssml_list[k] = ssml_list[k].replace(',', tmp_change)
            
            prev = -1
            for strNum in strNum_list:
                pos = word.index(strNum)
                if prev == pos:  # 약식 값 중복 처리
                    continue
                wList = [word[0:pos], word[pos: pos + len(strNum)], word[pos + len(strNum):]]
                wList = [w for w in wList if not w == '']
                check = ""
                # 처음이 0으로 시작하면 한문-분리
                if strNum[0] == '0':
                    check = "한문-분리"
                    if word_list[k-1] == '카드번호는':
                        tmp_change = strNum + "<sub alias=\"" + '다시' + "\">" + '-' + "</sub>"
                        ssml_list[k] = ssml_list[k].replace(strNum+'-', tmp_change)
                    else:
                        tmp_change = strNum + "<sub alias=\"" + '에' + "\">" + '-' + "</sub>"
                        ssml_list[k] = ssml_list[k].replace(strNum+'-', tmp_change)
                else:
                    for i, w in enumerate(wList):
                        # 숫자 뒤에 붙는 것이 없을 때, 한문
                        if len(wList) == (i + 1):
                            if k > 1:
                                if word_list[k - 1][0] == '-':
                                    check = "한문-분리"
                                    break
                            if k + 1 < len(word_list):
                                if word_list[k + 1][0] == '-':
                                    check = "한문-분리"
                                elif word_list[k + 1][0] in korean_end_word:
                                    check = "한글"
                                else:
                                    check = "한문"
                            else:
                                check = "한문"
                            break
                        elif w == strNum:
                            # 숫자 뒤에 붙는 것에 따라 한글, 한문 선택
                            if wList[i + 1][0] in korean_end_word:
                                check = "한글"
                            else:
                                check = "한문"
                            break

                tmp_change = "<say-as interpret-as=\"" + check + "\">" + strNum + "</say-as>"
                ssml_list[k] = ssml_list[k].replace(strNum, tmp_change)

    result = ssml_list[0]
    for ssml in ssml_list[1:]:
        result += ' ' + ssml
    return result


def emotionTTS_preprocessing(ssml, speaker, emotion):
    # print("Emotion과 Prosody에 대한 처리를 해주는 부분입니다.")
    # Emotion, Prosody을 처리해 줍니다.
    cur = "                <voice name=\"" + speaker + "\">" + "<emotion class=\"" + emotion + "\">"
    cur = cur + ssml
    cur = cur + "</emotion></voice>\n"
    return cur


def ssml_preprocessing(whole_ssml, korean_normalization):
    # print("ssml 형태를 잡아주는 부분입니다.")
    # SSML 전체 베이스를 잡아주고, 문단 별 처리가 필요한 부분이 구현될 부분입니다
    if korean_normalization == 'True':
        cur = "<speak><lexicon dict=\"국립국어원 표준국어대사전.csv\" xml:id=“국립국어원 표준국어대사전\"/>\n    <lookup ref=“국립국어원 표준국어대사전”>\n        <p>\n"
        for ssml in whole_ssml:
            cur += "            <s>\n"
            cur += ssml
            cur += "            </s>\n"
        cur += "        </p>\n    </lookup>\n</speak>\n"
    else:
        cur = "<speak>\n    <p>\n"
        for ssml in whole_ssml:
            cur += "            <s>\n"
            cur += ssml
            cur += "            </s>\n"
        cur += "        </p>\n</speak>\n"

    return cur
