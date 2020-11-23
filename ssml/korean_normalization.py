
def txt_preprocessing(txt):
    word_list = txt.split(' ')
    ssml_list = txt.split(' ')

    for k, word in enumerate(word_list):
        strNum_list = re.findall('\d+', word)
        prev = list()
        if strNum_list:
            for key, value in sub.items():
                if key.lower() in word:
                    tmp_change = "<sub alias=\"" + value + "\">" + key + "</sub>"
                    ssml_list[k] = ssml_list[k].replace(key, tmp_change)

        for strNum in strNum_list:
            if strNum in prev:
                continue
            pos = word.index(strNum)
            if len(word) > pos + len(strNum):
                if strNum[0] == "0":
                    check = "한문-분리"
                elif word[pos + len(strNum)] in korean_end_word:
                    check = "한글"
                else:
                    check = "한문"
            else:
                if strNum[0] == "0":
                    check = "한문-분리"
                else:
                    check = "한문"

            tmp_change = "<say-as interpret-as=\"" + check + "\">" + strNum + "</say-as>"
            ssml_list[k] = ssml_list[k].replace(strNum, tmp_change)
            prev.append(strNum)

    result = ssml_list[0]
    for ssml in ssml_list[1:]:
        result += ' ' + ssml
    return result

def emotionTTS_preprocessing(ssml, speaker, emotion):
    # print("Emotion과 Prosody에 대한 처리를 해주는 부분입니다.")
    # Emotion, Prosody을 처리해 줍니다.
    cur = "\t\t\t\t<voice name=\"" + speaker + "\">" + "<emotion class=\"" + emotion + "\">"
    cur = cur + ssml
    cur = cur + "</emotion></voice>\n"
    return cur


def ssml_preprocessing(whole_ssml):
    # print("ssml 형태를 잡아주는 부분입니다.")
    # SSML 전체 베이스를 잡아주고, 문단 별 처리가 필요한 부분이 구현될 부분입니다.
    cur = "<speak><lexicon dict=\"국립국어원 표준국어대사전.csv\" xml:id=“국립국어원 표준국어대사전\"/>\n\t<lookup ref=“국립국어원 표준국어대사전”>\n\t\t<p>\n"
    for ssml in whole_ssml:
        cur += "\t\t\t<s>\n"
        cur += ssml
        cur += "\t\t\t</s>\n"
    cur += "\t\t</p>\n\t</lookup>\n</speak>\n"
    return cur
