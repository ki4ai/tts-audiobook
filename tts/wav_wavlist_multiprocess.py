from multiprocessing import Pool
import librosa
import os
import re

def checkLength(wavfile):
    input_dir = '/data3/sejikpark/data/data_22050/'
    cur_wav = '/data3/sejikpark/data/data_22050/' + wavfile + '.wav'
    # cur_fileList = ['02', '06', '08', '10', '14']
    # cur_fileList = ['02']
    cur_fileList = ['03_1_006']
    
    if wavfile[0:8] in cur_fileList:
    # if wavfile[0:2] not in cur_fileList: # and wavfile[0:8] != '05_1_019':
        y, sr = librosa.load(cur_wav)
        wav_duration = librosa.get_duration(y=y, sr=22050)
        if wav_duration < 10:
            return True
    return False


if __name__ == '__main__':
    input_dir = '/data3/sejikpark/data/data_22050/'
    result_dir = '/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/filelists/'
    result_file = 'ETRI_w_ten_second.txt'
    result_file = result_dir + result_file
    
    file_list = os.listdir(input_dir)
    wav_list = [file.replace('.wav', '') for file in file_list if file.endswith(".wav")]
    txt_list = [file.replace('.txt', '') for file in file_list if file.endswith(".txt")]
    
    assert len(set(wav_list) - set(txt_list)) == 0
    assert len(set(txt_list) - set(wav_list)) == 0
    
    with Pool(32) as p:
        updated_list = p.map(checkLength, wav_list)
    
    # updated_list = [True] * len(wav_list)
    
    wf = open(result_file, 'w', encoding='utf-8')
    for i, updated in enumerate(updated_list):
        if updated:
            cur_input = wav_list[i]
            cur_wav = input_dir + cur_input + '.wav'
            cur_txt = input_dir + cur_input + '.txt'
            
            try:
                with open(cur_txt, 'r', encoding='utf-8') as rf:
                    txt = rf.read()
            except:
                with open(cur_txt, 'r', encoding='euc-kr') as rf:
                    txt = rf.read()
            txt = txt.replace('\n', '')
            txt = txt.replace('[[lipsmack]]', '[[]]')
            txt = txt.replace('[[background]]', '[[]]')
            
            cur = cur_wav + '|' + txt + '\n'
            wf.write(cur)
    wf.close()
