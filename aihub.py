from multiprocessing import Pool
from bs4 import BeautifulSoup
import random
import librosa
import shutil
import os
import re
import json


if __name__ == '__main__':
    text_emotion = list()
    
    root_name = 'AI-Hub-multimodal'
    middle_dir = os.listdir(root_name)
    for md in middle_dir:
        dir_name = os.path.join(root_name, md)
        filenames = os.listdir(dir_name)
        for filename in filenames:
            full_filename = os.path.join(dir_name, filename, filename + '.json')
    
            with open(full_filename) as json_file:
                json_data = json.load(json_file)
           
            for k1 in json_data['data'].keys():
                for k2 in json_data['data'][k1].keys():
                    if 'text' in json_data['data'][k1][k2].keys():
                        text_emotion.append((full_filename,
                                             json_data['data'][k1][k2]['text']['script'],
                                             json_data['data'][k1][k2]['emotion']['text']['emotion']))
    
    try:
        for emotion in ['happy', 'surprise', 'angry', 'sad', 'dislike', 'fear', 'contempt', 'neutral']:# set([t[2] for t in text_emotion]):
            result = set([t[0] + '|' + t[1] + '\n' for t in text_emotion if t[2] == emotion])
            wf = open(emotion + '.txt', 'w', encoding='utf-8')
            for r in result: wf.write(r)
            wf.close()
            print(emotion)
    except:
        import pdb; pdb.set_trace()
    