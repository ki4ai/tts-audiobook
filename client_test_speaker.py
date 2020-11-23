import requests
import itertools
import argparse
import time
import random

import sys
sys.path.append('waveglow/')
sys.path.append('tts/')

from tts.hparams import create_hparams


parser = argparse.ArgumentParser(description='Flask server script')
parser.add_argument('--port', type=int, default=8081, help='port number for api')
parser.add_argument('--select_num', type=int, default=50)
args = parser.parse_args()
s = requests.session()
ip_address = 'http://0.0.0.0:' + str(args.port)

hparams = create_hparams()

speaker_list = ['690', '698', '703', '14', '3', '8', '691', '2', '697', '13']
select_num = args.select_num

with open('sentences/KCTTS_selected.txt', encoding='utf-8') as f:
    sentences = [line.strip() for line in f]
sentences = list(set(sentences))
sentences = random.sample(sentences, select_num)

input_list  = list()
for sen in sentences:
    for spk in speaker_list:
        input_list.append((sen, spk, 'neutral'))

with open('./M2.4/tts_time_result.txt', 'w') as f:
    f.write('')
for cur in input_list:
    r = s.post(ip_address, data={'sentence': cur[0], 'speaker':'{}'.format(cur[1]), 'emotion': '{}'.format(cur[2]), 'intensity': 1, 'result_folder': 'speaker/'})
with open('./M2.4/tts_time_result.txt', 'r') as f:
    generate_time = [line.strip().split('|') for line in f]
    
total_generate_time = 0
total_wave_length = 0
total_sample_length = 0
for generate_time, wave_length, sample_length in generate_time:
    total_generate_time += float(generate_time)
    total_wave_length += float(wave_length)
    total_sample_length += float(sample_length)
if total_generate_time:
    total_generation_speed = total_wave_length / total_generate_time
    total_sample_speed = total_sample_length / total_generate_time
    print('{:.2f}s/s, {:.2f}samples/s: it takes {:.2f}s for {:.2f}s wave. sampling rate {}, number of samples {}'.format(total_generation_speed, total_sample_speed, total_generate_time, total_wave_length, hparams.sampling_rate, total_sample_length))
else:
    print('All cached genration')