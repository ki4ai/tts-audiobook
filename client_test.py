import requests
import itertools
import argparse
import time

import sys
sys.path.append('waveglow/')
sys.path.append('tts/')

from tts.hparams import create_hparams


parser = argparse.ArgumentParser(description='Flask server script')
parser.add_argument('--port', type=int, default=8081, help='port number for api')
parser.add_argument('--text', type=str,
                    default='안녕하세요. 저는 감정을 담아 말하는 음성합성기입니다.',
                    help='Text for inference')
parser.add_argument('--spk', type=str,
                    default='20',
                    help='Text for inference')
parser.add_argument('--emo', type=str,
                    default='neutral',
                    help='Text for inference')

args = parser.parse_args()
s = requests.session()
ip_address = 'http://0.0.0.0:' + str(args.port)

hparams = create_hparams()

r = s.post(ip_address, data={'sentence': args.text, 'speaker': args.spk, 'emotion': args.emo, 'intensity': 1, 'result_folder': 'samples/'})