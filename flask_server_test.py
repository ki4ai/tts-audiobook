import argparse
import time
import sys
import os
from os.path import join
from flask import Flask, jsonify, request, Response, make_response, send_file
import soundfile as sf
import librosa
import numpy as np
import copy

import torch

# sys.path.append('waveglow/')
sys.path.append('tts/')

from tts.model import load_model
from tts.hparams import create_hparams
from tts.text import cmudict, text_to_sequence
# from waveglow.denoiser import Denoiser
from VocGAN.model.generator import ModifiedGenerator
from VocGAN.utils.hparams import HParam, load_hparam_str
from VocGAN.denoiser import Denoiser as VocGAN_Denoiser 
from data_utils import collate_fn
from config_dict import emo_dict, spk_dict


bb = time.time()
app = Flask(__name__)
parser = argparse.ArgumentParser(description='Flask server script')

# data load
parser.add_argument('--port', type=int, default=8081, help='port number for api')
parser.add_argument('--init_cmudict', type=str,
                    default='./tts/data/cmu_dictionary')
parser.add_argument('--init_tts', type=str,
                    default='./models/checkpoint_79500')
parser.add_argument('--init_zerogst_tts', type=str,
                    default='./models/checkpoint_4000')
parser.add_argument('--init_VocGAN', type=str,
                    default='./models/VocGAN_0364.pt')
parser.add_argument('--config_VocGAN', type=str,
                    default='./VocGAN/config/default.yaml')
parser.add_argument('--use_GST',
                    default=True,
                    action='store_true')
parser.add_argument('--use_emotional_GST',
                    default=True,
                    action='store_true')
parser.add_argument('--use_individual_emotion',
                    default=False,
                    action='store_true')
parser.add_argument('--enable_gpus', type=str,
                        default='0,1,2,3,4,5,6,7',
                        help='number of gpus')

args = parser.parse_args()

use_GST = args.use_GST
if use_GST:
    tts_checkpoint_path = args.init_tts
else:
    tts_checkpoint_path = args.init_zerogst_tts
# waveglow_checkpoint_path = 'models/waveglow_256channels_ljs_v3.pt'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.enable_gpus

hparams = create_hparams()
hparams.max_decoder_steps=2000
tts = load_model(hparams).cuda().eval()
tts.load_state_dict(torch.load(tts_checkpoint_path)['state_dict'])
# waveglow = torch.load(waveglow_checkpoint_path)['model'].cuda().eval()
# denoiser = Denoiser(waveglow).cuda().eval()

hp = HParam(args.config_VocGAN)
checkpoint = torch.load(args.init_VocGAN)
VocGAN = ModifiedGenerator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
VocGAN.load_state_dict(checkpoint['model_g'])
VocGAN.eval(inference=True)
VocGAN_denoiser = VocGAN_Denoiser(VocGAN).cuda().eval()

arpabet_dict = cmudict.CMUDict(args.init_cmudict)
use_emotional_GST = args.use_emotional_GST
use_individual_emotion = args.use_individual_emotion

@app.route('/', methods=['POST'])
def calc():
    print('request.form : {}'.format(request.form))
    sentence = request.form['sentence'] if 'sentence' in request.form.keys() else None
    speaker = request.form['speaker'] if 'speaker' in request.form.keys() else 'etri_w'
    emotion = request.form['emotion'] if 'emotion' in request.form.keys() else 'neutral'
    result_folder = request.form['result_folder'] if 'result_folder' in request.form.keys() else 'samples/'
    result_filename = request.form['result_filename'] if 'result_filename' in request.form.keys() else None
    
    assert sentence is not None
    
    save_path = './M2.4/generated/'
    save_path += result_folder
    if not os.path.isdir(save_path):
            os.makedirs(save_path)
            os.chmod(save_path, 0o775)
    
    if len(sentence) > 80:
        file_name = copy.deepcopy(sentence[:80])
    else:
        file_name = copy.deepcopy(sentence)
    
    if result_filename is not None:
        outpath1 = save_path + result_filename
    else:
        outpath1 = save_path + '%s_%s_%s.wav' % (speaker, emotion, file_name)
    
    if os.path.exists(outpath1):
        return 'success'
    
    sentence = ' ' + sentence
    sentence = sentence.replace('?', '.')
    sentence = sentence.replace('!', '.')
    if sentence[-1] != '.':
        sentence += '.'
        
    text_padded = torch.LongTensor(text_to_sequence(sentence, hparams.text_cleaners, arpabet_dict, language=1))[None, :].cuda()
    
    speaker_ids = torch.tensor([spk_dict[speaker]]).cuda()
    if use_individual_emotion:
        if emo_dict[emotion] != 0:
            tmp_ids = emo_dict[emotion] - 20*8 + (spk_dict[speaker]%30)*8
            emotion_ids = torch.tensor([tmp_ids]).cuda()
        else:
            emotion_ids = torch.tensor([emo_dict[emotion]]).cuda()
    else:
        emotion_ids = torch.tensor([emo_dict[emotion]]).cuda()
    
    if use_GST:
        if use_emotional_GST:
            if speaker_ids.item() in [12, 13, 19, 20]:
                gst_mel = torch.load('./gst_mel/' + str(speaker_ids.item()) + '_' + emotion + '.pt').cuda()
            else:
                gst_mel = torch.load('./gst_mel/' + str(speaker_ids.item()) + '_' + 'neutral' + '.pt').cuda()
        else:
            gst_mel = torch.load('./gst_mel/' + str(speaker_ids.item()) + '_' + 'neutral' + '.pt').cuda()
    else:
        gst_mel = None
    inputs = (text_padded.cuda(), speaker_ids.cuda(), emotion_ids.cuda())
        
    start = time.time()
    
    tts.decoder.max_decoder_steps = int(text_padded.shape[-1] * 10) + 500
    with torch.no_grad():
        _, mel_outputs_postnet, _, _ = tts.inference(inputs, gst_mel=gst_mel, logging=False)
        # wave = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.7), 0.1)
        wave = VocGAN_denoiser(VocGAN.inference(mel_outputs_postnet).squeeze(0), 0.01)[:, 0]

    generate_time = time.time() - start
    wave = wave.squeeze()
    wave = wave[:-(hp.audio.hop_length*10)]
    sample_length = len(wave)
    wave_length = sample_length / hparams.sampling_rate
    generation_speed = wave_length / generate_time
    sample_speed = sample_length / generate_time
    print('{:.2f}s/s, {:.2f}samples/s: it takes {:.2f}s for {:.2f}s wave. sampling rate {}, number of samples {}'.format(generation_speed, sample_speed, generate_time, wave_length, hparams.sampling_rate, sample_length))
    
    with open('./M2.4/tts_time_result.txt', 'a') as f:
        f.write(str(generate_time) + '|' + str(wave_length) + '|' + str(sample_length) + '\n')
    
    # amplifying
    wave = wave.squeeze().cpu().numpy()
    wave = librosa.core.resample(wave, hparams.sampling_rate, 44100)
    wave = np.stack((wave, wave))
    maxv = 2 ** (16 - 1)
    wave /= max(abs(wave.max()), abs(wave.min()))
    wave = (wave * maxv * 0.95).astype(np.int16)

    sf.write(outpath1, wave.T, 44100, format='WAV', endian='LITTLE', subtype='PCM_16')

    return 'success'

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - bb))
    app.run(host='0.0.0.0', port=args.port)
