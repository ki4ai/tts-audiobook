import argparse
import time
import sys
import os
from os.path import join
from flask import Flask, jsonify, request, Response, make_response, send_file
import soundfile as sf
import librosa
import numpy as np

import torch

sys.path.append('waveglow/')
sys.path.append('tts/')

from tts.model import load_model
from tts.hparams import create_hparams
from tts.text import cmudict, text_to_sequence
from waveglow.denoiser import Denoiser
from VocGAN.model.generator import ModifiedGenerator
from VocGAN.utils.hparams import HParam, load_hparam_str
from VocGAN.denoiser import Denoiser as VocGAN_Denoiser 
from data_utils import collate_fn
from config_dict import reverse_emo_dict


bb = time.time()
app = Flask(__name__)
parser = argparse.ArgumentParser(description='Flask server script')

# data load
parser.add_argument('--port', type=int, default=8081, help='port number for api')
parser.add_argument('--init_cmudict', type=str,
                    default='./tts/data/cmu_dictionary')
parser.add_argument('--init_VocGAN', type=str,
                    default='./models/VocGAN_0364.pt')
parser.add_argument('--config_VocGAN', type=str,
                    default='./VocGAN/config/default.yaml')
args = parser.parse_args()

tts_checkpoint_path = 'models/checkpoint_79500'
waveglow_checkpoint_path = 'models/waveglow_256channels_ljs_v3.pt'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="7"

hparams = create_hparams()
hparams.max_decoder_steps=2000
tts = load_model(hparams).cuda().eval()
tts.load_state_dict(torch.load(tts_checkpoint_path)['state_dict'])
waveglow = torch.load(waveglow_checkpoint_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

hp = HParam(args.config_VocGAN)
checkpoint = torch.load(args.init_VocGAN)
VocGAN = ModifiedGenerator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
VocGAN.load_state_dict(checkpoint['model_g'])
VocGAN.eval(inference=True)
VocGAN_denoiser = VocGAN_Denoiser(VocGAN).cuda().eval()

arpabet_dict = cmudict.CMUDict(args.init_cmudict)


@app.route('/', methods=['POST'])
def calc():
    print('request.form : {}'.format(request.form))
    dataset_filepath = request.form['dataset'] if 'dataset' in request.form.keys() else None
    
    if dataset_filepath is None:
        return 'failed'
    
    dataset = torch.load(dataset_filepath)['data_loader']
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    result_sentence_num = list()
    result_txt = list()
    result_audio = list()
    for batch in loader:
        with torch.no_grad():
            txt, sentence_num, text_padded, speaker_ids, emotion_ids = batch
            inputs = (text_padded.cuda(), speaker_ids.cuda(), emotion_ids.cuda())
            
            if reverse_emo_dict[emotion_ids.item()] == 'neutral':
                gst_mel = torch.load('./gst_mel/' + str(speaker_ids.item()) + '_' + reverse_emo_dict[emotion_ids.item()] + '.pt').cuda()
            else:
                gst_mel = torch.load('./gst_mel/' + '20' + '_' + reverse_emo_dict[emotion_ids.item()] + '.pt').cuda()
            
            _, mel_outputs_postnet, _, _ = tts.inference(inputs, gst_mel=gst_mel, logging=False)
            # audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.7), 0.1)
            audio = VocGAN_denoiser(VocGAN.inference(mel_outputs_postnet).squeeze(0), 0.01)[:, 0]
        
        result_sentence_num.extend(sentence_num)
        result_txt.extend(txt)
        result_audio.append(torch.cat([audio, torch.zeros(audio.size(0), 200).cuda()], dim=1))
    
    result_txt = [txt for _, txt in sorted(zip(result_sentence_num, result_txt), key=lambda x: x[0], reverse=True)]
    result_audio = [audio for _, audio in sorted(zip(result_sentence_num, result_audio), key=lambda x: x[0], reverse=True)]
    
    if len(result_txt) != 1:
        txt = ' '.join(result_txt)
    else:
        txt = result_txt[0]
    wave = torch.cat(result_audio, dim=1)
    
    save_path = './M2.4/'
    outpath = save_path + 'SAVETO_' + txt + '.wav'
    os.makedirs(save_path, exist_ok=True)
                      
    wave = wave.squeeze().cpu().numpy()
    wave = librosa.core.resample(wave, hparams.sampling_rate, 44100)
    wave = np.stack((wave, wave))
    maxv = 2 ** (16 - 1)
    wave /= max(abs(wave.max()), abs(wave.min()))
    wave = (wave * maxv * 0.95).astype(np.int16)

    # write to file
    sf.write(outpath, wave.T, 44100, format='WAV', endian='LITTLE', subtype='PCM_16')

    return 'success'

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - bb))
    app.run(host='0.0.0.0', port=args.port)
