import random
import os
import re
import numpy as np
import librosa

import torch
import torch.utils.data
import torch.nn.functional as F

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.audiopaths_and_text = self.audiopaths_and_text
        
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_data(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        cur_audiopath = audiopath.split('_')
        try:
            language_id, speaker_id, emotion_id = self.get_style_id(cur_audiopath)
        except:
            f = open("/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/log.txt", 'a')
            f.write(filepath)
            f.write('|file_name\n')
            f.close()
            language_id = 0
            speaker_id = torch.IntTensor(0)
            emotion_id = torch.IntTensor(0)
        
        mel, no_sound = self.get_mel(audiopath)
        if no_sound:
            text = ' '
        text = self.get_text(text, language_id)
        return (text, mel, speaker_id, emotion_id)

    def get_style_id(self, cur_audiopath):
        language_id = int(cur_audiopath[-6])
        speaker_id = torch.IntTensor([int(cur_audiopath[-5])])
        if int(cur_audiopath[-3]) and int(cur_audiopath[-3]) not in [7, 9]:
            emotion_id = torch.IntTensor([(int(cur_audiopath[-5])%30)*8+int(cur_audiopath[-3])])
        else:
            emotion_id = torch.IntTensor([0])
        
        return (language_id, speaker_id, emotion_id)

    def get_mel(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        
        no_sound = False
        if audio_norm.shape[1] < self.filter_length // 2 + 1:
            audio_norm = torch.zeros((1, self.filter_lengths//2+1))
            no_sound = True
            f = open("/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/log.txt", 'a')
            f.write(filepath)
            f.write('|no_sound\n')
            f.close()
            
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec, no_sound

    def get_text(self, text, language_id=None):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, dictionary=self.cmudict, language=language_id))
            
        return text_norm

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        
        speaker_ids = torch.LongTensor(len(batch))
        emotion_ids = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            emotion_ids[i] = batch[ids_sorted_decreasing[i]][3]
            
        return (text_padded, input_lengths, mel_padded, gate_padded, output_lengths,
                speaker_ids, emotion_ids)
    
if __name__ == "__main__":
    
    cmudict = cmudict.CMUDict('/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/data/cmu_dictionary')
    
    def get_text(text, text_cleaners, cmudict, p_arpabet):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, dictionary=self.cmudict, language=language_id))
        return text_norm

    get_text('english 와 한국어를 테스트 중입니다.', ['korean_english_cleaners'], cmudict, 1.0)
    