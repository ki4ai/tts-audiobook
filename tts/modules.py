# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs):
        out = inputs.transpose(1,2).unsqueeze(1)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
            
        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        
        _, out = self.gru(out)
        
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads))
        d_q = hp.ref_enc_gru_size # hp.token_embedding_size // 2
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=hp.token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)
        
        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        self.stl = STL(hp)
        
    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed.squeeze(1)
    
    
class GradReverse(torch.autograd.Function):
    '''
    input & output: x
    backward: -grad    
    '''
    scale = 1.0
    spk_grad_clip_thresh = 0.5
    
    @staticmethod
    def forward(ctx, x):
        y = x.view_as(x)
        return y
    
    def clip_grad_norm(grad_output):
        
        max_norm = GradReverse.spk_grad_clip_thresh
        param_norm = grad_output.norm(2)
        
        clip_coef = max_norm / (param_norm + 1e-6)
        if clip_coef < 1:
            grad_output.mul_(clip_coef)
        return grad_output
    
    @staticmethod
    def backward(ctx, grad_output):
        return GradReverse.scale * GradReverse.clip_grad_norm(grad_output).neg()
        # return GradReverse.scale * grad_output.clamp_(min=-GradReverse.spk_grad_clip_thresh, max=GradReverse.spk_grad_clip_thresh).neg()
    
    
class SpeakerAdversarial(nn.Module):
    '''
    input:
        embedded_text
    output:
        spk_adv_outputs
    '''
    def __init__(self, hp):
        super().__init__()
        self.W_speaker = nn.Linear(in_features=hp.encoder_embedding_dim, out_features=hp.speaker_hidden, bias=False)
        self.W_classifier = nn.Linear(in_features=hp.speaker_hidden, out_features=hp.n_speakers, bias=False)
        
        self.scale = hp.scale
        self.spk_grad_clip_thresh = hp.spk_grad_clip_thresh
        
    def grad_reverse(self, x):
        GradReverse.scale = self.scale
        GradReverse.spk_grad_clip_thresh = self.spk_grad_clip_thresh
        return GradReverse.apply(x)
        
    def forward(self, embedded_text):
        embedded_text_view = self.grad_reverse(embedded_text)
        speaker = F.relu(self.W_speaker(embedded_text_view))
        classifier = self.W_classifier(speaker)
        spk_adv_outputs = F.softmax(classifier, dim=1)
        
        return spk_adv_outputs
    