from math import sqrt
import numpy as np
from numpy import finfo

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from modules import GST, SpeakerAdversarial

# torch.autograd.set_detect_anomaly(True)
drop_rate = 0.5


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    return model


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        
        self.previous_attention = None

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)

            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)

            attention_weights = F.softmax(alignment, dim=1)
            
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
    
    def inference(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)

            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)

            attention_weights = list()
            for batch_i, (align, prev) in enumerate(zip(alignment, self.previous_attention)):
                nxt = min(alignment.size(1), prev+3)
                prev = min(alignment.size(1)-2, prev)
                attention_weight = F.pad(F.softmax(align[prev:nxt], dim=0), (prev, alignment.size(1)-nxt))
                attention_weights.append(attention_weight)
                self.previous_attention[batch_i] = torch.argmax(attention_weight, dim=0)
        
        attention_weights = torch.stack(attention_weights, dim=0)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for i, linear in enumerate(self.layers):
            x = F.dropout(F.relu(linear(x)), p=drop_rate, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), drop_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), drop_rate, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

    
class Contents_Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Contents_Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.n_mel_channels * hparams.n_frames_per_step,
                         hparams.n_mel_channels * hparams.n_frames_per_step,
                         kernel_size=hparams.encoder_kernel_size, stride=2, padding=1,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.n_mel_channels * hparams.n_frames_per_step))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.kernel_size = hparams.encoder_kernel_size
        self.stride_size = 2
        self.padding = 1
        self.n_convs = hparams.encoder_n_convolutions
        
        self.lstm = nn.LSTM(hparams.n_mel_channels * hparams.n_frames_per_step,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = self.calculate_channels(input_lengths)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs, torch.tensor(input_lengths).cuda()

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
    
    def calculate_channels(self, L):
        for _ in range(self.n_convs): # 407 203 101 50
            L = (L - self.kernel_size + 2 * self.padding) // self.stride_size + 1
        return L
    

class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.style_embedding_dim = hparams.speaker_embedding_dim + hparams.emotion_embedding_dim + hparams.token_embedding_size
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + self.style_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.p_dropout_teacher = hparams.p_dropout_teacher
        
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)
        
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')
        
        self.attention_hidden_init_layer = Prenet(hparams.token_embedding_size, [8, hparams.attention_rnn_dim])
        self.attention_cell_init_layer = Prenet(hparams.token_embedding_size, [8, hparams.attention_rnn_dim])
        self.decoder_hidden_init_layer = Prenet(hparams.token_embedding_size, [8, hparams.attention_rnn_dim])
        self.decoder_cell_init_layer = Prenet(hparams.token_embedding_size, [8, hparams.attention_rnn_dim]) 

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input - 13.815

    def initialize_decoder_states(self, memory, embedded_gst, mask, init_gst=True):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        # initialize by memory
        if init_gst == True:
            self.attention_hidden = self.attention_hidden_init_layer(embedded_gst)
            self.attention_cell = self.attention_cell_init_layer(embedded_gst)
            self.decoder_hidden = self.decoder_hidden_init_layer(embedded_gst)
            self.decoder_cell = self.decoder_cell_init_layer(embedded_gst)
        else:
            self.attention_hidden = Variable(memory.data.new(
                B, self.attention_rnn_dim).zero_())
            self.attention_cell = Variable(memory.data.new(
                B, self.attention_rnn_dim).zero_())
            self.decoder_hidden = Variable(memory.data.new(
                B, self.decoder_rnn_dim).zero_())
            self.decoder_cell = Variable(memory.data.new(
                B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

        self.attention_layer.previous_attention = torch.zeros(B).long()

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) > 1:
            gate_outputs = gate_outputs.transpose(0, 1)
        else:
            gate_outputs = gate_outputs[None]
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_weights=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """        
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, attention_weights)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def inference_decode(self, decoder_input, attention_weights=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """        
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer.inference(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, attention_weights)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, embedded_gst, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, embedded_gst, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if len(mel_outputs) == 0 or np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing:
                decoder_input = decoder_inputs[len(mel_outputs)]
            else:
                decoder_input = self.prenet(mel_outputs[-1])
            
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, embedded_gst, logging=True, init_gst=True):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, embedded_gst, mask=None, init_gst=init_gst)

        end_trigger = 0
        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            
            mel_output, gate_output, alignment = self.inference_decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            
            if not logging:
                if end_trigger >= 20:
                    break
                elif (torch.sigmoid(gate_output.data) > self.gate_threshold) and end_trigger >= 10:
                    break
                elif len(alignments) > 3:
                    curr_att_idx = np.argmax(alignment[0].cpu().numpy())
                    if end_trigger:
                        end_trigger += 1
                    else:
                        if curr_att_idx >= memory.size(1)-1:
                            end_trigger = 1
            if len(mel_outputs) == self.max_decoder_steps:
                if not logging:
                    print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        
        self.speaker_embedding = nn.Embedding(
            hparams.n_speakers, hparams.speaker_embedding_dim)
        self.emotion_embedding = nn.Embedding(
            hparams.n_emotions, hparams.emotion_embedding_dim)
        
        self.speaker_layer = LinearNorm(hparams.speaker_embedding_dim,
                                      hparams.speaker_embedding_dim,
                                      bias=False)
        self.emotion_layer = LinearNorm(hparams.emotion_embedding_dim,
                                      hparams.emotion_embedding_dim,
                                      bias=False)
        
        self.p_gst_using = hparams.p_gst_using
        self.token_embedding_size = hparams.token_embedding_size
        self.gst = GST(hparams)
        
        self.freeze_style_encoder = hparams.freeze_style_encoder
        self.not_freeze_speaker = hparams.not_freeze_speaker
        # self.not_freeze_gst_speaker = hparams.not_freeze_gst_speaker
        
        self.contents_encoder = Contents_Encoder(hparams)
        
        self.VC_speaker = hparams.VC_speaker
        self.p_VC_ratio = hparams.p_VC_ratio

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids, emotion_ids = batch
        
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        
        speaker_ids = to_gpu(speaker_ids.data).long()
        emotion_ids = to_gpu(emotion_ids.data).long()
        
        return ((text_padded, input_lengths, mel_padded, output_lengths,
                 speaker_ids, emotion_ids),
                (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        inputs, input_lengths_txt, targets, output_lengths, \
            speaker_ids, emotion_ids = inputs
        input_lengths_txt, output_lengths = input_lengths_txt.data, output_lengths.data
        
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        embedded_text_txt = self.encoder(embedded_inputs, input_lengths_txt)
        embedded_text_mel, input_lengths_mel = self.contents_encoder(targets, output_lengths)
        
        # random to speaker index
        if self.VC_speaker:
            text_speaker = torch.clone(speaker_ids).zero_().bool()
            for ts in self.VC_speaker:
                text_speaker += (speaker_ids == ts)
            text_speaker += F.dropout(torch.arange(speaker_ids.shape[0]).float(), self.p_VC_ratio).bool().cuda()
        
        if True in text_speaker:
            max_txt_len = input_lengths_txt[~text_speaker].max()
            embedded_text_txt = embedded_text_txt[:, :max_txt_len, :]
            max_mel_len = input_lengths_mel[text_speaker].max()
            embedded_text_mel = embedded_text_mel[:, :max_mel_len, :]
            max_len = max(max_txt_len, max_mel_len)
            if input_lengths_txt.max() > max_len:
                embedded_text_txt = embedded_text_txt[:, :max_len, :]
            if input_lengths_mel.max() > max_len:
                embedded_text_mel = embedded_text_mel[:, :max_len, :]
            VC_padding = True
            if max_len != max_txt_len:
                VC_padding = False
            embedded_text = list()
            input_lengths = list()
            for i, ts in enumerate(text_speaker):
                if ts: # VC
                    if VC_padding:
                        embedded_text.append(F.pad(embedded_text_mel[i], (0, 0, 0, max_len-max_mel_len)))
                    else:
                        embedded_text.append(embedded_text_mel[i])
                    input_lengths.append(input_lengths_mel[i])
                else:
                    if not VC_padding:
                        embedded_text.append(F.pad(embedded_text_txt[i], (0, 0, 0, max_len-max_txt_len)))
                    else:
                        embedded_text.append(embedded_text_txt[i])
                    input_lengths.append(input_lengths_txt[i])
            embedded_text = torch.stack(embedded_text)
            input_lengths = torch.stack(input_lengths)
        else:
            embedded_text = embedded_text_txt
            input_lengths = input_lengths_txt

        embedded_speaker = self.speaker_embedding(speaker_ids)
        embedded_speaker = self.speaker_layer(embedded_speaker)
        embedded_emotion = self.emotion_embedding(emotion_ids)
        embedded_emotion = self.emotion_layer(embedded_emotion)
        
        embedded_gst = self.gst(targets)
        
        embedded_styles = torch.cat((embedded_speaker, embedded_emotion, embedded_gst), dim=1)[:,None].repeat(1, embedded_text.size(1), 1)
        
        if self.freeze_style_encoder:
            if not self.VC_speaker:
                text_speaker = torch.clone(speaker_ids).zero_().bool()
            for ts in self.not_freeze_speaker:
                text_speaker += (speaker_ids == ts)
            no_grad_embedded_text = torch.clone(embedded_text[~text_speaker]).detach()
            embedded_text[~text_speaker] = no_grad_embedded_text
        
        encoder_outputs = torch.cat(
            (embedded_text, embedded_styles), dim=2)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, embedded_gst, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, gst_token=None, gst_mel=None, logging=True, gst_intensity=None, question=None, gst_embedded=None, init_gst=True):
        inputs, speaker_ids, emotion_ids = inputs
        
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs)
        
        embedded_speaker = self.speaker_embedding(speaker_ids)
        embedded_speaker = self.speaker_layer(embedded_speaker)
        embedded_emotion = self.emotion_embedding(emotion_ids)
        embedded_emotion = self.emotion_layer(embedded_emotion)
        
        if gst_token:
            query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
            GST = torch.tanh(self.gst.stl.embed)
            key = GST[gst_token].unsqueeze(0).expand(1, -1, -1).repeat(1, embedded_text.size(1), 1)
            
            embedded_gst = self.gst.stl.attention(query, key)
                
            if gst_intensity:
                embedded_gst = embedded_gst * gst_intensity
        elif gst_embedded is not None:
            embedded_gst = gst_embedded
        elif gst_mel is not None:
            embedded_gst = self.gst(gst_mel)
        else:
            embedded_gst = torch.zeros(inputs.size(0), self.token_embedding_size).cuda()
        
        embedded_styles = torch.cat((embedded_speaker, embedded_emotion, embedded_gst), dim=1)[:,None].repeat(1, embedded_text.size(1), 1)
        encoder_outputs = torch.cat(
            (embedded_text, embedded_styles), dim=2)
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, embedded_gst, logging, init_gst=init_gst)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
    
    def VC_inference(self, inputs, gst_token=None, gst_mel=None, logging=True, gst_intensity=None, question=None):
        targets, speaker_ids, emotion_ids = inputs
        
        # embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # embedded_text = self.encoder.inference(embedded_inputs)
        embedded_text = self.contents_encoder.inference(targets)
        
        embedded_speaker = self.speaker_embedding(speaker_ids)
        embedded_speaker = self.speaker_layer(embedded_speaker)
        embedded_emotion = self.emotion_embedding(emotion_ids)
        embedded_emotion = self.emotion_layer(embedded_emotion)
        
        if gst_token:
            query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
            GST = torch.tanh(self.gst.stl.embed)
            key = GST[gst_token].unsqueeze(0).expand(1, -1, -1).repeat(1, embedded_text.size(1), 1)
            
            embedded_gst = self.gst.stl.attention(query, key)
                
            if gst_intensity:
                embedded_gst = embedded_gst * gst_intensity
        elif gst_mel is not None:
            embedded_gst = self.gst(gst_mel)
        else:
            embedded_gst = torch.zeros(inputs.size(0), self.token_embedding_size).cuda()
        
        embedded_styles = torch.cat((embedded_speaker, embedded_emotion, embedded_gst), dim=1)[:,None].repeat(1, embedded_text.size(1), 1)
        encoder_outputs = torch.cat(
            (embedded_text, embedded_styles), dim=2)
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, embedded_gst, logging)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        