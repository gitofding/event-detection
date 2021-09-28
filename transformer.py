import torch
import math
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import tensor2var
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=300, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Trasformer(nn.Module):
    def __init__(self, config):
        # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.3):
        # super(TrigRNN, self).__init__()
        super(Trasformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')

        self.config = config
        ninp = config.ninp
        dropout = config.trans_dropout
        nhid = config.nhid
        nhead = config.nhead  # 多头注意力机制
        nlayers = config.nlayers  # 3个Encoder Layer

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.wdrop = nn.Dropout(dropout)
        self.model = nn.LSTM(config.embed_dim, ninp, bidirectional=True, batch_first=True)
        self.position = PositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.trigdrop = nn.Dropout(config.dropout)
        self.hidden2tag = nn.Linear(ninp * 2, config.tagset_size)
        self = self.to(config.device)

    def forward(self, src, seq_lengths):
        embeds = self.word_embeddings(src)
        embeds = self.wdrop(embeds)
        (bsize, slen, _) = embeds.size()
        embeds = self.position(embeds)
        embeds = self.transformer_encoder(embeds)
        hidden = None
        # pack_input = pack_padded_sequence(embeds,seq_lengths, True)

        pack_input = pack_padded_sequence(embeds, seq_lengths.to("cpu"), True)
        output, hidden = self.model(pack_input, hidden)
        output, _ = pad_packed_sequence(output)
        # print(output.size())

        output = output.transpose(1, 0)
        hidden_in_trigs = self.trigdrop(output)
        tag_space = self.hidden2tag(hidden_in_trigs.contiguous().view(bsize * slen, -1))
        return tag_space
