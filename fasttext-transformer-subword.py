import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import FastText
import itertools
from fasttext import load_model
from torch.autograd import Variable
import numpy as np
import torchtext
import lib 

# taken from https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/FastTextEmbeddingBag.py
class FastTextEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, model_path, itos, device):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        self.itos = itos
        self.device = device
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, indices):
        #print(indices.shape)
        orig = indices.shape
        indices = indices.view(orig[0] * orig[1])

        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for index in indices:
            word = self.itos[index]
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = torch.LongTensor(word_subinds).to(device)
        offsets = torch.LongTensor(word_offsets).to(device)

        result = super().forward(ind, offsets)
        #print(result.shape)
        result = result.view(orig[0], orig[1], -1)
        #print(result.shape)
        return result

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device, fasttext_model_path, itos, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = FastTextEmbeddingBag(fasttext_model_path, itos, device)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.device = device

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

TEXT, (train_iter, val_iter, test_iter) = lib.get_dataset(torchtext.datasets.WikiText103, device="cpu", batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 100 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, device,
        fasttext_model_path="/home/ubuntu/fil9.bin", itos=TEXT.vocab.itos, dropout=dropout).to(device)

NO_TRAIN=False
#MODEL_PATH='./data/transformer-fasttext-big-subword.ckpt' # test loss  5.16 | test ppl   174.89
MODEL_PATH='./data/transformer-fasttext-wiki103.ckpt'
#MODEL_PATH='./data/transformer-fasttext-subword.ckpt'
#MODEL_PATH='./data/transformer-fasttext-dropout-big-subword.ckpt'

lib.main(device, model, TEXT, train_iter, val_iter, test_iter, MODEL_PATH, NO_TRAIN, epochs=15)
