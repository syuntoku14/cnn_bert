import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import FastText
import torchtext.data as data
import itertools
from fasttext import load_model
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import lib 
import torchtext

class FastTextAttentionEmbedding(nn.Embedding):
    def __init__(self, model_path, itos, device, nhead, nhid, nlayers, dropout):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        vocab_size, emb_size = input_matrix.shape # (vocab_size, emb_size)

        super().__init__(vocab_size, emb_size)
        self.weight.data.copy_(torch.FloatTensor(input_matrix))
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.emb_size = emb_size
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.itos = itos
        self.device = device
        self.src_mask = None

        self.subinds_cache = dict()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def transformer_forward(self, x):
        x = self.pos_encoder(x * math.sqrt(self.emb_size)) # (number_of_ngrams, emb_size)
        x = self.transformer_encoder(x, self.src_mask) # (embd_size)
        return x

    def get_subwords(self, word):
        if word in self.subinds_cache:
            return self.subinds_cache[word]

        _, subinds = self.model.get_subwords(word)
        self.subinds_cache[word] = torch.LongTensor(subinds).to(self.device)
        return self.subinds_cache[word]

    def single_forward(self, index):
        subinds = self.get_subwords(self.itos[index]) # (number_of_ngrams)
        embs = super().forward(subinds).unsqueeze(0) # (1, number_of_ngrams, emb_size)
        out = self.transformer_forward(embs) # (1, time, emb_size)
        return out.squeeze(0) # (time, emb_size)

    def rank_subwords_(self, index):
        norms = self.single_forward(index).norm(dim=1)
        subwords, _ = self.model.get_subwords(self.itos[index])
        res = list(map(lambda x: (x[0]/len(x[1]), x[1]), zip(norms, subwords)))
        return sorted(res)

    def rank_subwords(self, index):
        norms = self.single_forward(index).norm(dim=1)
        subwords, _ = self.model.get_subwords(self.itos[index])
        res = list(zip(norms, subwords))
        return sorted(res)

    def subwords_norm_dict_(self, index):
        norms = self.single_forward(index).norm(dim=1)
        subwords, _ = self.model.get_subwords(self.itos[index])
        table = dict(zip(subwords, norms))
        word = '<' + self.itos[index] + '>'
        res = []
        for i in range(2, len(word) - 1):
            a = word[0:i]
            b = word[i:len(word)]
            if a in table and b in table:
                res.append((table[a] + table[b], a + ' ' + b))

        return sorted(res)

    def subwords_norm_dict(self, index):
        norms = self.single_forward(index).norm(dim=1)
        subwords, _ = self.model.get_subwords(self.itos[index])
        table = dict(map(lambda x: (x[0], x[1]/len(x[0])), zip(subwords, norms)))
        word = '<' + self.itos[index] + '>'
        res = []
        for i in range(2, len(word) - 1):
            a = word[0:i]
            b = word[i:len(word)]
            if a in table and b in table:
                res.append((table[a] + table[b], a + ' ' + b))

        return sorted(res)

    def forward(self, indices):
        # TODO: deal with <unk> et al.
        # TEXT.vocab.itos[:5] -> ['<unk>', '<pad>', '<sos>', '<eos>', 'the']
        orig = indices.shape
        indices = indices.view(orig[0] * orig[1])

        subinds_list = [ self.get_subwords(self.itos[index]) for index in indices ] # list of (number_of_ngrams)
        # TEXT.vocab.stoi['<pad>'] == 1
        subinds_mat = pad_sequence(subinds_list, batch_first=True, padding_value=1) # (batch, max_number_of_ngrams) 
        embs_batch = super().forward(subinds_mat) # (batch, max_number_of_ngrams, emb_size)

        out = self.transformer_forward(embs_batch) # (batch, time, emb_size)
        out = out.sum(1) # (batch, emb_size)

        out = out.view(orig[0], orig[1], -1) # (orig[0], orig[1], emb_size)
        return out

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device, fasttext_model_path, itos, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = FastTextAttentionEmbedding(fasttext_model_path, itos, device, nhead=1, nhid=100, nlayers=1, dropout=dropout)
        #self.encoder = FastTextEmbeddingBag(fasttext_model_path, itos, device)
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
            mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
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
#MODEL_PATH='./data/transformer-fasttext-subword-attention.ckpt' # test loss  5.56 | test ppl   260.43
#MODEL_PATH='./data/transformer-fasttext-subword-attention-canonical.ckpt'
MODEL_PATH='./data/transformer-fasttext-subword-attention-wikitext103.ckpt'
#MODEL_PATH='./data/transformer-fasttext-dropout-big-subword-attention.ckpt'

lib.main(device, model, TEXT, train_iter, val_iter, test_iter, MODEL_PATH, NO_TRAIN, epochs=15, no_text_transfer=True)
