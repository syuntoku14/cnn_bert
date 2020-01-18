import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import FastText
import itertools
from fasttext import load_model
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

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

#    def forward(self, indices):
#        orig = indices.shape
#        indices = indices.view(orig[0] * orig[1])
#
#        out = []
#        for index in indices:
#            word = self.itos[index]
#            #_, subinds = self.model.get_subwords(word)
#            #embs = super().forward(torch.LongTensor(subinds).to(self.device))
#            subinds = self.get_subwords(word)
#            embs = super().forward(subinds)
#            embs = embs.unsqueeze(0) # (1, number_of_ngrams, emb_size)
#            o = self.transformer_forward(embs).squeeze(0)[-1]
#            out.append(o) # (1, emb_size)
#        out = torch.cat(out, dim=0) # (orig[0] * orig[1], emb_size)
#        out = out.view(orig[0], orig[1], -1) # (orig[0], orig[1], emb_size)
#        return out

    def forward(self, indices):
        orig = indices.shape
        indices = indices.view(orig[0] * orig[1])

        embs_list = []
        for index in indices:
            word = self.itos[index]
            #_, subinds = self.model.get_subwords(word)
            #embs = super().forward(torch.LongTensor(subinds).to(self.device))
            subinds = self.get_subwords(word)
            embs = super().forward(subinds) # (number_of_ngrams, emb_size)
            embs_list.append(embs)

        embs_batch = pad_sequence(embs_list, batch_first=True) # (batch, max_number_of_ngrams, emb_size)
        out = self.transformer_forward(embs_batch) # (batch, time, emb_size)
        out = out.permute(1, 0, 2) # (time, batch, emb_size)
        out = out[-1] # (batch, emb_size)

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


import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 100 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
#nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
#dropout = 0.4 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, device,
        fasttext_model_path="/home/ubuntu/fastText/result/fil9.bin", itos=TEXT.vocab.itos, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, bptt))):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

NO_TRAIN=False
#MODEL_PATH='./data/transformer-fasttext-big-subword-attention.ckpt'
MODEL_PATH='./data/transformer-fasttext-subword-attention.ckpt'
#MODEL_PATH='./data/transformer-fasttext-dropout-big-subword-attention.ckpt'

MAX_SENT_LEN=100
def sample_sentence(model):
    model.eval()
    sent = [[TEXT.vocab.stoi['<sos>']]]
    eos = TEXT.vocab.stoi['<eos>']
    while sent[-1][0] != eos and len(sent) < 100:
        logits = model(torch.LongTensor(sent).to(device))
        prob = F.softmax(logits.squeeze(1)[-1], dim=0)
        next_word = prob.multinomial(num_samples=1).item()
        sent.append([next_word])
    return list(itertools.chain.from_iterable(sent))

if NO_TRAIN:
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint)
    for _ in range(5):
        sentence = ' '.join(map(lambda i: TEXT.vocab.itos[i], sample_sentence(model)))
        print(sentence)

    import IPython
    IPython.embed()
else:
    best_val_loss = float("inf")
    epochs = 15 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

        for _ in range(5):
            sentence = ' '.join(map(lambda i: TEXT.vocab.itos[i], sample_sentence(model)))
            print(sentence)

    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)
    torch.save(best_model.state_dict(), MODEL_PATH)
