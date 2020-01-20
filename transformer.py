import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import itertools
from tqdm import tqdm

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
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

batch_size = 20
eval_batch_size = 10
bptt_len = 35
train_iter, val_iter, test_iter = data.BPTTIterator.splits(
        (train_txt, val_txt, test_txt), batch_sizes=(batch_size, eval_batch_size, eval_batch_size), bptt_len=bptt_len, device=device)

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 80 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
#nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
#dropout = 0.4 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, device, dropout).to(device)

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
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        output = model(batch.text)
        loss = criterion(output.view(-1, ntokens), batch.target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_iter), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_iter):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for batch in data_iter:
            output = eval_model(batch.text)
            loss = criterion(output.view(-1, ntokens), batch.target.view(-1)).item()
            total_loss += loss
    return total_loss / (len(data_iter) - 1)

NO_TRAIN=False
#MODEL_PATH='./data/transformer-big.ckpt' # test loss  5.19 | test ppl   179.86
MODEL_PATH='./data/transformer.ckpt' # test loss  5.37 | test ppl   213.86
#MODEL_PATH='./data/transformer-dropout.ckpt' # test loss  5.47 | test ppl   237.30

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
        val_loss = evaluate(model, val_iter)
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

    test_loss = evaluate(best_model, test_iter)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)
    torch.save(best_model.state_dict(), MODEL_PATH)
