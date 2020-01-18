import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
import re
import sys
import argparse
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.utils.data as data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dictionary(object):
    def __init__(self):
        self.word2id = {"<pad>":0}
        self.word_id = 1
        self.max_word_len = 1
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 "
        self.char2id = {"<pad>":0, "<bow>":1, "<eow>":2, "<eos>":3}  # bow, eow is not necessary?
        for i, char in enumerate(chars):
            self.char2id[char] = 4 + i

    def word_to_char_ids(self, word):
        if word == "<eos>":
            id_list = [self.char2id[word]]
        else:
            id_list = [self.char2id[char] for char in word]
        pad_list = [0 for _ in range(self.max_word_len - len(id_list))]
        id_list = [self.char2id["<bow>"]] + id_list + pad_list + [self.char2id["<eow>"]]
        
        return id_list
    
    def add_word(self, word):
        if not word in self.word2id:
            self.word2id[word] = self.word_id
            self.word_id += 1
    

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
    def normalize_word(self, word):
        if word == "<eos>":
            return word
        word = word.lower()
        word = re.sub('[^a-z.,0-9 ]+', '', word)
        return word

    def get_data(self, path):

        # get max word length
        with open(path, 'r') as f:
            for line in f:
                for punct in string.punctuation:
                    line = line.replace(punct, ' ') 
                words_line = line.split() + ['<eos>']
                for word in words_line: 
                    self.dictionary.max_word_len = \
                        max(self.dictionary.max_word_len, len(word))
            
        # Add words to the dictionary
        word_ids = []
        char_ids = []       
        with open(path, 'r') as f:
            for line in f:
                for punct in string.punctuation:
                    line = line.replace(punct, ' ')                
                words_line = line.split() + ['<eos>']
                wids_line = []
                cids_line = []
                for word in words_line: 
                    word = self.normalize_word(word)
                    self.dictionary.add_word(word)  
                    wids_line.append(self.dictionary.word2id[word])
                    cids_line.append(self.dictionary.word_to_char_ids(word))
                word_ids.append(torch.LongTensor(wids_line))
                char_ids.append(torch.LongTensor(cids_line))

        seq_lens = [len(ids) for ids in word_ids]
        return seq_lens, word_ids, char_ids


class BrownDataset(Dataset):
    def __init__(self, corpus_file, device):
        self.device = device
        self.corpus_file = corpus_file
        self.corpus = Corpus()
        self.seq_lens, self.word_ids, self.char_ids = self.corpus.get_data(corpus_file)
        
        self.char_size = len(self.corpus.dictionary.char2id)
        self.vocab_size = len(self.corpus.dictionary.word2id)
        self.id2word = {i:word for i, word in 
                        zip(self.corpus.dictionary.word2id.values(),
                            self.corpus.dictionary.word2id.keys())}
        self.id2char = {i:char for i, char in 
                        zip(self.corpus.dictionary.char2id.values(),
                            self.corpus.dictionary.char2id.keys())}
        
    def charids_to_word(self, _ids: int):
        def id2char(_id):
            if self.id2char[_id] == "<pad>":
                return ""
            return self.id2char[_id]
        return "".join(list(map(id2char, _ids)))

    def __len__(self):
        # -1 for the target
        return len(self.word_ids) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seq_lens = self.seq_lens[idx] # - 1
        _input = self.char_ids[idx][:-1]
        _target = self.word_ids[idx][1:]

        return seq_lens, _input, _target
    
    def collate(self, batch):
        seq_lens = [item[0] for item in batch]
        _input = [item[1] for item in batch]
        _target = [item[2] for item in batch]
        return seq_lens, _input, _target
    
def get_data(seq_lens, _input, _target, device):
    seq_lens = torch.LongTensor(seq_lens) - 1
    # sort dataset as the sequence length
    sorted_seq_lens, sort_idx = seq_lens.sort(dim=0, descending=True)
    sort_idx = sort_idx[sorted_seq_lens!=0]
    sorted_seq_lens = sorted_seq_lens[sorted_seq_lens!=0]

    sorted_input = rnn.pad_sequence(_input, batch_first=True)[sort_idx]
    sorted_target = rnn.pad_sequence(_target, batch_first=True)[sort_idx]

    return sorted_seq_lens.to(device), sorted_input.to(device), sorted_target.to(device)

class Highway(nn.Module):
    """Highway network"""
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = torch.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1-t, x)


class charLM(nn.Module):
    """CNN + highway network + LSTM
    # Input: 
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        vocab_size: num of unique words
        char_size: num of characters
        use_gpu: True or False
    """
    def __init__(self, char_emb_dim, word_emb_dim,  
                vocab_size, char_size, device):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.device = device

        # char embedding layer
        self.char_embed = nn.Embedding(char_size, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        
        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,           # in_channel
                    out_channel, # out_channel
                    kernel_size=(char_emb_dim, filter_width), # (height, width)
                    bias=True
                    )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        # LSTM
        self.lstm_num_layers = 2

        self.lstm = nn.LSTM(input_size=self.highway_input_dim, 
                            hidden_size=self.word_emb_dim, 
                            num_layers=self.lstm_num_layers,
                            bias=True,
                            dropout=0.5,
                            batch_first=True)

        # output layer
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

        
        for x in range(len(self.convolutions)):
            self.convolutions[x] = self.convolutions[x].to(device)
        self.highway1 = self.highway1.to(device)
        self.highway2 = self.highway2.to(device)
        self.lstm = self.lstm.to(device)
        self.dropout = self.dropout.to(device)
        self.char_embed = self.char_embed.to(device)
        self.linear = self.linear.to(device)
        self.batch_norm = self.batch_norm.to(device)


    def forward(self, x, hidden):
        # Input: Variable of Tensor with shape [num_seq, seq_len, max_word_len+2]
        # Return: Variable of Tensor with shape [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]
        
        x = self.char_embed(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]
        
        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]
        
        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway1(x)
        x = self.highway2(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.contiguous().view(lstm_batch_size,lstm_seq_len, -1)
        # [num_seq, seq_len, total_num_filters]
        
        x, hidden = self.lstm(x, hidden)
        # [seq_len, num_seq, hidden_size]
        
        x = self.dropout(x)
        # [seq_len, num_seq, hidden_size]
        
        x = x.contiguous().view(lstm_batch_size*lstm_seq_len, -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]
        return x, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.word_emb_dim).to(self.device), \
                torch.zeros(2, batch_size, self.word_emb_dim).to(self.device))


    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = torch.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)            
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)
        
        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)

def compute_loss(crit, pred, _target, lens):
    packed_target = rnn.pack_padded_sequence(_target, seq_lens, batch_first=True).data
    packed_pred = rnn.pack_padded_sequence(pred.view(_target.shape[0], _target.shape[1], -1),
                             seq_lens, batch_first=True).data
   
    loss = crit(packed_pred, packed_target)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--skip_train', type=bool, default=False)
    args = parser.parse_args()

    path_corpus = args.data_path + "/browncorpus.txt"

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BrownDataset(path_corpus, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=dataset.collate)
    vocab_size = dataset.vocab_size

    # test train
    SAMPLE_SIZE = len(dataset)
    TRAIN_SIZE = int(SAMPLE_SIZE * 0.8)
    train_dataset, val_dataset = data.random_split(
        dataset, [TRAIN_SIZE, SAMPLE_SIZE - TRAIN_SIZE])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=dataset.collate)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=dataset.collate)

    word_embed_dim = 300
    char_embedding_dim = 15
    vocab_size = dataset.vocab_size
    char_size = dataset.char_size
    # cnn_batch_size == lstm_seq_len * lstm_batch_size

    model = charLM(char_embedding_dim, 
                word_embed_dim, 
                vocab_size,
                char_size,
                device=device)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if args.skip_train:
        checkpoint = torch.load('./data/bert_cnn.ckpt')
        model.load_state_dict(checkpoint)
        IPython.embed()
    else:
        print("training start")
        # Train the model
        for epoch in range(args.num_epochs):
            model.train()
            for batch, (seq_lens, inputs, targets) in enumerate(train_dataloader):
                seq_lens, inputs, targets = get_data(seq_lens, inputs, targets, device)
                # Forward pass
                hidden = model.init_hidden(inputs.shape[0])
                pred, _ = model(inputs, hidden)
                loss = compute_loss(criterion, pred, targets, seq_lens)

                # Backward and optimize
                model.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 20 == 0:
                    print('Training: Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                          .format(epoch+1, args.num_epochs, batch, len(train_dataloader), loss.item()))

            scheduler.step()

            test_loss = 0
            model.eval()
            for batch, (seq_lens, inputs, targets) in enumerate(val_dataloader):
                seq_lens, inputs, targets = get_data(seq_lens, inputs, targets, device)
                # Forward pass
                hidden = model.init_hidden(inputs.shape[0])
                pred, _ = model(inputs, hidden)
                loss = compute_loss(criterion, pred, targets, seq_lens)
                test_loss += loss.item()

            print('Test: Epoch {}, Loss: {:.4f}'
                  .format(epoch+1, test_loss / len(val_dataloader)))

        # Save the model checkpoints
        torch.save(model.state_dict(), args.data_path + '/charlm.ckpt')
