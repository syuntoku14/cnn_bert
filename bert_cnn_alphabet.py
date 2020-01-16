import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.utils.data as data
import sys
from torch.nn.utils import clip_grad_norm_
import parser
import torch
import os
import IPython

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def char_to_idx(char):
    if char.islower():
        idx = ord(char) - ord("a")
    else:
        idx = ord(char) - ord("A") + 26
    return idx


def idx_to_char(idx):
    return alphabet[idx]


def is_word_unfeasible(word):
    def is_alphabet(word):
        return all(c in alphabet for c in word)
    return ("unused" in word
            or "#" in word
            or not is_alphabet(word)
            or len(word) < 4)


class BertDataset(Dataset):
    def __init__(self, device):
        self.device = device
        # get the tokenized words.
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # load BERT base model
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        # input characters
        self.CHAR_VOCAB_SIZE = 52
        words = self.tokenizer.vocab.keys()
        self.vocabs = [word for word in words if not is_word_unfeasible(word)]
        print("{} -> {}".format(len(words), len(self.vocabs)))
        self.chars = [torch.LongTensor([char_to_idx(c) for c in word])
                      for word in self.vocabs]
        self.chars = rnn.pad_sequence(self.chars).to(self.device).T

        # word embeddings of bert
        ids = torch.LongTensor(
            self.tokenizer.convert_tokens_to_ids(self.vocabs)).to(self.device)
        self.word_embed = self.bert.embeddings.word_embeddings(
            ids.unsqueeze(0)).squeeze(0)

    def __len__(self):
        return len(self.vocabs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.chars[idx], self.word_embed[idx]


class Conv1dBlockBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, p=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel,
                      kernel_size=kernel_size, stride=stride),
            nn.Dropout(p),
            nn.PReLU(),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CNN_LM(nn.Module):
    def __init__(self, char_vocab_size, char_len, embed_dim, chan_size, hid_size, bert_hid_size):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, embed_dim)
        convs = []
        for i in range(char_len - 1):
            if i == 0:
                convs.append(Conv1dBlockBN(embed_dim, chan_size, 2, stride=1))
            else:
                convs.append(Conv1dBlockBN(chan_size, chan_size, 2, stride=1))
        self.convs = nn.Sequential(*convs)
        self.fc1 = nn.Linear(chan_size, hid_size)
        self.fc2 = nn.Linear(hid_size, bert_hid_size)

    def forward(self, x):
        # (batch_size, embed_dim, context_width)
        x = self.embedding(x).permute(0, 2, 1)
        x = self.convs(x)  # (batch_size, chan_size, 1)
        x = x.squeeze(2)  # (batch_size, chan_size)
        x = F.relu(self.fc1(x))  # (batch_size, hid_size)
        x = self.fc2(x)  # (batch_size, vocab_size)
        return x


def to_word(char_idxs):
    return ''.join(map(lambda idx: alphabet[idx], filter(lambda idx: idx != 0, char_idxs)))


def find_closest_words(index, inputs, outputs, targets):
    words = inputs[torch.norm(targets - outputs[index],
                              dim=1).topk(20, largest=False).indices]
    #words = inputs[(torch.argsort(torch.norm(targets - outputs[index], dim=1)) < 20).nonzero().squeeze(1)]
    return list(map(to_word, words.to(torch.device('cpu')).tolist()))


def run_model_through_dataset(dataloader, model):
    inputs = []
    outputs = []
    targets = []
    model.eval()
    for inputs_, targets_ in dataloader:
        outputs_ = model(inputs_)
        inputs.append(inputs_)
        outputs.append(outputs_)
        targets.append(targets_)
    return (torch.cat(inputs, dim=0), torch.cat(outputs, dim=0), torch.cat(targets, dim=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--channel_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--skip_train', type=bool, default=False)
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = args.embed_size
    num_epochs = args.num_epochs

    dataset = BertDataset(device)
    # test train
    SAMPLE_SIZE = len(dataset)
    TRAIN_SIZE = int(SAMPLE_SIZE * 0.8)
    train_dataset, val_dataset = data.random_split(
        dataset, [TRAIN_SIZE, SAMPLE_SIZE - TRAIN_SIZE])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    CHAR_VOCAB_SIZE = 52
    BERT_EMBED_DIM = 768
    model = CNN_LM(char_vocab_size=CHAR_VOCAB_SIZE,
                   char_len=dataset.chars.shape[1], embed_dim=args.embed_size,
                   chan_size=args.channel_size, hid_size=args.hidden_size,
                   bert_hid_size=BERT_EMBED_DIM)
    model = model.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
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
            for batch, (inputs, targets) in enumerate(train_dataloader):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

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
            for batch, (inputs, targets) in enumerate(val_dataloader):
                # Forward pass
                outputs = model(inputs)
                test_loss += criterion(outputs, targets)
            print('Test: Epoch {}, Loss: {:.4f}'
                  .format(epoch+1, test_loss.item() / len(val_dataloader)))

        # Save the model checkpoints
        torch.save(model.state_dict(), './data/bert_cnn.ckpt')
