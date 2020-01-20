import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import torchtext
from torchtext.data.utils import get_tokenizer
import torchtext.data as data

def get_dataset(dataset, vectors=None, device=None, batch_size=128):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = dataset.splits(TEXT)
    if vectors is None:
        TEXT.build_vocab(train_txt)
    else:
        TEXT.build_vocab(train_txt, vectors=vectors)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dataset loaded on: ", device)

    eval_batch_size = 10
    bptt_len = 64
    return (TEXT, data.BPTTIterator.splits(
            (train_txt, val_txt, test_txt), batch_sizes=(batch_size, eval_batch_size, eval_batch_size), bptt_len=bptt_len, device=device))


def main(device, model, TEXT, train_iter, val_iter, test_iter, model_path, no_train, epochs):
    criterion = nn.CrossEntropyLoss()
    lr = 0.001 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    import time
    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(TEXT.vocab.stoi)
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            output = model(batch.text.to(device))
            loss = criterion(output.view(-1, ntokens), batch.target.view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 100
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, len(train_iter), optimizer.param_groups[0]['lr'],
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
                output = eval_model(batch.text.to(device))
                loss = criterion(output.view(-1, ntokens), batch.target.to(device).view(-1)).item()
                total_loss += loss
        return total_loss / (len(data_iter) - 1)

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

    if no_train:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        for _ in range(5):
            sentence = ' '.join(map(lambda i: TEXT.vocab.itos[i], sample_sentence(model)))
            print(sentence)

        import IPython
        IPython.embed()
    else:
        best_val_loss = float("inf")
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

            scheduler.step(val_loss)

            for _ in range(5):
                sentence = ' '.join(map(lambda i: TEXT.vocab.itos[i], sample_sentence(model)))
                print(sentence)

        test_loss = evaluate(best_model, test_iter)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        print('=' * 89)
        torch.save(best_model.state_dict(), model_path)
