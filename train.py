import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm

def main(device, model, TEXT, train_iter, val_iter, test_iter, model_path, no_train, epochs):
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

            scheduler.step()

            for _ in range(5):
                sentence = ' '.join(map(lambda i: TEXT.vocab.itos[i], sample_sentence(model)))
                print(sentence)

        test_loss = evaluate(best_model, test_iter)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        print('=' * 89)
        torch.save(best_model.state_dict(), model_path)
