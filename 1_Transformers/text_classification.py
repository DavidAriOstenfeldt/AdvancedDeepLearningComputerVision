import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
from time import time
from torchinfo import summary

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    test, val = test.split(split_ratio=0.5)
    print('training: ', len(train), 'val:', len(val), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, val_iter, test_iter


def main(model_name='', embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1
    ):

    
    
    loss_function = nn.CrossEntropyLoss()
    start = time()
    print('Preparing data iterator')
    train_iter, val_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                            batch_size=batch_size
    )
    print(f'-- took {time() - start:.3} seconds')


    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS,
                                  )
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))
    # We wish to save the best model based on the validation accuracy after each epoch
    best_acc = 0.0
    best_epoch = 0
    best_model = None
    # training loop
    model_params = {'num_heads': num_heads,
                    'num_layers': num_layers,
                    'pos_enc': pos_enc,
                    'pool': pool, 
                    'dropout': dropout, 
                    'num_epochs': num_epochs, 
                    'batch_size': batch_size, 
                    'lr': lr, 
                    'weight_decay': weight_decay,
                    'train_duration': 0.0}
    start = time()
    train_acc_list = []
    val_acc_list = []
    # training loop
    # iterator = tqdm.tqdm(range(num_epochs))
    for e in range(num_epochs):
        # print(f'\n epoch {e}')
        train_total = 0.0
        train_correct = 0.0
        print(f'\n epoch {e}')
        model.train()
        train_iterator = tqdm.tqdm(train_iter)
        for batch in train_iterator:
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label)
            loss.backward()
            train_total += float(input_seq.size(0))
            train_correct += float((label == out.argmax(dim=1)).sum().item())
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()
            train_acc = train_correct / train_total
            # train_iterator.set_postfix_str(train_acc)
            train_iterator.set_description(f'-- {"train"} accuracy {train_acc:.3}')

        train_acc_list.append(train_acc)
        end = time()
        model_params['train_duration'] = end - start
        model_params['train_accuracy'] = train_acc
        total_train_time = end - start

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for batch in test_iter:
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_acc_list.append(acc)
            print(f'-- validation accuracy {acc:.3}')
            if acc > best_acc:
                best_model, best_acc, best_epoch = save_best_model(model_name, model, e, acc, best_acc, best_epoch, model_params)
    
    # Test the best model
    for batch in val_iter:
        input_seq = batch.text[0]
        batch_size, seq_len = input_seq.size()
        label = batch.label - 1
        if seq_len > MAX_SEQ_LEN:
            input_seq = input_seq[:, :MAX_SEQ_LEN]
        out = best_model(input_seq).argmax(dim=1)
        tot += float(input_seq.size(0))
        cor += float((label == out).sum().item())
    acc = cor / tot
    print(f'-- test accuracy: {acc:.3}')        

    with open(f'1_Transformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'a') as f:
        # Add the test accuracy to the text file
        f.write(f'test accuracy: {acc}\n')
        # Add the total training time to the text file
        f.write(f'total training time: {total_train_time}\n')
        # Add the complete training accuracy list to the text file
        f.write(f'train accuracy list: {train_acc_list}\n')
        # Add the complete validation accuracy list to the text file
        f.write(f'val accuracy list: {val_acc_list}\n')

def save_best_model(model_name, model, epoch, acc, best_acc, best_epoch, model_params):
    # Delete the previous best model state dict
    if best_acc != 0.0:
        os.remove(f'1_Transformers/models/{model_name}_best_model_e{best_epoch+1}.pth')
        os.remove(f'1_Transformers/models/{model_name}_best_model_e{best_epoch+1}.txt')
    ##########
    best_epoch = epoch
    ##########
    best_acc = max(best_acc, acc)
    torch.save(model.state_dict(), f'1_Transformers/models/{model_name}_best_model_e{best_epoch+1}.pth')
    # Save model parameters to a text file
    text_output = ''
    for k, v in model_params.items():
        text_output += f'{k}: {v}\n'
    model_stats = summary(model, input_size=(1, 3, 32, 32), verbose=0)
    text_output += f'MACs: {model_stats.total_mult_adds}\n'
    text_output += f'Number of parameters: {model_stats.total_params}\n'
    text_output += f'Number of trainable parameters: {model_stats.trainable_params}\n'
    text_output += f'Validation accuracy: {best_acc}\n'
    # Save text_output to a file
    with open(f'1_Transformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'w') as f:
        f.write(text_output)
    print(f'(New Model Saved)')
    return model, best_acc, best_epoch

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main(model_name='nh4_nl4', num_epochs=50, num_heads=4, num_layers=4)
