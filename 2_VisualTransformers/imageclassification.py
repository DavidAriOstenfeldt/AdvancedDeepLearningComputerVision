import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import time
import matplotlib.pyplot as plt
from torchinfo import summary

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
from vit_grad_rollout import VITAttentionGradRollout

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./.data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./.data', train=False,
                                           download=True, transform=test_transform) 

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, valset = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False
    )
    return trainloader, valloader, testloader, trainset, valset, testset


def main(model_name='', image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
         
    ):

    loss_function = nn.CrossEntropyLoss()

    train_iter, val_iter, test_iter, _, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
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
    start = time.time()

    train_acc_list = []
    val_acc_list = []
    for e in range(num_epochs):
        train_total = 0.0
        train_correct = 0.0
        print(f'\n epoch {e}')
        model.train()
        train_iterator = tqdm.tqdm(train_iter)
        for image, label in train_iterator:
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_total += float(image.size(0))
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
        end = time.time()
        model_params['train_duration'] = end - start
        model_params['train_accuracy'] = train_acc
        total_train_time = end - start

        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for image, label in val_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image).argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_acc_list.append(acc)
            print(f'-- validation accuracy {acc:.3}')
            # train_iterator.set_description(f'-- {"validation"} accuracy {acc:.3}')
            if acc > best_acc:
                best_model, best_acc, best_epoch = save_best_model(model_name, model, e, acc, best_acc, best_epoch, model_params)
            
    # Test the best model
    for image, label in test_iter:
        best_model.eval()
        if torch.cuda.is_available():
            image, label = image.to('cuda'), label.to('cuda')
        out = best_model(image).argmax(dim=1)
        tot += float(image.size(0))
        cor += float((label == out).sum().item())
    acc = cor / tot
    print(f'-- test accuracy: {acc:.3}')
    
    with open(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'a') as f:
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
        os.remove(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.pth')
        os.remove(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt')
    ##########
    best_epoch = epoch
    ##########
    best_acc = max(best_acc, acc)
    torch.save(model.state_dict(), f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.pth')
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
    with open(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'w') as f:
        f.write(text_output)
    print(f'(New Model Saved)')
    return model, best_acc, best_epoch


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    print('------------------------------------------------------------')
    print('TRAINING nh4_nl4')
    main(model_name='nh4_nl4', num_epochs=50, num_heads=4, num_layers=4)
    print('------------------------------------------------------------')
    print('TRAINING nh8_nl8')
    main(model_name='nh8_nl8', num_epochs=50, num_heads=8, num_layers=8)
    print('------------------------------------------------------------')
    print('TRAINING nh8_nl12')
    main(model_name='nh8_nl12', num_epochs=50, num_heads=8, num_layers=12)
    # print('------------------------------------------------------------')
    # print('TRAINING nh12_nl16')
    # main(model_name='nh12_nl16', num_epochs=50, num_heads=8, num_layers=16)
    # print('------------------------------------------------------------')
    # print('TRAINING nh12_nl32')
    # main(model_name='nh12_nl32', num_epochs=50, num_heads=8, num_layers=32)

