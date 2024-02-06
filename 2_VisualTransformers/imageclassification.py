import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import time
from torchinfo import summary

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

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
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        for image, label in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()
        end = time.time()
        model_params['train_duration'] = end - start
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
            print(f'-- {"validation"} accuracy {acc:.3}')
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
    # Add the test accuracy to the text file
    with open(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'a') as f:
        f.write(f'Test accuracy: {acc}\n')
    # Add the total training time to the text file
    with open(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt', 'a') as f:
        f.write(f'Total training time: {total_train_time}\n')
        
def save_best_model(model_name, model, epoch, acc, best_acc, best_epoch, model_params):
    # Delete the previous best model state dict
    if best_acc != 0.0:
        os.remove(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.pth')
        os.remove(f'2_VisualTransformers/models/{model_name}_best_model_e{best_epoch+1}.txt')
    best_epoch = epoch
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
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main(model_name='nh4_nl4', num_epochs=20, num_heads=4, num_layers=4)
    main(model_name='nh8_nl8', num_epochs=20, num_heads=8, num_layers=8)
    main(model_name='nh16_nl16', num_epochs=20, num_heads=16, num_layers=16)
