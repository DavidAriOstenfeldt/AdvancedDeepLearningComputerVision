import os
from tqdm import tqdm
import argparse

# torch imports
import torch
import torch.nn as nn
from torch.optim import AdamW

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# custom imports
from model import Network
from loss import VFILoss

from dataset import TripletDataset
from dataset.helpers import *

#NOTE: DO NOT CHANGE THE SEEDs
torch.manual_seed(102910)
np.random.seed(102910)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=int, default=1, help='Configuration for losses (1-3)')
args = parser.parse_args()
conf = args.conf
assert conf in [1,2,3], f'Invalid configuration!'

"""
Data paths
"""
train_path = './data/DAVIS_Triplets/train/'

"""
Datasets and Dataloades
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_db = TripletDataset(train_path, transform)
train_loader = DataLoader(train_db, batch_size=64, shuffle=True)

"""
Define model
"""
model = Network()
model.to(device)
model.load_state_dict(torch.load(f'weights/given-conf-{conf}.pth', map_location=device))

"""
loss function
"""
if conf == 1: # only reconstruction loss
    losses_dict = {'rec_loss':1}
    lr = 1e-2
elif conf == 2: # reconstruction + biderectional loss
    losses_dict = {'rec_loss':1, 'bidir_rec_loss':1}
    lr = 1e-2
elif conf == 3: # reconstruction + biderectional loss + feature loss
    losses_dict = {'rec_loss':1, 'bidir_rec_loss':1, 'feature_loss':1}
    lr = 1e-3
else:
    raise AttributeError('Invalid configuration!')

loss_fn = VFILoss(losses_dict=losses_dict, device=device)

"""
Optimizer
"""
optimizer = AdamW(model.parameters(), lr=lr)
"""
Training loop
"""
#logger = SummaryWriter(os.path.join("runs", f'conf:{conf}'))
os.makedirs('runs', exist_ok=True)
os.makedirs('weights', exist_ok=True)
for epoch in tqdm(range(NUM_EPOCHS), desc=f'Training for conf {conf}'):
    model.train()
    total_train_loss = 0
    for f1, f2, f3 in tqdm(train_loader):
        # TASK 2: Implement the training loop
        f1, f2, f3 = f1.to(device), f2.to(device), f3.to(device)
        # forward pass
        f2_pred = model(f1, f3)
        
        batch_loss = loss_fn(f2_pred, f2)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_train_loss += batch_loss.item()
    
    total_train_loss /= len(train_loader)
    #logger.add_scalar('Train loss',total_train_loss, global_step=epoch+1)
torch.save(model.state_dict(), os.path.join('weights', f'conf-{conf}.pth'))