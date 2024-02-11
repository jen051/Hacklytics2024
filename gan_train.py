import re
import os
import sys
import pickle
sys.path.append("classes")
from pin import Pin
from delayobject import DelayObject
from node import Node
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from constants import *
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
from edge_generator import EdgeGenerator
from matrix_generator import MatrixGenerator
from combined_generator import CombinedGenerator
from discriminator import Discriminator

edge_generator = EdgeGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_EDGE_GEN)
matrix_generator = MatrixGenerator(INPUT_SIZE_GEN, HIDDEN_SIZE_GEN, OUTPUT_SIZE_MAT_GEN)
discriminator = Discriminator(INPUT_SIZE_DISCRIM, HIDDEN_SIZE_DISCRIM, OUTPUT_SIZE_DISCRIM)
combined = CombinedGenerator(edge_generator,matrix_generator)

with open("filtered_dataset.pickle",'rb') as file:
    dataset = pickle.load(file)

new_dataset = []
for data in dataset:
    if len(data.adj) == 16:
        new_dataset.append(torch.cat((data.x,data.adj),dim=1))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Fit the scaler to the data and transform it
for data in dataset:
    data.x = scaler.fit_transform(data.x)

for data in dataset:
    data.x = torch.tensor(data.x, dtype=torch.float32)
    data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)  # Corrected dtype
    data.y = torch.tensor(data.y, dtype=torch.float32)


train_ratio = 0.8
num_train = int(len(new_dataset) * train_ratio)
num_test = len(new_dataset) - num_train

# Using random_split to create training and testing datasets
train_dataset, test_dataset = random_split(new_dataset, [num_train, num_test])

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=2)

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.01)
combined_optimizer = optim.Adam(combined.parameters(), lr=0.001)

criterion = nn.MSELoss() 
device = torch.device("xpu" if torch.cuda.is_available() else "cpu")
num_epochs = 1000
iters = 0

G_losses = []
D_losses = []

def get_fake_data(batch_size, combined):
    fake_graphs=[]
    for i in range(batch_size):
        rand_noise = torch.randn(1, INPUT_SIZE_GEN)
        fake_graphs.append(combined(rand_noise))
    return fake_graphs 
    
for epoch in range(num_epochs):

    for i, batch in enumerate(train_loader,0):
        #DISCRIM TRAINING W/ REAL DATASET
        batch = batch.to(device)

        batch_flattened = batch.view(batch.size(0), -1) 

        discriminator.zero_grad()
        outputs = discriminator(batch_flattened).view(-1)

        label = torch.ones(64).float()

        errD_real = criterion(outputs,label)
        
        errD_real.backward()
        D_x = outputs.mean().item()

        #DISCRIM TRAINING W/ FAKE DATASET
        
        fake_data = get_fake_data(64, combined)
        tensor_batch = torch.stack(fake_data) 

        tensor_batch_flattened = tensor_batch.view(tensor_batch.size(0), -1) 

        label.fill_(0)
        outputs = discriminator(tensor_batch_flattened.detach()).view(-1)

        errD_fake = criterion(outputs, label)
        errD_fake.backward()

        D_G_z1 = outputs.mean().item()
        
        errD = errD_real + errD_fake
        discriminator_optimizer.step()

        #GENERATOR TRAINING
        
        combined.zero_grad()
        label.fill_(1)
        outputs = discriminator(tensor_batch_flattened).view(-1)
        errG = criterion(outputs, label)
        errG.backward()
        D_G_z2 = outputs.mean().item()

        combined_optimizer.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

ay = get_fake_data(64, combined)

def deconstructor(matrix):
    mat1 = matrix[:, :3]  # Columns 0 to 2
    mat2 = matrix[:, 3:]  # The rest of the columns
    return mat1, mat2

node_features, adj = deconstructor(ay[0])
edge_list = torch.nonzero(adj, as_tuple=False).t()
data_obj = Data(x=node_features, y = 0, edge_index = edge_list)

torch.save(combined.state_dict(), "model.pth")