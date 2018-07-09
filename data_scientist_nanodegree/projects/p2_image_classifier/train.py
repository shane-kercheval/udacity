import sys
import argparse
import os
import terminal_helper as th

import json
import time

import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from collections import OrderedDict
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, default='checkpoint.pth')
parser.add_argument('--data_dir', type=str, default='flowers')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units1', type=float, default=12544)
parser.add_argument('--hidden_units2', type=float, default=6272)
parser.add_argument('--drop1', type=float, default=0.5)
parser.add_argument('--drop2', type=float, default=None)
parser.add_argument('--arch', type=str, default='vgg16')


args, _ = parser.parse_known_args()

save_path = args.save_path
data_dir = args.data_dir
device = args.device
epochs = args.epochs

learning_rate = args.learning_rate
hidden_units1 = args.hidden_units1
hidden_units2 = args.hidden_units2
drop1 = args.drop1
drop2 = args.drop2
arch = args.arch

inner_layer1 = hidden_units1 if drop1 is None else [hidden_units1, drop1]
inner_layer2 = hidden_units2 if drop2 is None else [hidden_units2, drop2]
layer_sizes = [25088, inner_layer1, inner_layer2, 102]

# print(save_path)
# print(data_dir)
# print(device)
# print(epochs)
# print(learning_rate)
# print(hidden_units1)
# print(hidden_units2)
# print(drop1)
# print(drop2)
# print(arch)
# print(layer_sizes)

#data_dir = 'flowers'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

train_transforms = transforms.Compose([
    # resize 224x224
    transforms.Resize(224),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# print(cat_to_name)

if arch == 'vgg16':
    pretrained_model = models.vgg16(pretrained=True)
elif arch == 'vgg19':
    pretrained_model = models.vgg19(pretrained=True)
else:
    pretrained_model = models.vgg11(pretrained=True)

optimizer_algorithm = optim.Adam
model = th.create_model(pretrained_model, layer_sizes)
criterion = nn.NLLLoss()
optimizer = optimizer_algorithm(model.classifier.parameters(), lr=learning_rate)

th.do_deep_learning(model, train_loader, validation_loader, epochs, 40, criterion, optimizer, device)

# TODO: Save the checkpoint
checkpoint = {
              'layer_sizes': layer_sizes,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
             }

torch.save(checkpoint, save_path)