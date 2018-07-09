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

np.set_printoptions(suppress=True)

parser.add_argument('--json_map', type=str, default='cat_to_name.json')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save_path', type=str, default='checkpoint.pth')
parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--top_k', type=int, default=5)

args, _ = parser.parse_known_args()

json_map = args.json_map
device = args.device
save_path = args.save_path
image_path = args.image_path
arch = args.arch
top_k = args.top_k

if arch == 'vgg16':
    pretrained_model = models.vgg16(pretrained=True)
elif arch == 'vgg19':
    pretrained_model = models.vgg19(pretrained=True)
else:
    pretrained_model = models.vgg11(pretrained=True)

loaded_model = th.load_checkpoint(filepath=save_path,
                               pretrained_model=models.vgg16(pretrained=True))

image = th.process_image(image_path)

with open(json_map, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = th.predict(image_path, loaded_model, cat_to_name=cat_to_name, device=device, topk=top_k)

print('\n')
print('Top: {}'.format(top_k))
print('Probabilities: {}'.format(str(probs)))
print('Classes: {}'.format(str(classes)))
