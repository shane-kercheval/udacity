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

np.set_printoptions(suppress=True)

def create_classifier(layer_sizes):
    layers = OrderedDict()

    for index, value in enumerate(layer_sizes):
        layer_name = 'fc' + str(index + 1)
        #print((layer_name, value))

        if index == len(layer_sizes) - 1:  # if last index add softmax
            layers.update({'output': nn.LogSoftmax(dim=1)})
        else:
            # get next layer size; next item might be list
            current_size = value[0] if isinstance(value, list) else value

            next_value = layer_sizes[index + 1]
            next_size = layer_sizes[index + 1][0] if isinstance(next_value, list) else layer_sizes[index + 1]

            layers.update({layer_name: nn.Linear(current_size, next_size)})

            if index < len(layer_sizes) - 2:  # if second to last index, don't add relu
                layers.update({'relu' + str(index + 1): nn.ReLU()})

                if isinstance(value, list):  # add dropout
                    layers.update({'dropout' + str(index + 1): nn.Dropout(p=value[1])})

    return nn.Sequential(layers)

def create_model(pretrained_model, layer_sizes):
    # Freeze parameters so we don't backprop through them
    for param in pretrained_model.parameters():
        param.requires_grad = False

    classifier = create_classifier(layer_sizes)
    pretrained_model.classifier = classifier
    
    return pretrained_model

# Implement a function for the validationpass
def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for inputs, labels in loader:

        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def do_deep_learning(model, trainloader, validation_loader, epochs, print_every, criterion, optimizer, device):
    steps = 0
    running_loss = 0
    
    print('Device: `{}`'.format(device))
    model.to(device)

    for e in range(epochs):
        # https://classroom.udacity.com/nanodegrees/nd025/parts/55eca560-1498-4446-8ab5-65c52c660d0d/modules/627e46ca-85de-4830-b2d6-5471241d5526/lessons/e1eeafe1-2ba0-4f3d-97a0-82cbd844fdfc/concepts/43cb782f-2d8c-432e-94ef-cb8068f26042
        # PyTorch allows you to set a model in "training" or "evaluation" modes with model.train() and model.eval(), respectively. In training mode, dropout is turned on, while in evaluation mode, dropout is turned off.
        model.train() 

        for ii, (inputs, labels) in enumerate(trainloader):
            start = time.time()
            
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validation_loader, criterion,  device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)),
                      "\n\tTime per batch: {0} seconds".format(round(time.time() - start))
                     )

                running_loss = 0

                # Make sure training is back on
                model.train()

def check_accuracy_on_test(model, loader, device):
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))

def load_checkpoint(filepath, pretrained_model):
    checkpoint = torch.load(filepath)
    
    
    model = create_model(models.vgg16(pretrained=True), checkpoint['layer_sizes'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    width, height = img.size
    
    aspect_ratio = width / height
    short_side = 256

    # First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the thumbnail or resize methods.
    if width > height:
        # width, height
        # if width is greater than height then shortest side is height; change height to 256, adjust width
        # width should be 256 (i.e. height) multiplied by the same aspect ratio
        img.thumbnail((short_side * aspect_ratio, short_side))  # width > height
    else:
        img.thumbnail((short_side, short_side * aspect_ratio))  # width <= height

    # Then you'll need to crop out the center 224x224 portion of the image.
    width, height = img.size
    new_width = 224
    new_height = new_width

    left_margin = (img.width - new_width) / 2
    bottom_margin = (img.height - new_height) / 2
    right_margin = left_margin + new_width
    top_margin = bottom_margin + new_height
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # the network expects the images to be normalized in a specific way. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations  [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel, then divide by the standard deviation.
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, cat_to_name, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image= transforms.ToTensor()(image)
    image = image.view(1, 3, 224, 224)
    #image.to(device)
    #model.to(device)
    model.to(device)
    with torch.no_grad():
        output = model.forward(image.type(torch.FloatTensor).to(device))
    probabilities = torch.exp(output).cpu()  # used LogSoftmax so convert back
    top_probs, top_classes = probabilities.topk(topk)
    return top_probs.numpy()[0], [cat_to_name[str(cls)] for cls in top_classes.numpy()[0]]
