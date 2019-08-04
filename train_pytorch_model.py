# Imports ----------------------------------------------------------------------

import os
import time
from collections import OrderedDict
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Data transforms and loaders --------------------------------------------------

# Data directories
image_folder = "images_pytorch"
dir_train = os.path.join(image_folder, "train")
dir_valid = os.path.join(image_folder, "valid")
dir_test = os.path.join(image_folder, "test")

# Transforms for training, validation, testing sets
train_transform = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Datasets
data_train = datasets.ImageFolder(dir_train, transform=train_transform)
data_valid = datasets.ImageFolder(dir_valid, transform=valid_transform)
data_test = datasets.ImageFolder(dir_test, transform=test_transform)

# Dataloaders
batch_size = 16
trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validloader = DataLoader(data_valid, batch_size=batch_size, shuffle= True)
testloader = DataLoader(data_test, batch_size=batch_size)

# Number of batches and classe
batches = len(trainloader.batch_sampler)
num_classes = len(data_train.classes)   # 102 classes

# Check data
print("Images in training / validation / test data: {} / {} / {}".format(
    len(trainloader.dataset), len(validloader.dataset), len(testloader.dataset)))
print("Training / validation / test batches: {} / {} / {}".format(
    len(trainloader), len(validloader), len(testloader)))   
print("Classes:", num_classes)


# Network architecture ---------------------------------------------------------

# Load the pre-trained VGG16 model from torchvision
model = models.vgg16(pretrained = True)

# Freeze parameters to prevent backpropagation in the "features" part of the model
for param in model.parameters():
    param.requires_grad = False

# Replace the "classifier" part of the network with a custom one
input_layer = model.classifier[0].in_features   # nodes in the last "features" layer
hidden_layer = [4096, 1024]
output_layer = num_classes
classifier = nn.Sequential(OrderedDict([
    ("0", nn.Linear(input_layer, hidden_layer[0])),
    ("1", nn.ReLU()),
    ("2", nn.Dropout(p=0.25)),
    ("3", nn.Linear(hidden_layer[0], hidden_layer[1])),
    ("4", nn.ReLU()),
    ("5", nn.Dropout(p=0.25)),
    ("6", nn.Linear(hidden_layer[1], output_layer)),
    ("7", nn.LogSoftmax(dim = 1))
]))
model.classifier = classifier


# Training parameters ----------------------------------------------------------

# Define device and send model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

# Training parameters/hyperparameters 
num_epochs = 10
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Print out training progress every {print_every} batches
print_every = 100 


# Training ---------------------------------------------------------------------

# Printouts and visualization
start_time = time.time() 
running_total = 0
running_loss = 0
running_correct = 0
loss_list = [] 
print("Start of training - Device: {} - Epochs: {} - Batches: {} - Batch size: {}"
      .format(device, num_epochs, len(trainloader), batch_size))

# Set model to training mode
model.train()

# Reset gradients
model.zero_grad()

for epoch in range(num_epochs):       
    for i, (images, labels) in enumerate(trainloader):      
        
        # Send data to device
        images, labels = images.to(device), labels.to(device)        
        
        # Forward and backward pass
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Printouts and visualization 
        
        # Store running loss, total predictions, correct predictions, loss
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()         
        loss_list.append(loss.item())
    
        # Print out average training loss and accuracy every {print_every} batches
        if (i+1) % print_every == 0:         
            print("Epoch: {}/{} - TRAINING DATA - Batches: {}/{} - Loss: {:.3f} - Accuracy: {:.3f}".format(
                epoch+1, num_epochs, i+1, len(trainloader), 
                running_loss/print_every, 
                running_correct/running_total))
            
            # Reset running loss and accuracy
            running_loss = 0
            running_total = 0
            running_correct = 0
            
# Evaluate model on validation set

    valid_running_loss = 0
    valid_running_total = 0
    valid_running_correct = 0

    # Set model to evaluation mode
    model.eval()
    
    # Turn off gradients
    with torch.no_grad():
        for i, (images, labels) in enumerate(validloader):
            
            # Send data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)
            
            # Store running loss, total predictions, correct predictions
            valid_running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            valid_running_total += labels.size(0)
            valid_running_correct += (predicted == labels).sum().item()       
       
        print("Epoch: {}/{} - VALIDATION DATA - Loss: {:.3f} - Accuracy {:.3f}".format(
            epoch+1, num_epochs, 
            valid_running_loss/len(validloader),
            valid_running_correct/valid_running_total))
    
    # Set model back to training mode
    model.train()

print("Training complete - Total training time: {:.1f} minutes".format(
    (time.time() - start_time)/60))


# Model evaluation on test set--------------------------------------------------

running_loss = 0
labels_true = np.array([], dtype=int)
labels_pred = np.array([], dtype=int)

# Set model to evaluation mode
model.eval()

# Turn off gradients
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        
        # Send data to device
        images, labels = images.to(device), labels.to(device)
    
        # Forward pass
        output = model(images)
        loss = criterion(output, labels)
        
        # Store loss, true labels, predicted labels
        running_loss += loss.item()     
        _, predicted = torch.max(output.data, 1)      
        labels_true = np.append(labels_true, labels.cpu().numpy())
        labels_pred = np.append(labels_pred, predicted.cpu().numpy())

test_accuracy = np.equal(labels_pred, labels_true).mean()         
        
print("Evaluating network on {} images in test set".format(len(testloader.dataset)))
print("Loss: {:.3f} - Accuracy: {:.3f}".format(
    running_loss/len(testloader), test_accuracy))


# Save model checkpoint --------------------------------------------------------

# Dictionary of the complete model state
model_state = {
    "epoch": num_epochs,   # number of epochs already trained
    "state_dict": model.state_dict(),   # trained weights
    "optimizer_dict": optimizer.state_dict(),   # trained weights
    "classifier": classifier,   # customized fully connected layers
    "class_to_idx": data_train.class_to_idx    # mapping of labels to their indices
}

torch.save(model_state, "app/flower_classifier.pth")
print("Model checkpoint saved")