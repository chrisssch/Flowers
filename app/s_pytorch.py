# Imports
import os
import json
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms


def load_checkpoint(saved_model):
    '''Loads a saved model checkpoint from a model_state dictionary.'''
    
    # Load model to CPU
    model_state = torch.load(saved_model, map_location=torch.device("cpu"))
    
    # Load architecture and weights
    model = models.vgg16(pretrained=True)
    model.classifier = model_state["classifier"]
    model.load_state_dict(model_state["state_dict"])
    model.class_to_idx = model_state["class_to_idx"]

    return model


def process_image(filename, directory):        
    '''Loads and transform an image for use in a torchvision model.'''
    
    # Define transforms
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Upload and transform image
    image = Image.open(os.path.join(directory, filename)).convert("RGB")
    image = image_transform(image)
    
    return image


def pytorch_classify(filename, directory, model, model_name):
    ''' Classifies an image with a PyTorch torchvision model and maps classes 
    (like flower species) from a dictionary to the model classes generated from 
    the image folder names. Returns a dictionary with the name of the classified 
    image, an (arbitrary) model name, an (arbitrary) classifier_type label for 
    the model, and a list of classes and their scores.'''

    # Load mapping of classes to species
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    # Load and transform an image
    image = process_image(filename, directory).unsqueeze(0)

    # Set model to evaluation mode
    model.eval()
    
    # Predict scores
    with torch.no_grad():
        output = model(image)

    # Get lists of species and their scores (ordered by the index of the classes)
    scores = output.exp().squeeze().tolist()   # exponential because of nllloss   
    classes = [label_mapping[k] for k in model.class_to_idx.keys()]
    score_list = [{"class": classes[i], "score": scores[i]} for i in range(len(classes))]

    # Wrap everything into a dictionary
    pytorch_output = {
        "image": filename,
        "classifier_type": "Local PyTorch Model",
        "classifier": model_name,
        "classes": score_list
        }
    
    return pytorch_output