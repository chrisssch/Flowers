# Flowers

This repository contains (well, will soon contain) two somewhat related projects:

## flowers-pytorch
A jupyter notebook and Python scripts (the latter being WIP) where I train a convolutional neural network in PyTorch using a pre-trained model to predict the species of a flower from an image. The dataset used for training is the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) by Maria-Elena Nilsback and Andrew Zisserman from the University of Oxford. The dataset contains about 8000 images of 102 species of flowers. After training for just 10 epochs, the model achieves a classification accuracy of about 80% on the test set. 

## flowers-flask-watson
A flask web application that lets a user upload an image of a flower, sends the image to a custom IBM Watson Visual Recognition model for classification, returns the predicted species ot the user, and stores the data in a Microsoft SQL Server database. 
The custom IBM Watson classifier is trained on a subset of the categories of the same [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) because training it on the whole dataset would exceed the limit of my Lite subscription plan.
