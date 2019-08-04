# Flowers

Author: Christoph Schauer <br>
Created: 04.08.2019 <br>
Last Update: 04.08.2019

## Introduction

This repository contains a small web app written in Python/Flask for uploading images of flowers, predicting their species with an image classification model, and storing the model output in a SQL database. In addition, it contains scripts for training the models and setting up the database.
This is just a little project for practicing Flask, SQLAlchemy, PyTorch, IBM's Watson Visual Recognition service, and combining all of that into one single application.


## Flask app

After being uploaded, the app sends the image to either a locally deployed PyTorch model or to a custom IBM Watson Visual Recognition model in the cloud for classification. The PyTorch model uses a pre-trained convolutional neural network (VGG16) with the last set of layers being customized and trained for the task at hand: Recognizing 102 species of flowers.
The dataset used for training is the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) by Maria-Elena Nilsback and Andrew Zisserman from the University of Oxford. The dataset contains about 8000 images of 102 species of flowers. After training for just 10 epochs and without any tuning, the model achieves a classification accuracy of about 80% on the test set.
The custom IBM Watson Visual Recognition model is trained on subset of this dataset containing images for only 20 of the 102 species because the Lite Subscription for the IBM Cloud has certain usage limitations. While the Watson Visual Recognition model is pretty much a black box, it probably works in a similar way.
After the image is classified, the predictions of the model are displayed to the user. The results are then stored in a SQL database - I've been using Microsoft SQL Server here - with SQLAlchemy.


## Description of files

<i>/root/app></i>

<i>app.py</i>: Contains the code for running the Flask app.

<i>s_pytorch</i>: Objects and functions used in the /classify_pytorch route

<i>s_watson_vr</i>: Objects and functions used in the /classify_watson route

<i>s_sqlalchemy</i>: Objects and functions used in the /results_stored route ("IS THAT REALLY THE NAME?")

<i>s_flask_table</i>: Objects/functions used for displaying flask tables

<i>watson_credentials_dummy.py</i>: The API key for the IBM Watson Visual Recognition instance used in the Flask app. 'APIKEY' needs to be replaced with the real API key and the file needs to be renamed to watson_credentials.py.

<i>sqlserver_credentials_dummy.py</i>: Contains the information necessary for connecting to a Microsoft SQL Server database through SQLAlchemy. The value for "password" needs to be replaced with the password given to this login/user and the value for "server" with your local IP address and the port you opened up for your SQL database (where applicable). The file needs to be renamed to sqlserver_credentials.py.

<i>label_mapping.json</i>: Maps the labels used in the PyTorch model, where the classes are labeled with numbers ranging from 1 to 102 based on the folders that contain the images to the actual names of the flower species.  


<i>/root</i>

<i>dependencies.txt</i>: The Python version and all packages required to run the code.

<i>train_pytorch_model.py</i>: The code for training and saving the PyTorch model used in the app.

<i>train_watson_vr_model.py</i>: The code for training the custom IBM Watson Visual Recognition model used app.

<i>create_database_user_tables_dummy.sql</i>: The SQL statements for setting up the database and tables in Microsoft SQL Server to which the app writes the data as well the login and user which the app uses to connect to the database.

<i>flowers.ipynb</i>: This notebook contains the code for training and saving the PyTorch model used in the app; this part is identical to train_pytorch_model.py. I also evaluate the performance of the trained model and write functions for classifying new images and visualizing these predictions. The notebook evolved from my course project for the module Deep Learning of [Udacity's Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) that I did some time ago.

<i>generate_flowers_image.py</i>: Contains the code for generating the illustration in the beginning of flowers.ipynb.

The folders containing the images for training the PyTorch and IBM Watson Visual Recognition models are not included as they files are too big to upload on Github.
