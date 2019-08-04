
# Imports
import os
import json
from ibm_watson import VisualRecognitionV3


def watson_classify(filename, directory, classifier_id):
    '''Classifies an image with a custom IBM Watson Visual Recognition model
    with classifier_ids = classifer_id. Returns a dictionary with the name of 
    the classified image, the model name, an (arbitrary) classifier_type label
    for the model, and a list of classes and their scores.'''

    # Load apikey for the IBM Watson VR instance 
    with open("watson_credentials.json", "r") as f:
        watson_credentials = json.load(f)
    apikey = watson_credentials["apikey"]

    # Instantiate the Watson VR service
    visual_recognition = VisualRecognitionV3(
        version='2018-03-19', iam_apikey=apikey)

    # Classify
    with open(os.path.join(directory, filename), "rb") as images_file:
        result = visual_recognition.classify(
            images_file, threshold="0.0", classifier_ids=classifier_id
        ).get_result()

    # Wrap everything into a dictionary
    watson_output = {
        "image": filename,
        "classifier_type": "Custom IBM Watson Visual Recognition Model",
        "classifier": result["images"][0]["classifiers"][0]["name"],
        "classes": result["images"][0]["classifiers"][0]["classes"]
        }   

    return watson_output

