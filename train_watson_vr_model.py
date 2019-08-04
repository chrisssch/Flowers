# Imports
import os
import json
from ibm_watson import VisualRecognitionV3

# Load IBM Watson VR instance apikey
with open(os.path.join("app", "watson_credentials.json"), "r") as f:
    watson_credentials = json.load(f)
apikey = watson_credentials["apikey"]

# Initialize 
visual_recognition = VisualRecognitionV3(version="2018-03-19", iam_apikey=apikey)

# Image folder and paths
image_folder = "images_watson_vr"
files = os.listdir(image_folder)
paths = [os.path.join(image_folder, files[i]) for i in range(len(files))]
filenames = [files[i].split(".")[0] for i in range(len(files))]

# Send images to IBM Watson Visual Recognition instance for training
with open(
    paths[0], "rb"
) as filenames[0], open(
    paths[1], "rb"
) as filenames[1], open(
    paths[2], "rb"
) as filenames[2], open(
    paths[3], "rb"
) as filenames[3], open(
    paths[4], "rb"
) as filenames[4], open(
    paths[5], "rb"
) as filenames[5], open(
    paths[6], "rb"
) as filenames[6], open(
    paths[7], "rb"
) as filenames[7], open(
    paths[8], "rb"
) as filenames[8], open(
    paths[9], "rb"
) as filenames[9], open(
    paths[10], "rb"
) as filenames[10], open(
    paths[11], "rb"
) as filenames[11], open(
    paths[12], "rb"
) as filenames[12], open(
    paths[13], "rb"
) as filenames[13], open(
    paths[14], "rb"
) as filenames[14], open(
    paths[15], "rb"
) as filenames[15], open(
    paths[16], "rb"
) as filenames[16], open(
    paths[17], "rb"
) as filenames[17], open(
    paths[18], "rb"
) as filenames[18], open(
    paths[19], "rb"
) as filenames[19]:   
    model = visual_recognition.create_classifier(
        name = "flower_classifier", 
        positive_examples = {
            "alpine sea holly": filenames[0], 
            "bird of paradise": filenames[1], 
            "buttercup": filenames[2],
            "columbine": filenames[3], 
            "corn poppy": filenames[4], 
            "daffodil": filenames[5], 
            "english marigold": filenames[6], 
            "foxglove": filenames[7], 
            "gazania": filenames[8], 
            "giant white arum lily": filenames[9],
            "globe thistle": filenames[10], 
            "japanese anemone": filenames[11], 
            "moon orchid": filenames[12],
            "oxeye daisy": filenames[13], 
            "passion flower": filenames[14], 
            "peruvian lily": filenames[15], 
            "rose": filenames[16], 
            "sunflower": filenames[17], 
            "thorn apple": filenames[18], 
            "watercress": filenames[19],   
        }
    ).get_result()

# Show model
print(json.dumps(model, indent=2))

# Get classifier ID
classifier_id = model["classifier_id"]
print(classifier_id)


# Test Model

image = "moon orchid.jpg"
# classifier_id = "flower_classifier_1864201408"

with open(image, "rb") as images_file:
    result = visual_recognition.classify(
        images_file,
        classifier_ids=classifier_id,
        threshold="0.0"   # returns all scores
    ).get_result()

print(json.dumps(result, indent=2))
