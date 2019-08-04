# Imports

from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_table import Table, Col
import os
import json
from datetime import datetime

from PIL import Image
import torch
from torchvision import models, transforms
from ibm_watson import VisualRecognitionV3

import s_pytorch as spt
import s_watson_vr as swvr
import s_flask_table as sft
import s_sqlalchemy as ssa
from s_sqlalchemy import db


# Folder for uploading files and archive
upload_folder = "uploads"
file_archive = "image_archive"

# Function to check if an extension is allowed
allowed_extension = {"jpg", "jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extension

# Load the trained PyTorch flower_classifier model
model = spt.load_checkpoint(saved_model="flower_classifier.pth")
model_name = "flower_classifier"


# Initialize flask app
app = Flask(__name__)

# Generate SQL Server connection string from credentials
connection_string = ssa.gen_connection_string(credentials_file="sqlserver_credentials.json")

# Configure database (db)
app.config["SQLALCHEMY_DATABASE_URI"] = connection_string
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Register database (db from sqlalchemy_functions.py)
db.init_app(app)


# Upload image

@app.route("/")
def index():
    message = ""
    return render_template("upload_page.html", message=message)


# Display image

@app.route("/upload", methods=["POST"])
def upload_function():

    # Send user back if uploaded file is empty
    if "file" not in request.files:
        message = "Please select an image!"
        return render_template("upload_page.html", message=message)
    if request.files["file"].filename == "":
        message = "Please select an image!"
        return render_template("upload_page.html", message=message)

    file = request.files["file"]

    # Save file if extension is allowed
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(upload_folder, filename))
        return render_template("display_page.html", filename=filename)

    # Send user back to upload page if extension is not allowed
    else:
        message = "Only jpg and jpeg are allowed as extensions!"
        return render_template("upload_page.html", message=message)


# Display image - helper function

@app.route("/upload/<filename>")
def send_image_function(filename):
    return send_from_directory(upload_folder, filename)


# Classify image with a Pytorch model

@app.route("/classify_pytorch/<filename>", methods=["POST"])
def classify_pytorch_function(filename):

    # Classify image
    pytorch_output = spt.pytorch_classify(
        filename=filename, directory=upload_folder, model=model,
        model_name=model_name
    )

    # Save model output in file system
    with open(os.path.join(upload_folder, "model_output.json"), "w") as f:
        json.dump(pytorch_output, f)

    # Generate a flask table with top k classes and their scores
    flask_topk_table = sft.gen_flask_topk_table(
        score_list=pytorch_output["classes"], topk=5)

    return render_template(
        "result_page.html",
        filename=filename,
        flask_topk_table=flask_topk_table,
        classifier=pytorch_output["classifier"],
        classifier_type=pytorch_output["classifier_type"]
    )


# Classify image with a custom IBM Watson Visual Recognition model

@app.route("/classify_watson/<filename>", methods=["POST"])
def classify_watson_function(filename):

    # Predict scores with custom Watson VR model
    classifier_id = "flower_classifier_1864201408"
    watson_output = swvr.watson_classify(
        filename=filename, directory=upload_folder, classifier_id=classifier_id)

    # Save Watson output in file system
    with open(os.path.join(upload_folder, "model_output.json"), "w") as f:
            json.dump(watson_output, f)

    # Generate a flask table with top k classes and their scores
    flask_topk_table = sft.gen_flask_topk_table(
        score_list=watson_output["classes"], topk=5)

    return render_template(
        "result_page.html",
        filename=filename,
        flask_topk_table=flask_topk_table,
        classifier=watson_output["classifier"],
        classifier_type=watson_output["classifier_type"]
    )


# Write result to SQL database

@app.route("/results_stored/<filename>", methods=["POST", "GET"])  
def store_results_function(filename):

    with open(os.path.join(upload_folder, "model_output.json"), "r") as f:
        model_output = json.load(f)

    CreateDate = datetime.now()

    # write to ImageTable
    ImageName = model_output["image"]
    new_object = ssa.ImageObject(ImageName, CreateDate)   ### imported
    db.session.add(new_object)
    db.session.commit()

    # Write to ScoreTable
    new_object_list = ssa.gen_score_objects(
        model_output=model_output,
        ImageID=new_object.ImageID,
        CreateDate=CreateDate,
        topk=5)
    db.session.bulk_save_objects(new_object_list)
    db.session.commit()

    # Generate image table and classes table for website
    flask_image_table = sft.gen_flask_image_table(image_object=new_object)
    flask_score_table = sft.gen_flask_score_table(score_object_list=new_object_list)

    # Archive image and remove temporary files
    new_image_name = str(new_object.ImageID) + ".jpg"
    os.rename(
        os.path.join(upload_folder, filename),
        os.path.join(file_archive, new_image_name)
    )
    os.remove("uploads/model_output.json")

    return render_template(
        "store_results.html",
        flask_image_table=flask_image_table,
        flask_score_table=flask_score_table)


# Run flask app
if __name__ == "__main__":
    app.run(port=1111, debug=True)   # deactive debug when deploying the app
