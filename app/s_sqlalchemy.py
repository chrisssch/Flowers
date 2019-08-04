# Imports
from flask_sqlalchemy import SQLAlchemy
import json

# Initialize database
db = SQLAlchemy()


def gen_connection_string(credentials_file):
    '''Generates a connection string from a json file with credentials.'''
    
    with open(credentials_file, "r") as f:
        credentials = json.load(f)

    user = credentials["user"]
    pw = credentials["password"]
    server = credentials["server"]
    db = credentials["database"]
    driver = credentials["driver"]
    connection_string = "mssql+pyodbc://"+user+":"+pw+"@"+server+"/"+db+"?driver="+driver
    
    return connection_string


class ImageObject(db.Model):
    '''ImageObject Class/Model.'''

    __tablename__ = "ImageTable"
    ImageID = db.Column(db.BigInteger, primary_key=True)
    ImageName = db.Column(db.String)
    CreateDate = db.Column(db.DateTime)
    
    def __init__(self, ImageName, CreateDate):
        # ImageID: Auto-generated
        self.ImageName = ImageName
        self.CreateDate = CreateDate


class ScoreObject(db.Model):
    '''ScoresObject Class/Model.'''

    __tablename__ = "ScoreTable"
    RowID = db.Column(db.BigInteger, primary_key=True)
    ImageID = db.Column(db.BigInteger)
    Classifier = db.Column(db.String)
    ClassifierType = db.Column(db.String)
    Class = db.Column(db.String)
    Score = db.Column(db.Float)
    CreateDate = db.Column(db.DateTime)
    
    def __init__(self, ImageID, Classifier, ClassifierType, Class, Score, CreateDate):
        # RowID: Auto-generated
        self.ImageID = ImageID
        self.Classifier = Classifier
        self.ClassifierType = ClassifierType
        self.Class = Class
        self.Score = Score
        self.CreateDate = CreateDate

##########----------##########----------##########----------##########----------


def gen_score_objects(model_output, ImageID, CreateDate, topk):
    '''Gets the top k classes and their scores from the model output and 
    converts them into a list of SQLAlchemy objects to write to the 
    ScoreTable SQL table.'''

    Classifier = model_output["classifier"]
    ClassifierType = model_output["classifier_type"]

    dict_list = model_output["classes"]
    dict_list.sort(key=lambda x: x["score"], reverse=True)   
    dict_list = dict_list[0:topk]

    new_object_list = []
    for j in range(len(dict_list)):
        class_j = dict_list[j]["class"]
        score_j = dict_list[j]["score"]
        object_j = ScoreObject(
            ImageID, Classifier, ClassifierType, class_j, score_j, CreateDate)
        new_object_list.append(object_j) 
    
    return new_object_list