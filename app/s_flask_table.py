# Imports
from flask_table import Table, Col


class TopScores(Table):
    '''Object for the entries to the corresponding flask table.'''
    class_ = Col("Species")
    score = Col("Probability")

class ImageEntries(Table):
    '''Object for the entries to the corresponding flask table.'''
    ImageID = Col("ImageID")
    ImageName = Col("Image")
    CreateDate = Col("CreateDate")

class ScoreEntries(Table):
    '''Object for the entries to the corresponding flask table.'''
    ImageID = Col("ImageID")
    Classifier = Col("Classifier")
    ClassifierType = Col("ClassifierType")
    Class = Col("Class")
    Score = Col("Score")
    CreateDate = Col("CreateDate")


def gen_flask_topk_table(score_list, topk):
    '''Generates a flask table with top k classes and their scores.'''

    score_list.sort(key=lambda x: x["score"], reverse=True)
    score_list = score_list[0:topk]

    # Change key "class" to "class_" because "class" is a reserved expression 
    score_list_copy = [d.copy() for d in score_list]
    for i in range(len(score_list_copy)):
        score_list_copy[i]["class_"] = score_list_copy[i].pop("class") 

    # Round scores to 5 digits
    for i in range(len(score_list_copy)):
        score_list_copy[i]["score"] = round(score_list_copy[i]["score"], 5)

    # Populate the table
    topk_table = TopScores(score_list_copy)
    topk_table.border = True
    
    return topk_table


def gen_flask_image_table(image_object):
    '''Generates a flask table for the data written to the SQL table ImageTable.'''
    flask_image_table = ImageEntries([image_object])
    flask_image_table.border = True
    return flask_image_table

def gen_flask_score_table(score_object_list):
    '''Generates a flask table for the data written to the SQL table ScoreTable.'''
    flask_score_table = ScoreEntries(score_object_list)
    flask_score_table.border = True
    return flask_score_table