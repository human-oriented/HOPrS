#Downloads a QT file from database by qt_ref 

import os
import cv2
import numpy as np
from flask import request, render_template, url_for, current_app, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from utils.utils import (
    QuadTree, mark_as_removed,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    Matched, hex_to_binary_vector, collection,retrieve_quadtree,
    mark_as_matched
)

def download_qt(qt_ref:str):
    
        print(f"About to quwery for {qt_ref}")
        match_criteria = {"qt_ref": qt_ref}
        sort = {"path": 1}
        results = collection.find(match_criteria, sort=sort, limit=100000) #For now only return 1 result
        print("Have a collection")


        counter=0
        data = []
        for document in results:            
            #TODO WRONG ORDER SORT THIS OUT
            data.append({
                "id": document["_id"],
                "qt_ref": document["qt_ref"],
                "path": document["path"],
                "level": document["level"],
                "x0": document["x0"],
                "y0": document["y0"],
                "x1": document["x1"],
                "y1": document["y1"],
                "width": document["width"],
                "height": document["height"],
                "hash_algorithm": document["hash_algorithm"],
                "perceptual_hash_hex": document["perceptual_hash_hex"],
            })
            print(f"retrieved {document['qt_ref']} {document['path']}")
            counter+=1
            
            
        #TODO Wrangle data into a CSV
        print(f"returned {counter} records") 
        if (counter == 0):
            return "qt not found", 404
        return data, 200
