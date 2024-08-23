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
        csv = ""
        for document in results: 
            

            csv += f"{document['path']},{document['path'].count('-')},{document['x0']},{document['y0']},{document['x1']},{document['y1']},{document['width']},{document['height']},{document['hash_algorithm']},{document['perceptual_hash_hex']}\n"       
            print(f"retrieved {document['qt_ref']} {document['path']}")
            counter+=1
                        
        
        print(f"returned {counter} records") 

        if (counter == 0):
            return "qt not found", 404

        output_folder = current_app.config['OUTPUT_FOLDER']
        print("output_folder is " + output_folder)
        csv_file = f"csv_{qt_ref}.csv.qt"

        csv_path = os.path.join(output_folder, csv_file)
        print(f"Saving file to {csv_file}")
        
        try:
            with open(csv_path, 'w') as file:
                file.write(csv)
            print(f"Have saved file to {csv_file}")                                     
        except:
            return "Server issue writing to file", 500                         
        
        print("os.path.basename(csv_file) : " + os.path.basename(csv_file))
        print("os.path.basename(output_folder) : " + os.path.basename(output_folder))
        
        return {"csv_qt_file": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(csv_file), _external=True)},200
