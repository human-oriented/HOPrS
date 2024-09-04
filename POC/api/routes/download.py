#Downloads a QT file from database by qt_ref 

import os
from flask import  url_for, current_app
from utils.utils import (
    retrieve_quadtree_to_csv
)


def download_qt(qt_ref:str):
    csv = retrieve_quadtree_to_csv(qt_ref)
    
    if (csv == ""):
        return "qt not found", 404

    output_folder = current_app.config['OUTPUT_FOLDER']
    print("output_folder is " + output_folder)
    csv_file = os.path.basename(f"csv_{qt_ref}.csv.qt")

    csv_path = os.path.join(output_folder, csv_file)
    print(f"Saving file to {csv_path}")
    
    try:
        with open(csv_path, 'w') as file:
            file.write(csv)
        print(f"Have saved file to {csv_path}")
    except:
        return "Server issue writing to file", 500
    
#Try and figure out how to assemble a URL to the file.  
    print("csv_file : " + csv_file)
    print("output_folder : " + output_folder)
    print("os.path.basename(csv_file) : " + os.path.basename(csv_file))
    print("os.path.basename(output_folder) : " + os.path.basename(output_folder))
    print(f"URLs: {url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(csv_file), _external=True)}")

    return {"csv_qt_file": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(csv_file), _external=True)},200
