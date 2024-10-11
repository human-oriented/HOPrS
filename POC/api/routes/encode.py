import os
import cv2
import json
import datetime
from flask import url_for, jsonify, current_app
from pathlib import Path


from utils.utils import QuadTree, convert_heic


def encode_image(filepath, depth, algorithm, resize, crop, note, file, output_folder):
    root, ext = os.path.splitext(file.filename.lower())
    image = None
    if ext == '.heic':
        image_array = convert_heic(file)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else: 
        image = cv2.imread(filepath)
    
    if image is None:
            raise ValueError("Error opening image. Please check the file.")
    
    
    filename = Path(filepath).name
    #Our route to output involves everything in a subdir - let's call this subdir output (in tth output dir) 
    os.makedirs(os.path.join(output_folder,'output'), exist_ok=True)

    filepath_dot_hoprs = os.path.join(output_folder,'output', filename + ".hoprs")
    filepath_dot_qt = os.path.join(output_folder, 'output', filename + ".qt")
    print(f"filepath_dot_qt {filepath_dot_qt}")


    orig_x, orig_y = resize if resize else (0, 0)
    x0, y0, x1, y1 = crop if crop else (0, 0, 0, 0)
    now = datetime.datetime.now()
    
    with open(filepath_dot_hoprs, "w") as filehandle_dot_hoprs:
        current_time = now.strftime('%Y-%m-%d %H:%M:%SZ')
        data = {
            "When": current_time,
            "What": "ENCODED",
            "orig_width": str(orig_x),
            "orig_height": str(orig_y),
            "x0": str(x0),
            "y0": str(y0),
            "x1": str(x1),
            "y1": str(y1),
            "Format": "PNG",
            "Image_file": filepath,
            "QT_file": filepath_dot_qt,
            "Encoded_depth": str(depth),
            "Perceptual_algorithm": algorithm.upper(),
            "Comment": note
        }
        json.dump(data, filehandle_dot_hoprs, indent=4)

    # Assuming QuadTree and its methods are defined elsewhere
    quad_tree = QuadTree(image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, f"filepath: {filepath} time: {current_time} depth: {depth} note: { note}")
    quad_tree.print_tree(open(filepath_dot_qt, "w"))
    quad_tree.write_to_astra_db() #Consider what to do with other versions of this image already in the database. Skip this or not?



    return {
            "fingerprint_file": url_for('get_file', folder=os.path.basename(output_folder), filename=Path(filepath_dot_qt).name, _external=True)
     }


    return filepath_dot_qt
