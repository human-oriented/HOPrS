import os
import cv2
import json
from datetime import datetime, timezone
from utils.utils import QuadTree, convert_heic
from cassandra.query import BatchStatement, SimpleStatement,ConsistencyLevel


def encode_image(filepath, depth, algorithm, resize, crop, note, file):
    root, ext = os.path.splitext(file.filename.lower())
    image = None
    if ext == '.heic':
        image_array = convert_heic(file)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else: 
        image = cv2.imread(filepath)
    
    if image is None:
            raise ValueError("Error opening image. Please check the file.")
    
    filename_dot_hoprs = filepath + ".hoprs"
    filename_dot_qt = filepath + ".qt"

    orig_x, orig_y = resize if resize else (0, 0)
    x0, y0, x1, y1 = crop if crop else (0, 0, 0, 0)
    
    with open(filename_dot_hoprs, "w") as filehandle_dot_hoprs:
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
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
            "QT_file": filename_dot_qt,
            "Encoded_depth": str(depth),
            "Perceptual_algorithm": algorithm.upper(),
            "Comment": note
        }
        json.dump(data, filehandle_dot_hoprs, indent=4)

    # Assuming QuadTree and its methods are defined elsewhere
    quad_tree = QuadTree(image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, str(filepath) + " " + str(current_time))
    quad_tree.print_tree(open(filename_dot_qt, "w"))
    quad_tree.write_to_astra_db()

    return filename_dot_qt
