import os
import cv2
import json
from flask import request, render_template, send_file, send_from_directory, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
from utils import (
    QuadTree
)
from . import upload_bp


#This code is called when an encode of a single file is requested


@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    print(f"upload 1")

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    print(f"upload 0")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        depth = int(request.form.get('depth', 5))
        algorithm = request.form.get('algorithm', 'pdq')
        resize = request.form.get('resize', None)
        crop = request.form.get('crop', None)
        note = request.form.get('note', "Need a meaningful comment in here at some point")
        print(f"upload 1")
        if resize:
            resize = tuple(map(int, resize.split(',')))

        if crop:
            crop = tuple(map(int, crop.split(',')))

        try:
            image = cv2.imread(filepath)
            if image is None:
                return "Error opening image. Please check the file.", 400

            filename_dot_hoprs = filepath + ".hoprs"
            filename_dot_qt = filepath + ".qt"
            print(f"upload 2")

            orig_x, orig_y = resize if resize else (0, 0)
            x0, y0, x1, y1 = crop if crop else (0, 0, 0, 0)
#the .hoprs file may not be needed at all in the future.  It was intended to represent metadata such as the crop coordinates. 
            with open(filename_dot_hoprs, "w") as filehandle_dot_hoprs:
                from datetime import datetime, timezone
                
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
            print("1")
            quad_tree = QuadTree(image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, str(file.filename)+" "+str(current_time))
            print("2")
            quad_tree.print_tree(open(filename_dot_qt, "w"))
            print("3")
            quad_tree.write_to_astra_db(None, 0, '')
            print("4")

            return send_file(filename_dot_qt, as_attachment=True)

        except Exception as e:
            return str(e), 500
