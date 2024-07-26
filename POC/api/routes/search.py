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

def search_images(image):
    new_image_file = image
    if new_image_file.filename == '':
        return "No selected file", 400

    if new_image_file:
        filename = secure_filename(new_image_file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        new_image_file.save(filepath)
        
        uploaded_image = cv2.imread(filepath)
        
        if uploaded_image is None:
            return "Error opening image. Please check the file.", 400
        current_app.logger.debug("ab")
        orig_x, orig_y =  (0, 0)
        x0, y0, x1, y1 =  (0, 0, 0, 0)
        depth = int(request.form.get('compare_depth', 5))
        algorithm = 'pdq'
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
        
        quad_tree = QuadTree(uploaded_image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, str(new_image_file.filename)+" "+str(current_time))
        current_app.logger.debug("bc")
        vector = hex_to_binary_vector(quad_tree.root.phash)
        
        sort = {"$vector": vector}
        results = collection.find(sort=sort, limit=10, include_similarity=True) #For now only return 1 result
        
        new_image_filename = secure_filename(new_image_file.filename)
        output_folder = os.path.join(current_app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
        os.makedirs(output_folder, exist_ok=True)

        counter=0
        data = []
        for document in results:
            #TODO RELOAD THE IMAGE SO IT ISNT DRAWN ALL OVER
            # current_app.logger.debug("about to retr qt")
            db_quadtree_rootnode = retrieve_quadtree(document["qt_ref"]) 
            # current_app.logger.debug("fg")
            height_1, width_1 = uploaded_image.shape[:2]
            image_difference_mask = np.zeros((height_1, width_1, 3), np.uint8)
            image_difference_mask[:] = (255,255,255)

            pixel_counter = 0
            list_pixel_counter = [pixel_counter]
            list_images = [uploaded_image, image_difference_mask]

            threshold = int(request.form.get('threshold', 10))
            compare_depth = int(request.form.get('compare_depth', 5))

            # current_app.logger.debug(f"gh compare {compare_depth} threshold {threshold} filepath {filepath} upload {current_app.config['UPLOAD_FOLDER']} output {current_app.config['OUTPUT_FOLDER']}")
            
            # #Reset the removed flag from the quad tree for the image hta thas been specified so that 
            # #we can do subsequent comparisons
            current_app.logger.debug("Starting setting quad_tree to removed=False, Matched as Unknown")
            mark_as_removed(quad_tree.root,False)
            mark_as_matched(quad_tree.root, Matched.UNKNOWN)
            current_app.logger.debug("Finished setting quad_tree  to removed=False, Matched as Unknown")
            current_app.logger.debug("Starting setting db_quadtree  to removed=False, Matched as Unknown")
            mark_as_removed(db_quadtree_rootnode,False)
            mark_as_matched(db_quadtree_rootnode, Matched.UNKNOWN)
            current_app.logger.debug("Finished setting db_quadtree  to removed=False, Matched as Unknown")
            
            compare_and_output_images(
                list_images, list_pixel_counter, 
                db_quadtree_rootnode, quad_tree.root, filepath, 
                output_folder, 
                threshold, [0], compare_depth)

            # current_app.logger.debug("jj")
            difference_mask_path = os.path.join(output_folder, "difference_mask"+str(counter)+".png")
            cv2.imwrite(difference_mask_path, list_images[1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            # current_app.logger.debug("hi")
            highlight_image_path = os.path.join(output_folder, "highlighted_image"+str(counter)+".png")
            create_red_overlay(filepath, difference_mask_path, highlight_image_path, translucence=50)

            # current_app.logger.debug(f"jk {highlight_image_path}")

            unchanged_pixels = count_black_pixels(list_images[1])
            url = url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(highlight_image_path), _external=True),

            data.append({
                "id": document["_id"],
                "image_url": url,
                "similarity": document["$similarity"],
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

            counter+=1
                
            return data, 200