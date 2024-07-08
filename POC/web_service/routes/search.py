import os
import cv2

import numpy as np
from flask import request, render_template, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from utils import (
    QuadTree, mark_as_removed,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    Matched, hex_to_binary_vector, collection,retrieve_quadtree,
    mark_as_matched
)
from . import search_bp

# Define debug_mode
debug_mode = False
@search_bp.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('search.html')
    else:
        try:
            new_image_file = request.files['new_image']
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
                
                result_html = '<html><body><table border="1">'
                result_html += '<tr><th>image</th><th>Vector Similarity</th><th>Row ID</th><th>qt_ref Reference</th><th>Path</th><th>Level</th><th>X0</th><th>Y0</th><th>X1</th><th>Y1</th><th>Width</th><th>Height</th><th>Hash Algorithm</th><th>Top level Perceptual Hash Hex</th><th>Quality</th></tr>'
                current_app.logger.debug("de")
                counter=0
                for document in results:
                    #TODO RELOAD THE IMAGE SO IT ISNT DRAWN ALL OVER
                    current_app.logger.debug("about to retr qt")
                    db_quadtree_rootnode = retrieve_quadtree(document["qt_ref"]) 
                    current_app.logger.debug("fg")
                    height_1, width_1 = uploaded_image.shape[:2]
                    image_difference_mask = np.zeros((height_1, width_1, 3), np.uint8)
                    image_difference_mask[:] = (255,255,255)

                    pixel_counter = 0
                    list_pixel_counter = [pixel_counter]
                    list_images = [uploaded_image, image_difference_mask]

                    threshold = int(request.form.get('threshold', 10))
                    compare_depth = int(request.form.get('compare_depth', 5))

                    current_app.logger.debug(f"gh compare {compare_depth} threshold {threshold} filepath {filepath} upload {current_app.config['UPLOAD_FOLDER']} output {current_app.config['OUTPUT_FOLDER']}")
                    
                    #Reset the removed flag from the quad tree for the image hta thas been specified so that 
                    #we can do subsequent comparisons
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
                        current_app.config['OUTPUT_FOLDER'], 
                        threshold, [0], compare_depth)

                    current_app.logger.debug("jj")
                    difference_mask_path = os.path.join(current_app.config['OUTPUT_FOLDER'], "difference_mask"+str(counter)+".png")
                    cv2.imwrite(difference_mask_path, list_images[1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    current_app.logger.debug("hi")
                    highlight_image_path = os.path.join(current_app.config['OUTPUT_FOLDER'], "highlighted_image"+str(counter)+".png")
                    create_red_overlay(filepath, difference_mask_path, highlight_image_path, translucence=50)

                    current_app.logger.debug(f"jk {highlight_image_path}")

                    unchanged_pixels = count_black_pixels(list_images[1])

                    current_app.logger.debug(f"highlight_image_path is {highlight_image_path} and url formatted {url_for('compare_bp.output_file', filename=highlight_image_path)}")
                    result_html += f'<tr>'
                    result_html += "<td><IMG SRC='"
                    result_html += url_for('compare_bp.output_file', filename=os.path.basename(highlight_image_path))
                    result_html += "' width=100px  ALT='one of the results'/></td>"
                    
                    result_html += f'<td>{document["$similarity"]}</td>'
                    result_html += f'<td>{document["_id"]}</td>'
                    result_html += f'<td>{document["qt_ref"]}</td>'
                    result_html += f'<td>{document["path"]}</td>'
                    result_html += f'<td>{document["level"]}</td>'
                    result_html += f'<td>{document["x0"]}</td>'
                    result_html += f'<td>{document["y0"]}</td>'
                    result_html += f'<td>{document["x1"]}</td>'
                    result_html += f'<td>{document["y1"]}</td>'
                    result_html += f'<td>{document["width"]}</td>'
                    result_html += f'<td>{document["height"]}</td>'
                    result_html += f'<td>{document["hash_algorithm"]}</td>'
                    result_html += f'<td>{document["perceptual_hash_hex"]}</td>'
                    result_html += f'<td>{document["quality"]}</td>'
                    result_html += f'</tr>'

                    
                    counter+=1
               
                result_html += '</table></body></html>'
                current_app.logger.debug(f"OUTPUT_FOLDER folder is {current_app.config['OUTPUT_FOLDER']}")
                
                return result_html, 200

        except Exception as e:
            return str(e), 500

