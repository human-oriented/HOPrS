import os
import cv2
import numpy as np
from flask import request, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from utils.utils import (
    QuadTree, mark_as_removed,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    Matched, hex_to_binary_vector, session, retrieve_quadtree,
    mark_as_matched, hamming_distance
)
from utils.utils import (
    session, ASTRA_DB_KEYSPACE, ASTRA_DB_TABLE
)
from cassandra.query import BatchStatement, ConsistencyLevel


def search_images(image, threshold = 5, compare_depth = 5):
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
        
        #Retrieve all the hamming distances and iterate through.  NOT using a vector search.  This is a brute force search.
        query = "SELECT image_id, hash, misc FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " where path=''  ALLOW FILTERING; "    
        print(f"query {query}")
        prepared_stmt = session.prepare(query)
        results = session.execute(prepared_stmt )
    
        lowest_hamming = 99999 #Higher than the highest possible hamming distance
        image_id = None
        lowest_misc = "None found yet"
        count = 0
        #Not trusting the similarty metric        #Have found and located a match. 
        for document in results: 
            count+=1
            ham = hamming_distance(document[1], quad_tree.root.phash)

            
            print(f"{count} hamming_distance:  {ham} image_id: {document[0]}")
            
            if(ham < lowest_hamming):
                lowest_hamming = ham
                image_id = document[0]
                lowest_misc = document[2]
                print(f'**Found a better match at hamming distance of {document[1]} document misc details are : {lowest_misc} ')

        
        print(f"Chosen a match at lowest_hamming distance of {lowest_hamming} document misc details are : {lowest_misc} ")
        
        new_image_filename = secure_filename(new_image_file.filename)
        output_folder = os.path.join(current_app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
        os.makedirs(output_folder, exist_ok=True)

        #Load the QT into memory
        db_quadtree_rootnode = retrieve_quadtree(image_id) 
        if db_quadtree_rootnode is None:
            return "Error loading qt from DB. ", 400

        height_1, width_1 = uploaded_image.shape[:2]
        image_difference_mask = np.zeros((height_1, width_1, 3), np.uint8)
        image_difference_mask[:] = (255,255,255)

        pixel_counter = 0
        list_pixel_counter = [pixel_counter]
        list_images = [uploaded_image, image_difference_mask]

        compare_depth = int(request.form.get('compare_depth', 5))
        
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
        current_app.logger.debug("Finished comparison")
        difference_mask_path = os.path.join(output_folder, "difference_mask.png")
        cv2.imwrite(difference_mask_path, list_images[1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        highlight_image_path = os.path.join(output_folder, "highlighted_image.png")
        create_red_overlay(filepath, difference_mask_path, highlight_image_path, translucence=50)

        unchanged_pixels = count_black_pixels(list_images[1])        

        return {
            "difference_mask": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(difference_mask_path), _external=True),
            "highlight_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(highlight_image_path), _external=True),
            "unchanged pixels": unchanged_pixels
         },200


        
