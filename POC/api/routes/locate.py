import os
import cv2
import numpy as np
from flask import request, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from utils.utils import (
    QuadTree, 
    hex_to_binary_vector, session
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
        vector = hex_to_binary_vector(quad_tree.root.phash)
        print("Searching for vector: ", vector)
        
        #LIMITATION!  For now only search for the root node of a quad tree, find the closest greater than some arbitary amount. 
        #query = "SELECT count(*), similarity_euclidean(hash_vec, ?) AS vec_score FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " WHERE path='' AND vec_score > 0.5 LIMIT 1;"
        query = "SELECT image_id, similarity_euclidean(hash_vec, ?), misc FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + "  ORDER BY hash_vec ANN OF ? LIMIT 10;"    
        
        prepared_stmt = session.prepare(query)
        results = session.execute(prepared_stmt,[vector, vector] )
    
        #Have found and located a match. 
        #Euclidian matching returns a 1.0 as a perfect match. 
        str_results = ""
        for document in results: 
            str_results += f"Found a match at distance of {document[1]} document misc details are : {document[2]} "
            #No match fo
            
            
        return {
            "results": str_results
            
         },200


        
