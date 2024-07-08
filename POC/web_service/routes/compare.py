import os
import cv2
import json
import shutil
import numpy as np
from flask import request, render_template, send_file, send_from_directory, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
from utils import (
    QuadTree, hamming_distance, mark_as_removed, draw_comparison,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    parse_file_to_tree, Matched
)
from . import compare_bp  # Import the Blueprint instance

# Define debug_mode
debug_mode = False

@compare_bp.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'GET':
        # Send back page to enter the values
        return render_template('compare.html')
    else:
        if 'original_image_qt' not in request.files or 'new_image' not in request.files:
            return "No file part", 400

        original_image_qt_file = request.files['original_image_qt']
        new_image_file = request.files['new_image']

        if original_image_qt_file.filename == '' or new_image_file.filename == '':
            return "No selected file", 400
        
        original_image_qt_filename = secure_filename(original_image_qt_file.filename)
        new_image_filename = secure_filename(new_image_file.filename)
        original_image_qt_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], original_image_qt_filename)
        new_image_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], new_image_filename)

        original_image_qt_file.save(original_image_qt_filepath)
        new_image_file.save(new_image_filepath)

        threshold = int(request.form.get('threshold', 10))
        compare_depth = int(request.form.get('compare_depth', 5))
        output_folder = os.path.join(current_app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
        os.makedirs(output_folder, exist_ok=True)

        original_image = new_image_filepath  # Reuse the uploaded new image as the base image
        new_image_qt_filepath = new_image_filepath + ".qt"

        try:
            image = cv2.imread(new_image_filepath)
            if image is None:
                return "Error opening new image. Please check the file.", 400

            image_hoprs = new_image_filepath + ".hoprs"
            with open(image_hoprs, "w") as file_hoprs:
                current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
                data = {
                    "When": current_time,
                    "What": "ENCODED",
                    "Format": "PNG",
                    "Image_file": new_image_filepath,
                    "QT_file": new_image_qt_filepath,
                    "Encoded_depth": str(compare_depth),
                    "Perceptual_algorithm": "PDQ",
                    "Comment": "Generated for comparison"
                }
                json.dump(data, file_hoprs, indent=4)
            
            height, width = image.shape[:2]
            
            quad_tree = QuadTree(image, 
                                 compare_depth, 
                                 orig_x=width,
                                 orig_y=height,
                                 x0=0, y0=0,
                                 x1=width, y1=height, 
                                 hash_algorithm='pdq', 
                                 unique_qt_reference=str(new_image_filename)+" "+str(current_time))
            
            quad_tree.print_tree(open(new_image_qt_filepath, "w"))
            
            def main(original_image, original_image_qt, new_image, new_image_qt, output_folder, threshold, compare_depth):
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                tree1 = parse_file_to_tree(original_image_qt)
                tree2 = parse_file_to_tree(new_image_qt)
                image = cv2.imread(original_image)  # Reuse the uploaded new image as the base image

                image_derivative = cv2.imread(new_image)
                height_1, width_1 = image.shape[:2]
                image_1 = np.zeros((height_1, width_1, 3), np.uint8)
                image_1[:] = (255, 255, 255)

                pixel_counter = 0
                list_pixel_counter = [pixel_counter]
                list_images = [image_derivative, image_1]

                compare_and_output_images(list_images, list_pixel_counter, tree1, tree2, original_image, output_folder, threshold, [0], compare_depth)

                difference_mask_path = os.path.join(output_folder, "difference_mask.png")
                cv2.imwrite(difference_mask_path, list_images[1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                highlight_image_path = os.path.join(output_folder, "highlighted_image.png")
                create_red_overlay(new_image, difference_mask_path, highlight_image_path)
                new_image_output_path = os.path.join(output_folder, os.path.basename(new_image))
                shutil.copy(new_image, new_image_output_path)

                unchanged_pixels = count_black_pixels(list_images[1])
                draw_comparison(list_images, list_pixel_counter, tree1, tree2, output_folder, [-1], threshold, compare_depth)

                tree1.purge_tree()
                tree1.optimise_tree()

                for file in os.listdir(output_folder):
                    if file.startswith("comparison_") and file.endswith(".jpg"):
                        comparison_image_path = os.path.join(output_folder, file)
                        break

                if comparison_image_path is None:
                    return "Comparison image not found.", 500, None, None

                height, width = image.shape[:2]
                image_pixels = width * height
                proportion = unchanged_pixels / image_pixels
                stats = {
                    "matched_pixels": unchanged_pixels,
                    "total_pixels": image_pixels,
                    "proportion": proportion
                }
                return stats, comparison_image_path, highlight_image_path, new_image_output_path

            stats, comparison_image_path, highlight_image_path, new_image_output_path = main(original_image, original_image_qt_filepath, new_image_filepath, new_image_qt_filepath, output_folder, threshold, compare_depth)

            file_difference = os.path.join(os.path.basename(output_folder), os.path.basename("difference_mask.png"))
            file_image_path = os.path.join(os.path.basename(output_folder), os.path.basename(comparison_image_path))
            file_highlight = os.path.join(os.path.basename(output_folder), os.path.basename(highlight_image_path))
            file_new_image = os.path.join(os.path.basename(output_folder), os.path.basename(new_image_output_path))

            return render_template('result.html',
                                   difference_mask=url_for('compare_bp.output_file', filename=file_difference),
                                   comparison_image=url_for('compare_bp.output_file', filename=file_image_path),
                                   highlight_image=url_for('compare_bp.output_file', filename=file_highlight),
                                   new_image=url_for('compare_bp.output_file', filename=file_new_image),
                                   stats=stats)

        except Exception as e:
            return str(e), 500

@compare_bp.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename)
