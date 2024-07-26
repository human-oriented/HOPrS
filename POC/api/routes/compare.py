import os
import shutil
import numpy as np
from flask import url_for, jsonify
import json
from datetime import datetime, timezone
import cv2
from utils.utils import (
    QuadTree, draw_comparison,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    parse_file_to_tree
)

def compare_image(original_image, original_image_qt_filepath, new_image_filepath, new_image_filename, new_image_qt_filepath, threshold, compare_depth, output_folder):
    image = cv2.imread(new_image_filepath)
    if image is None:
        return jsonify({"error": "Error opening new image. Please check the file."}), 400

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

        quad_tree = QuadTree(
            image,
            compare_depth,
            orig_x=width,
            orig_y=height,
            x0=0, y0=0,
            x1=width, y1=height,
            hash_algorithm='pdq',
            unique_qt_reference=str(new_image_filename)+" "+str(current_time))

        quad_tree.print_tree(open(new_image_qt_filepath, "w"))

        response_data = create_images(original_image, original_image_qt_filepath, new_image_filepath, new_image_qt_filepath, output_folder, threshold, compare_depth)

        return jsonify(response_data)
    
def create_images(original_image, original_image_qt, new_image, new_image_qt, output_folder, threshold, compare_depth):
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
        return None, "Comparison image not found."

    height, width = image.shape[:2]
    image_pixels = width * height
    proportion = unchanged_pixels / image_pixels
    stats = {
        "matched_pixels": unchanged_pixels,
        "total_pixels": image_pixels,
        "proportion": proportion
    }
    
    return {
            "difference_mask": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(difference_mask_path), _external=True),
            "comparison_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(comparison_image_path), _external=True),
            "highlight_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(highlight_image_path), _external=True),
            "new_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(new_image_output_path), _external=True),
            "stats": stats
     }
