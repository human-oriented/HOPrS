import os
import shutil
import numpy as np
from flask import Flask, request, current_app, send_file, url_for, jsonify, send_from_directory
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import json
from datetime import datetime, timezone
import cv2  # Assuming you have OpenCV installed
from utils import (
    QuadTree, hamming_distance, mark_as_removed, draw_comparison,
    compare_and_output_images, count_black_pixels, create_red_overlay,
    parse_file_to_tree, Matched
)

app = Flask(__name__)
upload_folder = './upload'  # Change this to your desired upload folder path
app.config['UPLOAD_FOLDER'] = upload_folder

output_folder = './output'  # Change this to your desired upload folder path
app.config['OUTPUT_FOLDER'] = output_folder

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

api = Api(app, version='1.0', title='HOPrS', description='Open standard for content authentication')
ns = api.namespace('hoprs', description='File upload operations')

@app.route('/output/<path:folder>/<path:filename>')
def get_file(folder, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), filename)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True, help='Original file to encode')
upload_parser.add_argument('depth', type=int, required=False, help='Depth for encoding', default=5)
upload_parser.add_argument('algorithm', type=str, required=False, help='Perceptual algorithm to use', default='pdq')
upload_parser.add_argument('resize', type=str, required=False, help='Resize dimensions (comma-separated)', default=None)
upload_parser.add_argument('crop', type=str, required=False, help='Crop coordinates (comma-separated)', default=None)
upload_parser.add_argument('note', type=str, required=False, help='Comment or note', default="Need a meaningful comment in here at some point")

@ns.route('/encode')
class FileUpload(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        file = args['file']
        depth = args['depth']
        algorithm = args['algorithm']
        resize = args['resize']
        crop = args['crop']
        note = args['note']

        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

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
            quad_tree = QuadTree(image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, str(file.filename) + " " + str(current_time))
            quad_tree.print_tree(open(filename_dot_qt, "w"))
            quad_tree.write_to_astra_db(None, 0, '')

            return send_file(filename_dot_qt, as_attachment=True)

        except Exception as e:
            return str(e), 500
        
compare_parser = api.parser()
compare_parser.add_argument('original_image_qt', location='files', type=FileStorage, required=True, help='Original QT image to be compared')
compare_parser.add_argument('new_image', location='files', type=FileStorage, required=True, help='New image to compare')
compare_parser.add_argument('threshold', type=int, required=False, help='Threshold for comparison', default=10)
compare_parser.add_argument('compare_depth', type=int, required=False, help='Depth for comparison', default=5)

@ns.route('/compare')
@ns.response(404, 'Task not found')
class Compare(Resource):
    @api.expect(compare_parser)
    def post(self):
        args = compare_parser.parse_args()
        original_image_qt_file = args['original_image_qt']
        new_image_file = args['new_image']

        if original_image_qt_file.filename == '' or new_image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

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
                    return None, "Comparison image not found."

                height, width = image.shape[:2]
                image_pixels = width * height
                proportion = unchanged_pixels / image_pixels
                stats = {
                    "matched_pixels": unchanged_pixels,
                    "total_pixels": image_pixels,
                    "proportion": proportion
                }
                return stats, comparison_image_path, highlight_image_path, new_image_output_path, difference_mask_path

            stats, comparison_image_path, highlight_image_path, new_image_output_path, difference_mask_path = main(original_image, original_image_qt_filepath, new_image_filepath, new_image_qt_filepath, output_folder, threshold, compare_depth)

            response_data = {
                "difference_mask": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(difference_mask_path), _external=True),
                "comparison_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(comparison_image_path), _external=True),
                "highlight_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(highlight_image_path), _external=True),
                "new_image": url_for('get_file', folder=os.path.basename(output_folder), filename=os.path.basename(new_image_output_path), _external=True),
                "stats": stats
            }

            return jsonify(response_data)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
