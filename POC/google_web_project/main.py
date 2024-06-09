from flask import Flask, request, render_template, send_file, send_from_directory, jsonify, redirect, url_for, flash
import os
import cv2
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import pdqhash
import imagehash
import numpy as np
from enum import Enum
import shutil
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/output'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

debug_mode = False

class QuadTreeNode:
    def __init__(self, image, box, depth, hash_algorithm='pdq', path=''):
        self.box = box
        self.children = []
        self.depth = depth
        self.hash_algorithm = hash_algorithm

        segment = image[box[1]:box[3], box[0]:box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        if hash_algorithm == 'pdq':
            vector, quality = pdqhash.compute(segment_rgb)
            self.phash = bits_to_hex(vector)
            self.quality = quality
        else:
            hash_func = getattr(imagehash, hash_algorithm)  # Get the hash function dynamically
            self.phash = str(hash_func(Image.fromarray(segment_rgb)))
            self.quality = None  # Initialize quality to None for other hash algorithms

    def is_leaf_node(self):
        return len(self.children) == 0

class QuadTree:
    def __init__(self, image, file_hoprs, file_qt, max_depth, orig_x=0, orig_y=0, x0=0, y0=0, x1=0, y1=0, hash_algorithm='pdq'):
        self.root = None
        self.max_depth = max_depth
        self.image = image
        self.hash_algorithm = hash_algorithm
        self.build_tree(orig_x, orig_y, x0, y0, x1, y1)

    def build_tree(self, orig_x, orig_y, x0, y0, x1, y1):
        image = self.image

        if not (x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0):
            full_image = np.zeros((orig_y, orig_x, 3), dtype=np.uint8)
            new_width = x1 - x0
            new_height = y1 - y0
            resized_image = cv2.resize(image, (new_width, new_height))
            full_image[y0:y1, x0:x1] = resized_image
            image = full_image
        elif orig_x != 0 or orig_y != 0:
            image = cv2.resize(image, (orig_x, orig_y), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{self.image_path}_tmp_resized.png", image)

        height, width = image.shape[:2]
        self.root = self.split_image(image, (0, 0, width, height), 1, '')

    def split_image(self, image, box, depth, path=''):
        x0, y0, x1, y1 = box
        node = QuadTreeNode(image, box, depth, self.hash_algorithm, path)

        if depth >= self.max_depth:
            return node

        width, height = x1 - x0, y1 - y0
        half_width, half_height = width // 2, height // 2
        segments = [
            (x0, y0, x0 + half_width, y0 + half_height),
            (x0 + half_width, y0, x1, y0 + half_height),
            (x0, y0 + half_height, x0 + half_width, y1),
            (x0 + half_width, y0 + half_height, x1, y1)
        ]

        for index, segment in enumerate(segments):
            new_path = f"{path}{index+1}-" if path else f"{index+1}-"
            child_node = self.split_image(image, segment, depth + 1, new_path)
            node.children.append(child_node)

        return node

    def print_tree(self, file, node=None, level=0, path=''):
        if node is None:
            node = self.root

        x0, y0, x1, y1 = node.box
        width, height = x1 - x0, y1 - y0
        file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},{node.hash_algorithm},{node.phash},{node.quality}\n")

        for index, child in enumerate(node.children):
            new_path = f"{path}{index+1}-" if path else f"{index+1}-"
            self.print_tree(file, child, level + 1, new_path)

def bits_to_hex(bits):
    binary_string = ''.join(str(bit) for bit in bits)
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string

def list_available_algorithms():
    available_algorithms = [name for name in dir(imagehash) if name[0].islower() and callable(getattr(imagehash, name))]
    return available_algorithms

class TreeNode:
    def __init__(self, line, is_root=False):
        parts = line.split(',') if line else []
        self.path = "" if is_root else parts[0]
        self.hash = parts[9] if parts else None
        self.algorithm = parts[8] if parts else parts[7]
        self.line = line
        self.children = {}
        self.removed = False
        self.ham_distance = -1
        self.matched = Matched.UNKNOWN
        self.purge = False #Will be set to true later if all subnodes don't match
        self.optimise = False
        
    def add_child(self, path_segment, child_node):
        self.children[path_segment] = child_node

    def should_purge(self):
        if self.matched != Matched.NO:
            return False
        for child in self.children.values():
            if not child.should_purge():
                return False
        self.purge = True
        return True
    
    def purge_tree(self):
        self.should_purge()
        for child in self.children.values():
            child.purge_tree()

    def should_optimise(self):
        if self.purge == False and (self.matched == Matched.YES or self.matched == Matched.NO):
            return False
        for child in self.children.values():
            if not child.should_optimise():
                return False
        self.optimise = True
        return True
    
    def optimise_tree(self):
        self.should_optimise()
        for child in self.children.values():
            child.optimise_tree()

    def print_tree(self, file, unpurged_only):
        parts = self.line.split(',')
        x0, y0, x1, y1 = map(int, parts[2:6])
        width, height = x1 - x0, y1 - y0
        path = parts[0]
        level = parts[1]
        
        if len(parts) > 10:
            quality = parts[10]
        else:
            quality = None
        
        if unpurged_only and self.purge:
            return

        file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{self.hash},{quality}\n")

        for child in self.children.values():
            child.print_tree(file, unpurged_only)

    def print_optimised_tree(self, file):
        parts = self.line.split(',')
        x0, y0, x1, y1 = map(int, parts[2:6])
        width, height = x1 - x0, y1 - y0
        path = parts[0]
        level = parts[1]
        
        if len(parts) > 10:
            quality = parts[10]
        else:
            quality = None
        
        if not self.optimise:
            file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{self.hash},{quality}\n")

            for child in self.children.values():
                child.print_optimised_tree(file)

class Matched(Enum):
    YES = 1
    NO = 2
    UNKNOWN = 3

def parse_file_to_tree(filepath):
    with open(filepath, 'r') as f:
        first_line = next(f, None)
        root = TreeNode(first_line.strip(), is_root=True) if first_line else None
        for line in f:
            parts = line.strip().split(',')
            path_segments = parts[0].split('-')
            current = root
            for segment in path_segments:
                if segment:
                    if segment not in current.children:
                        current.add_child(segment, TreeNode(line.strip()))
                    current = current.children[segment]
    return root

def hamming_distance(hash1, hash2):
    b1 = bin(int(hash1, 16))[2:].zfill(256)
    b2 = bin(int(hash2, 16))[2:].zfill(256)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))

def mark_as_removed(node):
    node.removed = True
    for child in node.children.values():
        mark_as_removed(child)

def draw_comparison(image_list, list_pixel_counter, node1, node2, output_path, counter, threshold_cli, compare_depth_cli):
    parts = node1.line.split(',')
    x0, y0, x1, y1 = map(int, parts[2:6])
    x = int(x0 + (x1 - x0) / 2)
    y = int(y0 + (y1 - y0) / 2)
    if (node1.removed):
        list_pixel_counter[0] += (x1-x0) * (y1-y0)
    color = (0, 255, 0) if node1.removed else (0, 0, 255)
    cv2.rectangle(image_list[0], (x0, y0), (x1, y1), color, 4)
    cv2.putText(image_list[0], str(node1.ham_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    if node1.removed:
        cv2.rectangle(image_list[1], (x0, y0), (x1, y1), (0,0,0), -1)
    cv2.rectangle(image_list[0], (0, 0), (4000, 120), (30, 30, 30), -1)
    text = f"threshold:{threshold_cli}  depth:{compare_depth_cli}  perceptual_alg:{node1.algorithm}"
    cv2.putText(image_list[0], text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    if not debug_mode and counter[0] != -1:
        return
    cv2.imwrite(f"{output_path}/comparison_t{threshold_cli}_d{compare_depth_cli}_{node1.algorithm}_{counter[0]:04}.jpg", image_list[0], [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def compare_and_output_images(image_list, list_pixel_counter, node1, node2, image_path, output_path, threshold, counter=[0], compare_depth=99):
    if len(node1.path.split('-')) - 1 >= compare_depth:
        return
    
    if node1.hash and node2.hash and not node1.removed and not node2.removed:

        if node1.algorithm != node2.algorithm:
            print(f"Error: Perceptual hashing algorithms used for comparison are different for node path [{node1.path}]: {node1.algorithm} vs {node2.algorithm}")
            return

        distance = hamming_distance(node1.hash, node2.hash)
        node1.ham_distance = distance
        node2.ham_distance = distance

        if distance <= threshold:
            mark_as_removed(node1)
            mark_as_removed(node2)
            node1.matched = Matched.YES
        else:
            node1.matched = Matched.NO
                
        draw_comparison(image_list, list_pixel_counter, node1, node2, output_path, counter, threshold, compare_depth)
        counter[0] += 1
    
    for key in node1.children:
        if key in node2.children:
            compare_and_output_images(image_list, list_pixel_counter, node1.children[key], node2.children[key], image_path, output_path, threshold, counter, compare_depth)

def count_black_pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)
    num_black_pixels = np.count_nonzero(inverted_image)
    return num_black_pixels

def create_red_overlay(original_image_path, mask_image_path, output_image_path, translucence=50):
    original_image = Image.open(original_image_path)
    mask_image = Image.open(mask_image_path).convert('L')
    translucence_value = int(255 * (translucence / 100))
    semi_transparent_mask = mask_image.point(lambda p: translucence_value if p > 0 else 0)
    red_overlay = Image.new('RGB', original_image.size, color=(255, 0, 0))
    red_overlay.putalpha(semi_transparent_mask)
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    final_image = Image.alpha_composite(original_image, red_overlay)
    final_image.save(output_image_path)
    logging.info(f"Overlay image created successfully with {translucence}% translucence: {output_image_path}")
    print(f"Overlay image created successfully with {translucence}% translucence: {output_image_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            depth = int(request.form.get('depth', 5))
            algorithm = request.form.get('algorithm', 'pdq')
            resize = request.form.get('resize', None)
            crop = request.form.get('crop', None)
            note = request.form.get('note', "Need a meaningful comment in here at some point")

            if resize:
                resize = tuple(map(int, resize.split(',')))

            if crop:
                crop = tuple(map(int, crop.split(',')))

            try:
                image = cv2.imread(filepath)
                if image is None:
                    return "Error opening image. Please check the file.", 400

                image_hoprs = filepath + ".hoprs"
                image_qt = filepath + ".qt"

                orig_x, orig_y = resize if resize else (0, 0)
                x0, y0, x1, y1 = crop if crop else (0, 0, 0, 0)

                with open(image_hoprs, "w") as file_hoprs:
                    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')
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
                        "QT_file": image_qt,
                        "Encoded_depth": str(depth),
                        "Perceptual_algorithm": algorithm.upper(),
                        "Comment": note
                    }
                    json.dump(data, file_hoprs, indent=4)

                quad_tree = QuadTree(image, file_hoprs, image_qt, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm)
                quad_tree.print_tree(open(image_qt, "w"))

                return send_file(image_qt, as_attachment=True)

            except Exception as e:
                logging.error(f"Error processing upload: {e}")
                print(f"Error processing upload: {e}")
                return str(e), 500
    except Exception as e:
        logging.error(f"Error in upload route: {e}")
        print(f"Error in upload route: {e}")
        return str(e), 500

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    try:
        if request.method == 'POST':
            if 'original_image_qt' not in request.files or 'new_image' not in request.files:
                return "No file part", 400

            original_image_qt_file = request.files['original_image_qt']
            new_image_file = request.files['new_image']

            if original_image_qt_file.filename == '' or new_image_file.filename == '':
                return "No selected file", 400

            if original_image_qt_file and new_image_file:
                original_image_qt_filename = secure_filename(original_image_qt_file.filename)
                new_image_filename = secure_filename(new_image_file.filename)
                original_image_qt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_image_qt_filename)
                new_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_image_filename)

                original_image_qt_file.save(original_image_qt_filepath)
                new_image_file.save(new_image_filepath)

                threshold = int(request.form.get('threshold', 10))
                compare_depth = int(request.form.get('compare_depth', 5))
                output_folder = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
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

                    quad_tree = QuadTree(image, file_hoprs, new_image_qt_filepath, compare_depth, hash_algorithm='pdq')
                    quad_tree.print_tree(open(new_image_qt_filepath, "w"))

                    def main(original_image, original_image_qt, new_image, new_image_qt, output_folder, threshold, compare_depth):
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        
                        tree1 = parse_file_to_tree(original_image_qt)
                        tree2 = parse_file_to_tree(new_image_qt)
                        
                        image = cv2.imread(original_image)
                        image_derivative = cv2.imread(new_image)
                        height_1, width_1 = image.shape[:2]
                        image_1 = np.zeros((height_1, width_1, 3), np.uint8)
                        image_1[:] = (255,255,255)
                        
                        pixel_counter = 0
                        list_pixel_counter = [pixel_counter]
                        list_images = [image_derivative, image_1]
                        
                        compare_and_output_images(list_images, list_pixel_counter, tree1, tree2, original_image, output_folder, threshold, [0], compare_depth)
                        
                        comparison_image_path = None
                        for file in os.listdir(output_folder):
                            if file.startswith("comparison_") and file.endswith(".jpg"):
                                comparison_image_path = os.path.join(output_folder, file)
                                break

                        logging.info(f"comparison_image_path: {comparison_image_path}")
                        print(f"comparison_image_path: {comparison_image_path}")

                        difference_mask_path = os.path.join(output_folder, "difference_mask.png")
                        cv2.imwrite(difference_mask_path, list_images[1], [int(cv2.IMWRITE_JPEG_QUALITY), 50])

                        highlight_image_path = os.path.join(output_folder, "highlighted_image.png")
                        create_red_overlay(new_image, difference_mask_path, highlight_image_path)

                        new_image_output_path = os.path.join(output_folder, os.path.basename(new_image))
                        shutil.copy(new_image, new_image_output_path)

                        logging.info(f"difference_mask_path: {difference_mask_path}")
                        logging.info(f"highlight_image_path: {highlight_image_path}")
                        logging.info(f"new_image_output_path: {new_image_output_path}")
                        print(f"difference_mask_path: {difference_mask_path}")
                        print(f"highlight_image_path: {highlight_image_path}")
                        print(f"new_image_output_path: {new_image_output_path}")

                        unchanged_pixels = count_black_pixels(list_images[1])

                        draw_comparison(list_images, list_pixel_counter, tree1, tree2, output_folder, [-1], threshold, compare_depth)

                        tree1.purge_tree()
                        tree1.optimise_tree()
                        
                        with open(os.path.join(output_folder, 'tmp.purged.qt'), 'w') as f:
                            tree1.print_tree(f, True)

                        with open(os.path.join(output_folder, 'tmp.optimised.qt'), 'w') as f:
                            tree1.print_optimised_tree(f)

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

                    if not comparison_image_path:
                        comparison_image_path = difference_mask_path

                    return render_template('result.html', 
                                           output_image=url_for(
                                               'output_file', 
                                               filename=os.path.join(os.path.basename(output_folder), 
                                                                     "difference_mask.png")), 
                                           comparison_image=url_for('output_file', 
                                                filename=os.path.join(os.path.basename(output_folder), 
                                                                      os.path.basename(comparison_image_path))), 
                                           highlight_image=url_for('output_file', 
                                                filename=os.path.join(os.path.basename(output_folder), 
                                                                      os.path.basename(highlight_image_path))), 
                                           new_image=url_for('output_file', 
                                                filename=os.path.join(os.path.basename(output_folder), 
                                                                      os.path.basename(new_image_output_path))), 
                                           stats=stats)

                except Exception as e:
                    logging.error(f"Error processing comparison: {e}")
                    print(f"Error processing comparison: {e}")
                    return str(e), 500
    except Exception as e:
        logging.error(f"Error in compare route: {e}")
        print(f"Error in compare route: {e}")
        return str(e), 500

    return render_template('compare.html')

@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
