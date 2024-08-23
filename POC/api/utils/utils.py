from flask import current_app, send_file
import os
import cv2
import csv
import io
from io import StringIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import pillow_heif
from enum import Enum
import imagehash
import pdqhash
import magic
from astrapy.client import DataAPIClient
from dotenv import load_dotenv
import tempfile


load_dotenv()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic'}
MAGIC_NUMBERS = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'heic': 'image/heic'
}

# Initialize the client and get a "Database" object
client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
database = client.get_database_by_api_endpoint(os.environ["ASTRA_DB_API_ENDPOINT"])
print(f"Database connected: {database.info().name}\n")

try:
    print("Trying to get the collection")
    collection = database.get_collection("quadtree_records")
except astrapy.DataApiException as e:
    print(f"Exception {e}")

print(f"Opened - Collection: {collection.full_name}\n")


# Helper function to convert a list of bits to a hexadecimal string
def bits_to_hex(bits):
    binary_string = ''.join(str(bit) for bit in bits)
    hex_string = format(int(binary_string, 2), '064x')  # Pad to ensure it is 64 hex digits
    return hex_string

def hex_to_binary_vector(hex_str):
    # Convert the hex string to binary string
    binary_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)
    # Convert binary string to a list of integers (0 and 1)
    vector = [int(bit) for bit in binary_str]
    if len(vector) != 256:
        print(f"Error: Vector length is {len(vector)}, expected 256. Hex string: {hex_str}")
    return vector

def validate_vectors(json_representation):
    for record in json_representation:
        vector = record['$vector']
        if len(vector) != 256:
            print(f"Record ID {record['_id']} has incorrect vector length: {len(vector)}")

class Matched(Enum):
    YES = 1
    NO = 2
    UNKNOWN = 3

class QuadTreeNode:
    def __init__(self, image=None, box=None, depth=0, hash_algorithm='pdq', path='', phash=None, quality=None, line=None, is_root=False):
        self.box = box if box else (0, 0, 0, 0)
        self.children = {}
        self.depth = depth
        self.hash_algorithm = hash_algorithm
        self.phash = phash
        self.quality = quality
        self.path = path
        self.removed = False
        self.ham_distance = -1
        self.matched = Matched.UNKNOWN
        self.purge = False
        self.optimise = False
    
        if image is not None:
            self.compute_hash(image)
        elif line:
            self.init_from_line(line, is_root)

    def compute_hash(self, image):
        segment = image[self.box[1]:self.box[3], self.box[0]:self.box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        if self.hash_algorithm == 'pdq':
            vector, quality = pdqhash.compute(segment_rgb)
            self.phash = bits_to_hex(vector)
            self.quality = quality
        else:
            hash_func = getattr(imagehash, self.hash_algorithm)
            self.phash = str(hash_func(Image.fromarray(segment_rgb)))

    def init_from_line(self, line, is_root=False):
        parts = line.split(',') if line else []
        if len(parts) < 10:
            print(f"Warning: Incomplete line: {line}")
        self.path = "" if is_root else parts[0]
        self.hash_algorithm = parts[8] if len(parts) > 8 else self.hash_algorithm
        self.phash = parts[9] if len(parts) > 9 else None
        self.quality = parts[10] if len(parts) > 10 else None
        self.box = tuple(map(int, parts[2:6])) if len(parts) > 5 else self.box

    def load_from_data(self, data):
        self.phash = data['perceptual_hash_hex']
        #self.quality = data.get('quality', None)

    def is_leaf_node(self):
        return len(self.children) == 0

#TODO understand if level is used elsewhere, prefer counting in path each time. 
    def store_in_astra_db(self, path, level, unique_qt_reference, jsonrepresentation):
        vector = hex_to_binary_vector(self.phash)
        if len(vector) != 256:
            print(f"Error: Vector length is {len(vector)}, expected 256. Path: {path}, Phash: {self.phash}")
        jsonrepresentation.append({
            "_id": unique_qt_reference + ' ' + path,
            'qt_ref': unique_qt_reference,
            'path': path,
            'level': path.count('-'),
            'x0': self.box[0],
            'y0': self.box[1],
            'x1': self.box[2],
            'y1': self.box[3],
            'width': self.box[2] - self.box[0],
            'height': self.box[3] - self.box[1],
            'hash_algorithm': self.hash_algorithm,
            'perceptual_hash_hex': self.phash,
            '$vector': vector
            #ty': self.quality
        })

    def add_child(self, path_segment, child_node):
        print(f"add_child called {path_segment} {child_node.path}")
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
        if not self.purge and (self.matched == Matched.YES or self.matched == Matched.NO):
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

    def print_tree(self, file, unpurged_only: bool = False):
        x0 = self.box[0]
        y0 = self.box[1]
        x1 = self.box[2]
        y1 = self.box[3]
        
        width, height = x1 - x0, y1 - y0
        path = self.path 
        level = self.depth

        if unpurged_only and self.purge:
            return

        file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{self.phash},{self.quality}\n")
        for child in self.children.values():
            child.print_tree(file, unpurged_only)

    def print_optimised_tree(self, file):
        x0 = self.box[0]
        y0 = self.box[1]
        x1 = self.box[2]
        y1 = self.box[3]
        
        width, height = x1 - x0, y1 - y0
        path = self.path 
        level = self.depth

        if not self.optimise:
            file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{self.phash},{self.quality}\n")

            for child in self.children.values():
                child.print_optimised_tree(file)

#Class to wrap a tree of quadtreenodes when creating a structuree
#from an image
class QuadTree:
    def __init__(self, image=None, max_depth=0, orig_x=0, orig_y=0, x0=0, y0=0, x1=0, y1=0, hash_algorithm='pdq', unique_qt_reference=""):
        current_app.logger.debug(f"quadtree constructor called {orig_x} {orig_y} {x0} {y0} {x1} {y1}")
        self.root = None
        self.max_depth = max_depth
        self.image = image
        self.hash_algorithm = hash_algorithm
        if image is not None:
            current_app.logger.debug("Quadtree constructor - have image")
            self.build_tree(orig_x, orig_y, x0, y0, x1, y1)
        self.unique_qt_reference = unique_qt_reference
        self.jsonrepresentation = []
        self.node_map = {}

    def build_tree(self, orig_x, orig_y, x0, y0, x1, y1):
        image = self.image
        current_app.logger.debug(f"build_tree called {orig_x} {orig_y} {x0} {y0} {x1} {y1}")

        if not (x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0):
            full_image = np.zeros((orig_y, orig_x, 3), dtype=np.uint8)
            new_width = x1 - x0
            new_height = y1 - y0
            resized_image = cv2.resize(image, (new_width, new_height))
            full_image[y0:y1, x0:x0+new_width] = resized_image
            image = full_image
        elif orig_x != 0 or orig_y != 0:
            image = cv2.resize(image, (orig_x, orig_y), interpolation=cv2.INTER_CUBIC)

        height, width = image.shape[:2]
        self.root = self.split_image(image, (0, 0, width, height), 1, '')

    def split_image(self, image, box, depth, path=''):
        current_app.logger.debug(f"split_image called {path}")
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
            #print(f"  split_image path:{path} index+1:{index+1} new_path:{new_path} depth:{depth}  ")
            child_node = self.split_image(image, segment, depth + 1, new_path)
            childkey = f"{index+1}-"
            node.children[childkey] = child_node

        return node

    def append_json_representation(self, node=None, level=0, path=''):
        current_app.logger.debug(f"append_json_representation called {path}")
        if node is None:
            node = self.root

        node.store_in_astra_db(path, level, self.unique_qt_reference, self.jsonrepresentation)
        
        for new_path, child in node.children.items():
            self.append_json_representation(child, child.path.count('-'), child.path)

    def write_to_astra_db(self, node:QuadTreeNode=None, level=0, path=''):
        current_app.logger.debug(f"write_to_astra_db called {path}")

        self.append_json_representation(node, path.count('-'), path)
        current_app.logger.debug(f"len of jsonrepresentation is {len(self.jsonrepresentation)}")
        
        validate_vectors(self.jsonrepresentation)
        
        current_app.logger.debug(f"First item {self.jsonrepresentation[0]}")
        current_app.logger.debug(f"Second item {self.jsonrepresentation[1]}")
        
        try:
            insertion_result = collection.insert_many(self.jsonrepresentation)
            current_app.logger.debug(f"* Inserted {len(insertion_result.inserted_ids)} items.")
        except Exception as e:
            current_app.logger.debug(f"* Exception inserting: {e}")


    def print_tree(self, file, node:QuadTreeNode=None, level:int=0, path=''):
        current_app.logger.debug(f"print_tree called on QuadTree object {path}")
        if node is None:
            node = self.root
        node.print_tree(file, False)    #Called on the root QuadTreeNode Object


    def add_node(self, path, node):

        current_app.logger.debug(f"add_node called {path}")
        self.node_map[path] = node
        if path == '':
            self.root = node
        else:
            parent_path = '-'.join(path.split('-')[:-1])
            parent_node = self.node_map.get(parent_path, None)
            if parent_node:
                parent_node.children[path] = node
# Read the image from the disk
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image from {image_path}")
    return image

# Initialize QuadTree with the image
def create_quadtree_from_image(image_path, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm):
    # Read the image
    image = read_image(image_path)    
    # Generate a unique reference
    current_time = int(time.time())
    unique_qt_reference = f"{os.path.basename(image_path)} {current_time}"
    
    # Create the QuadTree
    quad_tree = QuadTree(image, depth, orig_x, orig_y, x0, y0, x1, y1, algorithm, unique_qt_reference)
    
    return quad_tree

#Retrieve from database
def retrieve_quadtree(unique_qt_reference):
    query = {
        "qt_ref": {"$eq": unique_qt_reference}
    }
    csv_str = ""
    results = collection.find(query)
    current_app.logger.debug(f"About to retrieve QT from astra with uniqe ref : {unique_qt_reference}")

    for record in results:
        current_app.logger.debug(f"found record {record}")
        csv_str += f'{record["path"]},{record["level"]},{record["x0"]},{record["y0"]},{record["x1"]},{record["y1"]},{record["width"]},{record["height"]},pdq,{record["perceptual_hash_hex"]},\n'

    current_app.logger.debug(f"finished importing records from db into CSV")
    qt_root = parse_string_to_tree(sort_csv_by_first_field(csv_str))
    return qt_root

def parse_string_to_tree(csv_str):
    current_app.logger.debug(f"parse_string_to_tree {csv_str}")

    string_io = io.StringIO(csv_str)
    first_line = next(string_io, None)
    root = QuadTreeNode(line=first_line.strip(), is_root=True) if first_line else None
    for line in string_io:
        parts = line.strip().split(',')
        path_segments = parts[0].strip().strip('-').split('-') #Lose whitespace at end and trailing dash
        current_app.logger.debug(f"parse_string_to_tree path_segments should not have trailing - : {path_segments}")
        current = root
        for segment in path_segments:
            segment+= '-'
            if segment:
                if segment not in current.children:
                    current.add_child(segment, QuadTreeNode(line=line.strip()))
                current = current.children[segment]
    return root

def parse_file_to_tree(filepath):
    with open(filepath, 'r') as f:
        csv_str = f.read()
    return parse_string_to_tree(sort_csv_by_first_field(csv_str))

# Computes the Hamming distance between two perceptual hashes
def hamming_distance(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

# Marks a node and its children as removed
def mark_as_removed(node : QuadTreeNode, IsRemoved=True):
    current_app.logger.debug(f"mark_as_removed called setting node at path:{node.path} Isremoved:{IsRemoved}")
    node.removed = IsRemoved
    for child in node.children.values():
        mark_as_removed(child, IsRemoved)

def mark_as_matched(node : QuadTreeNode, MatchedStatus=Matched.UNKNOWN):
    current_app.logger.debug(f"mark_as_matched called setting node at path:{node.path} Isremoved:{MatchedStatus}")
    node.matched = MatchedStatus
    for child in node.children.values():
        mark_as_matched(child, MatchedStatus)


# Draws a comparison between two nodes and outputs the result image
def draw_comparison(image_list, list_pixel_counter, node1 : QuadTreeNode, node2 : QuadTreeNode, output_path, counter, threshold_cli, compare_depth_cli):
    current_app.logger.debug(f"draw_comparison called {node1.path} {node2.path} {output_path} {counter[0]} {threshold_cli} {compare_depth_cli}")
    
    x0 = node1.box[0]
    y0 = node1.box[1]
    x1 = node1.box[2]
    y1 = node1.box[3]

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
    text = f"threshold:{threshold_cli}  depth:{compare_depth_cli}  perceptual_alg:{node1.hash_algorithm}"
    cv2.putText(image_list[0], text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    if counter[0] != -1:
        return
    current_app.logger.debug(f"Writing {output_path}/comparison_t{threshold_cli}_d{compare_depth_cli}_{node1.hash_algorithm}_{counter[0]:04}.jpg")
    cv2.imwrite(f"{output_path}/comparison_t{threshold_cli}_d{compare_depth_cli}_{node1.hash_algorithm}_{counter[0]:04}.jpg", image_list[0], [int(cv2.IMWRITE_JPEG_QUALITY), 50])

# Compares two nodes and outputs the comparison images
def compare_and_output_images(
        image_list, 
        list_pixel_counter, 
        node1: QuadTreeNode, 
        node2: QuadTreeNode, 
        image_path, 
        output_path, 
        threshold, 
        counter=[0], 
        compare_depth=99):
    current_app.logger.debug(f"compare_and_output_images called {node1.path} {node2.path} {image_path} {output_path} {threshold} {counter[0]} {compare_depth} node1.removed {node1.removed } node2.removed {node2.removed }")
    
    current_app.logger.debug(f"node1.phash:{node1.phash} node2.phash:{node2.phash} node1.removed:{node1.removed} node2.removed:{node2.removed}")
    if node1.phash and node2.phash and not node1.removed and not node2.removed:
        current_app.logger.debug(f" -2 compare_and_output_images ")

        if node1.hash_algorithm != node2.hash_algorithm:
            current_app.logger.debug(f"Error: Perceptual hashing algorithms used for comparison are different for node path [{node1.path}]: {node1.hash_algorithm} vs {node2.hash_algorithm}")
            return
        distance = hamming_distance(node1.phash, node2.phash)
        node1.ham_distance = distance
        node2.ham_distance = distance
        current_app.logger.debug(f" -2a compare_and_output_images node1.path:{node1.path} with phash: {node1.phash} vs node2.path: {node2.path} with phash:{node2.phash}  distance {distance} against threshold {threshold}")

        #This section matches (below threshold). 
        #Let's remove the nodes beneath this so that we can generate an optimised version later
        #And speed comparison
        if distance <= threshold:
            current_app.logger.debug(f"Matched (green box) removing nodes below threshold {distance} {threshold}")
            mark_as_removed(node1)
            mark_as_removed(node2)
            node1.matched = Matched.YES
        else:
            current_app.logger.debug(f"Doesn't match.  (will be drawing a red box here) NOT removing nodes as above threshold {distance} {threshold}")
            node1.matched = Matched.NO

        draw_comparison(image_list, list_pixel_counter, node1, node2, output_path, counter, threshold, compare_depth)
        counter[0] += 1
    current_app.logger.debug(f" -2b compare_and_output_images counter {counter[0]}")
    
    for key in node1.children:
        current_app.logger.debug(f" -3 compare_and_output_images key:{key} len node1: {len(node1.children)}  len node2:{len(node2.children)}____ ")
        
        for k1 in node1.children:
            current_app.logger.debug(f"  -3. key in node1 is {k1}___")    
        for k2 in node2.children:
            current_app.logger.debug(f"  -3. key in node2 is {k2}___")
        
        if key in node2.children:
            current_app.logger.debug(f"recursing down from {node1.path} to {node1.children[key].path} and {node2.children[key].path}")
            compare_and_output_images(image_list, list_pixel_counter, node1.children[key], node2.children[key], image_path, output_path, threshold, counter, compare_depth)
        else:
            current_app.logger.debug(f"key {key} not found in node2 children, possibly hit bottom of tree or tree is incorrect")

# Counts the number of black pixels in an image
def count_black_pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)
    num_black_pixels = np.count_nonzero(inverted_image)
    return num_black_pixels

# Creates a red overlay on an image based on a mask image
def create_red_overlay(original_image_path, mask_image_path, output_image_path, translucence=50):
    current_app.logger.debug(f"create_red_overlay called {original_image_path} {mask_image_path} {output_image_path} {translucence}")
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
    current_app.logger.debug(f"Overlay image created successfully with {translucence}% translucence: {output_image_path}")

# Returns a list of available perceptual hashing algorithms
def list_available_algorithms():
    available_algorithms = [name for name in dir(imagehash) if name[0].islower() and callable(getattr(imagehash, name))]
    return available_algorithms


def sort_csv_by_first_field(csv_data):
    # Use StringIO to read the string as a CSV file
    input_file = StringIO(csv_data)
    reader = csv.reader(input_file)
    
    
    # Sort the data by the first field (alphabetically)
    sorted_data = sorted(reader, key=lambda row: row[0])
    
    # Prepare to write the sorted data back to a string
    output_file = StringIO()
    writer = csv.writer(output_file)
    writer.writerows(sorted_data)  # Write the sorted data
    
    # Return the sorted CSV data as a string
    return output_file.getvalue()

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    try:
        # Check if the file's extension is allowed
        if not allowed_image(file.filename):
            return False, "Unsupported file extension"

        # Use python-magic to check the file's magic number
        file_mime_type = magic.from_buffer(file.read(2048), mime=True)
        file.seek(0)  # Reset file pointer after reading
        
        # Get the expected MIME type based on the file extension
        expected_mime_type = MAGIC_NUMBERS.get(file.filename.rsplit('.', 1)[1].lower())
        if file_mime_type != expected_mime_type:
            return False, f"Mismatched MIME type: expected {expected_mime_type}, got {file_mime_type}"
        
        # Open the image file using Pillow to check format
        # try:
        #     img = Image.open(file)
        #     img.verify()  # Verify that the image file is not corrupted
        #     file.seek(0)  # Reset file pointer for further operations
        # except UnidentifiedImageError:
        #     return False, "The file is not a valid image or is corrupted"
        
        # Image is valid
        return True, "Image is valid"
    except Exception as e:
        return False, str(e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'qt', 'csv'}
    
def validate_quadtree(file):
    try:
        # Check if the file's extension is allowed
        if not allowed_file(file.filename):
            return False, "Unsupported file extension"

        # Use python-magic to check the file's magic number
        file_mime_type = magic.from_buffer(file.read(2048), mime=True)
        file.seek(0)  # Reset file pointer after reading
        
        # Get the expected MIME type based on the file extension
        allowed_mime_types = ['text/csv', 'application/csv']
        if file_mime_type not in allowed_mime_types:
            return False, f"Mismatched MIME type: expected {allowed_mime_types}, got {file_mime_type}"
        
        # Image is valid
        return True, "File is valid"
    except Exception as e:
        return False, str(e)
    
def convert_heic(file):
    try:
            # Read HEIC file from the request
            current_app.logger.debug(f"Convert heic to png - {file.filename}")
            heif_file = pillow_heif.open_heif(file)
            
            # Convert to Pillow Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride
            )
            current_app.logger.debug(f"Convert heic to png - pillow image created")
            # Save image to an in-memory bytes buffer in PNG format
            image_buffer = io.BytesIO()
            image.save(image_buffer, format='PNG')
            image_buffer.seek(0)  # Rewind the buffer to the beginning
            current_app.logger.debug(f"Convert heic to png - saved to buffer")
            # Convert image buffer to a numpy array and then decode with OpenCV
            image_array = np.frombuffer(image_buffer.getvalue(), dtype=np.uint8)
            return image_array
            # image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # current_app.logger.debug(f"Convert heic to png - image decoded")

            # if image_cv is not None:
            #     current_app.logger.debug(f"image cv not null")
            #     return image_cv
            # else:
            #     raise ValueError("OpenCV could not read the image")
            
    except Exception as e:
        raise ValueError("Could not convert heic to png")
    
