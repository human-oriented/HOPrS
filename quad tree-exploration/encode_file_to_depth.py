import sys
import cv2
import pdqhash
import hashlib
import os
import numpy as np
import argparse
import json
from datetime import datetime



def bits_to_hex(bits):
    binary_string = ''.join(str(bit) for bit in bits)
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string
class QuadTreeNode:
    def __init__(self, image, box, depth, path=''):
        self.box = box
        self.children = []
        self.depth = depth

        segment = image[box[1]:box[3], box[0]:box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        vector, quality = pdqhash.compute(segment_rgb)
        self.phash = bits_to_hex(vector)
        self.quality = quality
        
        
    def is_leaf_node(self):
        return len(self.children) == 0

class QuadTree:
    def __init__(self, image, file_hoprs, file_qt, max_depth, orig_x=0, orig_y=0, x0=0, y0=0, x1=0, y1=0):
        self.root = None
        self.max_depth = max_depth
        self.image = image
        self.build_tree(orig_x, orig_y, x0, y0, x1, y1)
        

    def build_tree(self, orig_x, orig_y, x0, y0, x1, y1):
        image = self.image

#Interpolate to the same size as the original to ensure that we don't misalign comparison boundaries by compounding errors
#May not be making much difference.  To be tested
        if not (x0 == 0 and y0 == 0 and x1 ==0 and y1 ==0):
            #This is a crop.  We need to resize 
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
        node = QuadTreeNode(image, box, depth, path)

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
            child_node = self.split_image(image, segment, depth+1, new_path)
            node.children.append(child_node)

        return node

    def print_tree(self, file, node=None, level=0, path=''):
        if node is None:
            node = self.root

        x0, y0, x1, y1 = node.box
        width, height = x1 - x0, y1 - y0
        file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{node.phash},quality {node.quality}\n")

        for index, child in enumerate(node.children):
            new_path = f"{path}{index+1}-" if path else f"{index+1}-"
            self.print_tree(file, child, level + 1, new_path)

if __name__ == "__main__":
 
#        print("Usage: python encode_file_to_depth.py -i <image> -d <depth> [-crop x0 y1 x1 y1] [-resize prev_width prev_height] --debug")
#        print("origsize is the size of the original image in x and y pixels, using this option will request that the image is resized to that size using bucubic expansion before the tree is built")
#        print("x0 y0 x1 y1 are the coordinates of this image in the coordinate system of the original. Specify these if you have cropped an image")
#        print("Think of these as the coordinates of the rectangular crop box")
#        

    parser = argparse.ArgumentParser(description='Encode an image to a hoprs and qt file.')
    parser.add_argument('-i', '--input',  type=str, required=True, help='Input image filename (and related <filename>.hoprs) file')
    parser.add_argument('-d', '--depth',  type=int, required=True, help='Depth of quad tree')
    
    parser.add_argument('-r', '--resize',  type=str, nargs=2, required=False, help='Dimensions that this image was resized FROM (not current size)')
    parser.add_argument('-c', '--crop',  type=str, nargs=4, required=False, help='top left and bottom right coordinates in px x0 y0 x1 y1')
    parser.add_argument('-n', '--note', type=str, required=False, help="A short description of what this edit was in quotes")
    args = parser.parse_args()

    image_path = args.input
    image_hoprs = args.input + ".hoprs" 
    image_qt = args.input + ".qt"
    max_depth = args.depth
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error opening image {image_path}. Please check the path and try again.")
            sys.exit(1)
    except IOError:
        print(f"Error opening image {image_path}. Please check the path and try again.")
        sys.exit(1)
    print(f"Opened (read) {image_path}")
    
    try:
        file_qt = open(image_qt, "w")
        print(f"Opened (write) {image_qt}")
        
    except: 
        print(f"ERROR: Couldn't open {image_qt} for write" )
        sys.exit(-1)

    if args.crop:
        x0 =     args.crop[0]
        y0 =     args.crop[1]
        x1 =     args.crop[2]
        y1 =     args.crop[3]
    else:
        x0=0
        y0=0
        x1=0
        y1=0
    
    if args.resize:
        orig_x = args.resize[0]
        orig_y = args.resize[1]
    else:
        orig_x = 0
        orig_y = 0

    try:
        file_hoprs = open(image_hoprs, "w")
        print(f"Opened (write) {image_hoprs}")        
        # Get the current date and time in UTC
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')

        if args.note:
            comment = args.note
        else:
            comment = "Need a meaningful comment in here at some point"
       # Data to be written to JSON, with the current time
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
            "Image_file": image_path,
            "QT_file": image_qt,
            "Encoded_depth": str(max_depth),
            "Perceptual_algorithm": "PDQ",
            "Comment": comment
        }
        json.dump(data, file_hoprs, indent=4)
        print(f"Written .hoprs file data")
        file_hoprs.close()
    except: 
        print(f"ERROR: Writing {image_hoprs}" )
        sys.exit(-1)

    quad_tree = QuadTree(image, file_hoprs, file_qt, max_depth, orig_x, orig_y, x0, y0, x1, y1)

    print("Building quad tree")
    quad_tree.print_tree(file_qt)
    print("Finished constructing quadtree")
    file_qt.close()

    print("Finishing up closing qt")


