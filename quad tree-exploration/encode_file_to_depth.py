import sys
import cv2
import pdqhash
import hashlib
import os
import numpy as np


#TODO - better pattern needed for command line parsing
debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')



def bits_to_hex(bits):
    binary_string = ''.join(str(bit) for bit in bits)
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string

class QuadTreeNode:
    def __init__(self, image, box, depth, path='', image_path=''):
        self.box = box
        self.children = []
        self.depth = depth

        segment = image[box[1]:box[3], box[0]:box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        vector, quality = pdqhash.compute(segment_rgb)
        self.phash = bits_to_hex(vector)
        self.quality = quality
        
        
        if not debug_mode:
            #don't output the segments
            return

        coords = f"{box[0]},{box[1]},{box[2]},{box[3]}".replace(',', '.')
        filename = f"tmp.D{depth}.{path}.pdq.{self.phash}.quality{quality}.{os.path.basename(image_path)}.segment.{coords}.png".replace(',', '.')
        success = cv2.imwrite(filename, segment)
        
        if not success:
            print(f"Failed to write {filename}. Check the path, permissions, and disk space.")
        
    def is_leaf_node(self):
        return len(self.children) == 0

class QuadTree:
    def __init__(self, image_path, max_depth, orig_x=0, orig_y=0, x0=0, y0=0, x1=0, y1=0):
        self.root = None
        self.max_depth = max_depth
        self.image_path = image_path
        self.build_tree(orig_x, orig_y,x0, y0, x1, y1)
        

    def build_tree(self, orig_x, orig_y, x0, y0, x1, y1):
        try:
            #print(f"DEBUG: reading {self.image_path}")
            image = cv2.imread(self.image_path)
            if image is None:
                print("Error opening the image file. Please check the path and try again.")
                sys.exit(1)
        except IOError:
            print("Error opening the image file. Please check the path and try again.")
            sys.exit(1)

    

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
        node = QuadTreeNode(image, box, depth, path, self.image_path)

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
    if not (len(sys.argv) == 3 or len(sys.argv) == 5 or len(sys.argv)==9):
        print("Usage: python encode_file_to_depth.py image_path max_depth [origsize_x] [origsize_y] [x0] [y0] [x1] [y1] --debug")
        print("--debug will emit the images that make up the segments that are being compared")
        print("origsize is the size of the original image in x and y pixels, using this option will request that the image is resized to that size using bucubic expansion before the tree is built")
        print("x0 y0 x1 y1 are the coordinates of this image in the coordinate system of the original. Specify these if you have cropped an image")
        print("Think of these as the coordinates of the rectangular crop box")
        
        sys.exit(1)

    image_path = sys.argv[1]
    max_depth = int(sys.argv[2])
    
    
    try:
        path = image_path+".hoprs"
        file = open(path, "w")
        print(f"Creating {path}")
    except: 
        print("ERROR: Couldn't open " +image_path + ".hoprs for write" )

    
    if len(sys.argv) == 5:
        #specifying a resize
        orig_x = int(sys.argv[3])
        orig_y = int(sys.argv[4])
        quad_tree = QuadTree(image_path, max_depth, orig_x, orig_y)
        file.write(f"Origin reference,{0},{0},{0},{0},{orig_x},{orig_y}\n")

    elif len(sys.argv) == 3:
        #clean config
        quad_tree = QuadTree(image_path, max_depth)
        file.write(f"Origin reference,{0},{0},{0},{0},{0},{0}\n")

    elif len(sys.argv) == 9 :
        #Specifying a crop
        orig_x = int(sys.argv[3])
        orig_y = int(sys.argv[4])
        x0 =     int(sys.argv[5])
        y0 =     int(sys.argv[6])
        x1 =     int(sys.argv[7])
        y1 =     int(sys.argv[8])
        print(f"Specifying a crop {x0},{y0},{x1},{y1},{orig_x},{orig_y}")
        quad_tree = QuadTree(image_path, max_depth, orig_x, orig_y, x0, y0, x1, y1)        
        print("DEBUG 2")
        file.write(f"Origin reference,{x0},{y0},{x1},{y1},{orig_x},{orig_y}\n")
    print("Building quad tree")
    quad_tree.print_tree(file)
    print("Finished constructing quadtree")
    file.close()

    print("Finishing up")


