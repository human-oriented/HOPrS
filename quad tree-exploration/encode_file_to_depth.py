import sys
import cv2
import pdqhash
import hashlib
import os

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
        
        coords = f"{box[0]},{box[1]},{box[2]},{box[3]}".replace(',', '.')
        filename = f"tmp.D{depth}.{path}.pdq.{self.phash}.quality{quality}.{os.path.basename(image_path)}.segment.{coords}.png".replace(',', '.')
        success = cv2.imwrite(filename, segment)
        if not success:
            print(f"Failed to write {filename}. Check the path, permissions, and disk space.")
        
    def is_leaf_node(self):
        return len(self.children) == 0

class QuadTree:
    def __init__(self, image_path, max_depth, orig_x, orig_y):
        self.root = None
        self.max_depth = max_depth
        self.image_path = image_path
        self.build_tree(orig_x, orig_y)

    def build_tree(self, orig_x, orig_y):
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                print("Error opening the image file. Please check the path and try again.")
                sys.exit(1)
        except IOError:
            print("Error opening the image file. Please check the path and try again.")
            sys.exit(1)

#Interpolate to the same size as the original to ensure that we don't misalign comparison boundaries by compounding errors
#Not sure this is ultimately successful however.
        if orig_x != 0 or orig_y != 0:
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

    def print_tree(self, node=None, level=0, path=''):
        if node is None:
            node = self.root

        x0, y0, x1, y1 = node.box
        width, height = x1 - x0, y1 - y0

        print(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{node.phash},quality {node.quality}")

        for index, child in enumerate(node.children):
            new_path = f"{path}{index+1}-" if path else f"{index+1}-"
            self.print_tree(child, level + 1, new_path)

if __name__ == "__main__":
    if not (len(sys.argv) == 3 or len(sys.argv) == 5):
        print("Usage: python encode_file_to_depth.py image_path max_depth [orig_x] [orig_y]")
        sys.exit(1)

    image_path = sys.argv[1]
    max_depth = int(sys.argv[2])
    
    if len(sys.argv) == 5:
        orig_x = int(sys.argv[3])
        orig_y = int(sys.argv[4])
        quad_tree = QuadTree(image_path, max_depth, orig_x, orig_y)
    else:
        quad_tree = QuadTree(image_path, max_depth, 0, 0)
    
    quad_tree.print_tree()
