import sys
from PIL import Image as PILImage  # Make sure to import PIL.Image for conversion if needed
import imagehash
import pdqhash
import cv2


class QuadTreeNode:
    def __init__(self, image, box):
        self.box = box  # (x0, y0, x1, y1)
        self.children = []  # List of child nodes

        # Extract segment using OpenCV slicing (image is a numpy.ndarray)
        segment = image[box[1]:box[3], box[0]:box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)

        # Since pdqhash.compute expects a numpy.ndarray, pass the segment directly
        vector, quality = pdqhash.compute(segment_rgb)

        self.phash = bits_to_hex(vector)

    def is_leaf_node(self):
        # A node is a leaf if it has no children
        return len(self.children) == 0

class QuadTree:
    def __init__(self, image_path, min_dimension=32):
        self.root = None
        self.min_dimension = min_dimension
        self.image_path = image_path
        self.build_tree()

    def build_tree(self):
        try:
            image = cv2.imread(self.image_path)
            
            # Ensure the image was loaded
            if image is None:
                print("Error opening the image file. Please check the path and try again.")
                sys.exit(1)
        except IOError:
            print("Error opening the image file. Please check the path and try again.")
            sys.exit(1)
        
        # Use shape to get image dimensions
        height, width = image.shape[:2]
        self.root = self.split_image(image, (0, 0, width, height))

    def split_image(self, image, box):
        x0, y0, x1, y1 = box
        node = QuadTreeNode(image, box)
        
        width = x1 - x0
        height = y1 - y0

        # Stop condition
        if width <= self.min_dimension or height <= self.min_dimension:
            return node

        half_width = width // 2
        half_height = height // 2

        segments = [
            (x0, y0, x0 + half_width, y0 + half_height),  # Top-left
            (x0 + half_width, y0, x1, y0 + half_height),  # Top-right
            (x0, y0 + half_height, x0 + half_width, y1),  # Bottom-left
            (x0 + half_width, y0 + half_height, x1, y1)   # Bottom-right
        ]

        for segment in segments:
            child_node = self.split_image(image, segment)
            node.children.append(child_node)

        return node

    def print_tree(self, node=None, level=0, path=''):
        if node is None:
            node = self.root
        
        # Calculate size of the segment
        x0, y0, x1, y1 = node.box
        width = x1 - x0
        height = y1 - y0

        # Format the output as CSV
        # Note: Removed the additional dash for path formatting in CSV output for clarity
        print(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},'pdq',{node.phash}")
        
        for index, child in enumerate(node.children):
            new_path = f"{path}{index + 1}-" if path else f"{index + 1}-"
            self.print_tree(child, level + 1, new_path)

def bits_to_hex(bits):
    """
    Convert a NumPy array of bits to a hexadecimal string.
    
    Args:
    - bits: NumPy array of bits.
    
    Returns:
    - A string representing the hexadecimal representation of the bits.
    """
    # Ensure the array is of type int, and then convert to a binary string without the '0b' prefix
    binary_string = ''.join(str(bit) for bit in bits)
    # Convert the binary string to an integer, and then format that integer as hex
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python encode_file.py image_path smallest_chunk")
        sys.exit(1)

    image_path = sys.argv[1]
    min_dimension = int(sys.argv[2])
    quad_tree = QuadTree(image_path, min_dimension)
    quad_tree.print_tree()
