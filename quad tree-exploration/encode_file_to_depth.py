import sys
from PIL import Image as PILImage  # Make sure to import PIL.Image for conversion if needed
import imagehash
import hashlib #cryptographic hashes
import pdqhash
import cv2


def hash_string_to_hex(input_string):
    # Encode the string to bytes
    input_bytes = input_string.encode('utf-8')
    # Create a SHA-256 hash object
    hasher = hashlib.sha256()
    # Update the hash object with the bytes to hash
    hasher.update(input_bytes)
    # Convert the hash to a hexadecimal string
    hex_hash = hasher.hexdigest()
    
    return hex_hash



class QuadTreeNode:
    def __init__(self, image, box, depth):
        self.box = box  # (x0, y0, x1, y1)
        self.children = []  # List of child nodes
        self.depth = depth
        # Since pdqhash.compute expects a numpy.ndarray, pass the segment directly

        # Extract segment using OpenCV slicing (image is a numpy.ndarray)
        segment = image[box[1]:box[3], box[0]:box[2]]
        segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        vector, quality = pdqhash.compute(segment_rgb)

        self.phash = bits_to_hex(vector)


        coords = f"{box[0]},{box[1]},{box[2]},{box[3]}".replace(',','.')
        filename = f"{image_path}_tmp_segment_{coords}.jpg".replace(',','.')
        hashpart = str(hash_string_to_hex(coords)).replace(',','.')
        filename = f"tmp.uniq.{depth}.{hashpart}_pdq.{self.phash}_{filename}" #add a hash so that the segments sort better
        cv2.imwrite(filename, segment , [int(cv2.IMWRITE_JPEG_QUALITY), 95])#just for debug. Use shell : for file in tmp.*;do  rm $file; done

    def is_leaf_node(self):
        # A node is a leaf if it has no children
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
            
            # Ensure the image was loaded
            if image is None:
                print("Error opening the image file. Please check the path and try again.")
                sys.exit(1)
        except IOError:
            print("Error opening the image file. Please check the path and try again.")
            sys.exit(1)
        
        if not ((orig_x == 0) or (orig_y == 0)):
            #print(f"resizing to {orig_x},{orig_y}")
            #If we've asked this to manually resample this to a larger size
            image = cv2.resize(image, (orig_x ,orig_y), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{image_path}_tmp_resized.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])#just for debug

        # Use shape to get image dimensions
        height, width = image.shape[:2]
        self.root = self.split_image(image, (0, 0, width, height), 1)

    def split_image(self, image, box, depth):
        x0, y0, x1, y1 = box
        node = QuadTreeNode(image, box, depth)
        
        width = x1 - x0
        height = y1 - y0
    
        # Stop condition
        if depth > self.max_depth:
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
            child_node = self.split_image(image, segment,depth+1)
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
        #Consider outputting to a file here ot use the encode shellscript
        
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
    if not (len(sys.argv) == 3 or len(sys.argv) == 5) :
        print("Usage: python encode_file_to_depth.py image_path max_depth [orig_x] [orig_y]") #orig_x and y dimensions are those of the original image
        sys.exit(1)

    image_path = sys.argv[1]
    max_depth = int(sys.argv[2])
    
    if (len(sys.argv) == 5):
        orig_x = int(sys.argv[3])
        orig_y = int(sys.argv[4])
        quad_tree = QuadTree(image_path, max_depth, orig_x, orig_y)
    else:
        quad_tree = QuadTree(image_path, max_depth, 0, 0)
    quad_tree.print_tree()
