import cv2
import sys
import os

# Initialize debug_mode based on command line arguments
debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')

class TreeNode:
    def __init__(self, line, is_root=False):
        parts = line.split(',') if line else []
        self.path = "" if is_root else parts[0]
        self.hash = parts[9] if parts else None
        self.line = line
        self.children = {}
        self.removed = False
        self.ham_distance = -1

    def add_child(self, path_segment, child_node):
        self.children[path_segment] = child_node

def parse_file_to_tree(filepath):
    with open(filepath, 'r') as f:
        first_line = next(f, None)
        #discard first line which is details of the cropping or not
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

def draw_comparison(image, node1, node2, output_path, counter):


    parts = node1.line.split(',')
    x0, y0, x1, y1 = map(int, parts[2:6])
    x = int(x0 + (x1 - x0) / 2)
    y = int(y0 + (y1 - y0) / 2)
    
    color = (0, 255, 0) if node1.removed else (0, 0, 255)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 4)
    cv2.putText(image, str(node1.ham_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    
    cv2.rectangle(image, (0, 0), (4000, 120), (30, 30, 30), -1)  # Box behind text
    text = f"Path: {node1.path} Hash1: {node1.hash} vs Hash2: {node2.hash} Counter: {counter[0]} Removed: {node1.removed}"
    cv2.putText(image, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    if not debug_mode and counter[0] != -1:  # Skip saving intermediate images unless in debug mode
        return
    cv2.imwrite(f"{output_path}/comparison_{counter[0]:04}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def compare_and_output_images(image, node1, node2, image_path, output_path, threshold, counter=[0], compare_depth=99):
    if len(node1.path.split('-')) - 1 >= compare_depth:
        return
    
    if node1.hash and node2.hash and not node1.removed and not node2.removed:
        distance = hamming_distance(node1.hash, node2.hash)
        node1.ham_distance = distance
        node2.ham_distance = distance
        
        if distance <= threshold:
            mark_as_removed(node1)
            mark_as_removed(node2)
        
        draw_comparison(image, node1, node2, output_path, counter)
        counter[0] += 1
    
    for key in node1.children:
        if key in node2.children:
            compare_and_output_images(image, node1.children[key], node2.children[key], image_path, output_path, threshold, counter, compare_depth)

def main(image_path: str, file1_path: str, file2_path: str, output_path: str, threshold: int, compare_depth: int):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    tree1 = parse_file_to_tree(file1_path)
    tree2 = parse_file_to_tree(file2_path)
    
    image = cv2.imread(image_path) #caution of rotated images that cv2 doesn't handle
 
    compare_and_output_images(image, tree1, tree2, image_path, output_path, threshold, [0], compare_depth)
    
    # In non-debug mode, save the last image again to ensure the final state is preserved
    if not debug_mode:
        draw_comparison(image, tree1, tree2, output_path, [-1])  # Use a unique counter to indicate the last image

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python script.py image_path file1_path file2_path output_path threshold depth [--debug]")
        print("-debug will emit a series of images that show the build up of the quad tree and comparision process")
        sys.exit(1)
    
    image_path, file1_path, file2_path, output_path = sys.argv[1:5]
    threshold = int(sys.argv[5])
    compare_depth = int(sys.argv[6])
    
    main(image_path, file1_path, file2_path, output_path, threshold, compare_depth)
