import cv2
import sys
import os
from enum import Enum
# Initialize debug_mode based on command line arguments
debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')

class Matched(Enum):
    YES = 1
    NO = 2
    UNKNOWN = 3


class TreeNode:
    def __init__(self, line, is_root=False):
        parts = line.split(',') if line else []
        self.path = "" if is_root else parts[0]
        self.hash = parts[9] if parts else None
        self.line = line
        self.children = {}
        self.removed = False
        self.ham_distance = -1
        self.matched = Matched.UNKNOWN
        self.purge = False #Will be set to true later if all subnodes don't match


    def add_child(self, path_segment, child_node):
        self.children[path_segment] = child_node

    def should_purge(self):
        print(f"should_purge entered {str(self.path)} matched {self.matched}")
        # Check if the current node should be purged
        if self.matched != Matched.NO:
            print(f"  1 Matched(green) - don't look further than this {str(self.path)} matched={self.matched} ")
            return False
        
        #print(f"  3 red.  Number of children {len(self.children.values())} ")
        # Recursively check if all children should be purged
        for child in self.children.values():
            #print(f"  4 this node {self.path} and iterated child: {child.path} ")
            if not child.should_purge():
                #print(f"  2 NO False child says it shouldn't purge: {str(child.path)} ")
                return False
            
        # If the node and all its children should be purged,
        # set purge to True and return True
        self.purge = True
        print(f"  Purging this node {str(self.path)} which has hamming {self.ham_distance}")
        return True
    
    def purge_tree(self):
        self.should_purge()
        for child in self.children.values():
            child.purge_tree()

    def print_tree(self, file, unpurged_only):
        
        parts = self.line.split(',')
        x0, y0, x1, y1 = map(int, parts[2:6])
        quality = parts[10]
        level = parts[1]
        path = parts[0]
        width, height = x1 - x0, y1 - y0
        
        if unpurged_only:            
            if self.purge:
                #explicitly stop recursion into sub elements although they /should/ all be purged=True
                return
        #print(f"print_tree Called with path {str(path)}")
        
        file.write(f"{path},{level},{x0},{y0},{x1},{y1},{width},{height},pdq,{self.hash},{quality}\n")

        for child in self.children.values():
            child.print_tree(file, unpurged_only)



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

#removed from the comparison - we've got a match, we don't need to go further.  
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
        print(f"DEBUG Comparing {distance} {node1.path} {node1.hash} {node2.hash}")

        if distance <= threshold:
            mark_as_removed(node1)#removed from comparison because it (or a parent node) is matched - this is distinct from purged
            mark_as_removed(node2)
            node1.matched = Matched.YES
        else:
            node1.matched = Matched.NO
                
        draw_comparison(image, node1, node2, output_path, counter)
        counter[0] += 1
    
    for key in node1.children:
        if key in node2.children:
            compare_and_output_images(image, node1.children[key], node2.children[key], image_path, output_path, threshold, counter, compare_depth)

def main(original_image: str, original_image_qt: str, new_image: str, new_image_qt: str, new_image_output_path: str, threshold:int, compare_depth:int):
    
    if not os.path.exists(new_image_output_path):
        os.makedirs(new_image_output_path)
    
    tree1 = parse_file_to_tree(original_image_qt)
    tree2 = parse_file_to_tree(new_image_qt)
    
    image = cv2.imread(original_image) #caution of rotated images that cv2 doesn't handle
 
    compare_and_output_images(image, tree1, tree2, original_image, new_image_output_path, threshold, [0], compare_depth)
    
    # In non-debug mode, save the last image again to ensure the final state is preserved
    if not debug_mode:
        draw_comparison(image, tree1, tree2, new_image_output_path, [-1])  # Use a unique counter to indicate the last image

    tree1.purge_tree()#remove all the nodes that are unmatched AND all children are also unmatched

    with open('tmp.purged.qt', 'w') as f:
        tree1.print_tree(f, True) #write the newly purged tree to 


if __name__ == "__main__":
    if len(sys.argv) != 5 :
        print("Usage: python iterative_qtree_comparison.py original_image new_image threshold depth [--debug]")
        print("--debug will emit a series of images that show the build up of the quad tree and comparision process - may be a *lot* of images")
        print("Expects a qt file with a .qt extension that is <image>.hoprs and will overwrite <new_image>.hoprs with a pruned tree")
        sys.exit(1)

    original_image  = sys.argv[1]
    original_image_qt = original_image + ".qt"
    print(f"original image {original_image}")
    print(f"original image QT {original_image_qt}")
    
    new_image  = sys.argv[2]
    new_image_qt  = new_image + ".qt"
    print(f"new image {new_image}")
    print(f"new image QT {new_image_qt}")
    
    new_image_output_path = "output_" + new_image
    print(f"new image output {new_image_output_path}")
    
    #Concatenate the two HOPRS files
    # Open file1.txt and file2.txt in read mode, and file3.txt in write mode
    with open(original_image + '.hoprs', 'r') as file1, open(new_image + '.hoprs', 'r') as file2, open(os.path.basename(new_image_output_path) + '.hoprs', 'w') as file3:
        # Read and write the contents of file1 to file3
        contents_of_file1 = file1.read()
        file3.write(contents_of_file1)
        
        # Optionally, add a newline or separator if needed
        file3.write('\n')  # This ensures there is a newline separating the contents of file1 and file2
        # Read and write the contents of file2 to file3
        contents_of_file2 = file2.read()
        file3.write(contents_of_file2)

    
    threshold = int(sys.argv[3])
    compare_depth = int(sys.argv[4])
    
    main(original_image, original_image_qt, new_image, new_image_qt, new_image_output_path, threshold, compare_depth)
