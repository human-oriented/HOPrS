import cv2
import sys
import os

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
    
    parts = node1.line.split(',')  #coordinates relative to NODE 1
    x0, y0, x1, y1 = map(int, parts[2:6])
    x = int(x0 + (x1-x0)/2)
    y = int(y0 + (y1-y0)/2)
    
    if node1.removed:
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 4)        
        cv2.putText(image, str(node1.ham_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    else:
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 4)
        cv2.putText(image, str(node1.ham_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
    
    cv2.rectangle(image, (0, 0), (4000,120), (30, 30, 30), -1) #box behind text
    
    text = f"Path: {node1.path} Hash1: {node1.hash} vs Hash2: {node2.hash} counter: {counter} Removed: {node1.removed} "

    cv2.putText(image, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imwrite(f"{output_path}/comparison_{counter:04}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

def compare_and_output_images(image, node1, node2, image_path, output_path, threshold, counter=[0], compare_depth=99):
    current_depth = len(node1.path.split('-'))
    
    if (current_depth-1 >= compare_depth):
        return
    
    if node1.hash and node2.hash:
        if not (node1.removed or node2.removed):
            #print("Comparing:")
            #print(f"Removed?: {node1.removed} {node1.line}")
            #print(f"Removed?: {node2.removed} {node2.line}")
            
            distance = hamming_distance(node1.hash, node2.hash)
            node1.ham_distance = distance
            node2.ham_distance = distance
            print(f"hamming distance is {distance} for path: {node1.path}")
            
            if distance <= threshold:
                #print("distance <= threshold marking as removed")
                mark_as_removed(node1)
                mark_as_removed(node2)
            
            #print("About to draw comparison")
            if image is not None:
                draw_comparison(image, node1, node2, output_path, counter[0])
                counter[0] += 1
            #return
        #else:
            #print("Node has been removed not processing next 2 lines")
            #print(f"Removed?: {node1.removed} {node1.line}")
            #print(f"Removed?: {node2.removed} {node2.line}")        
    else:
        print ("Warning - Missing hash - check your data")
        
    for key in list(node1.children.keys()):
        if key in node2.children:
            compare_and_output_images(image, node1.children[key], node2.children[key], image_path, output_path, threshold, counter,compare_depth)

def main(image_path: str, file1_path: str, file2_path: str, output_path: str, threshold: int, compare_depth: int):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tree1 = parse_file_to_tree(file1_path)
    tree2 = parse_file_to_tree(file2_path)
    image = cv2.imread(image_path)    

    compare_and_output_images(image, tree1, tree2, image_path, output_path, threshold, [0], compare_depth)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python script.py image_path file1_path file2_path depth output_path threshold")
        sys.exit(1)

#depth is the depth to which to compare to.  If you want ot compare as mucha s possible then just specify 
#a large number say 99

    image_path, file1_path, file2_path,    = sys.argv[1:4]
    compare_depth = int(sys.argv[4])
    output_path = sys.argv[5]
    threshold = int(sys.argv[6])
    
    main(image_path, file1_path, file2_path, output_path, threshold, int(compare_depth))
