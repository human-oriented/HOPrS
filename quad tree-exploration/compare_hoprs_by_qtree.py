import sys

class TreeNode:
    def __init__(self, line, is_root=False):
        parts = line.split(',') if line else []
        self.path = "" if is_root else parts[0]
        self.hash = parts[-1] if parts else None  # The last part is the perceptual hash for all nodes, including the root.
        self.line = line
        self.children = {}
        if is_root:
            print(f"Initialized root node with hash {self.hash}")
        else:
            print(f"Created node path {self.path} with hash {self.hash}")

    def add_child(self, path_segment, child_node):
        self.children[path_segment] = child_node

def parse_file_to_tree(filepath):
    print(f"Building tree from {filepath}...")
    with open(filepath, 'r') as f:
        first_line = next(f, None)
        if first_line:
            root = TreeNode(first_line.strip(), is_root=True)
        else:
            print("File is empty, no root node created.")
            return None
        
        for line in f:
            parts = line.strip().split(',')
            path_segments = parts[0].split('-')
            current = root
            for segment in path_segments:
                if segment:
                    if segment not in current.children:
                        current.add_child(segment, TreeNode(line.strip()))
                    current = current.children[segment]
    print("Tree built successfully.")
    return root

def hamming_distance(hash1, hash2):
    b1 = bin(int(hash1, 16))[2:].zfill(256)
    b2 = bin(int(hash2, 16))[2:].zfill(256)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))

def compare_and_remove_matching_nodes(node1, node2, threshold):
    if node1.hash is not None and node2.hash is not None:
        distance = hamming_distance(node1.hash, node2.hash)
        if distance <= threshold:
            print(f"Nodes {node1.path} and {node2.path} removed, Hamming distance {distance} within threshold {threshold}")
            return True
    keys_to_remove = []
    for key, child1 in node1.children.items():
        child2 = node2.children.get(key)
        if child2 and compare_and_remove_matching_nodes(child1, child2, threshold):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del node1.children[key]
        del node2.children[key]
    return False

def output_remaining_nodes(node, file, path=""):
    if node.line and path:
        file.write(node.line + '\n')
    for segment, child in node.children.items():
        next_path = f"{path}-{segment}" if path else segment
        output_remaining_nodes(child, file, next_path)

def main(file1_path, file2_path, output_path, threshold):
    tree1 = parse_file_to_tree(file1_path)
    tree2 = parse_file_to_tree(file2_path)

    print("Comparing trees and removing matching nodes...")
    compare_and_remove_matching_nodes(tree1, tree2, threshold)
    print("Comparison complete. Generating output...")

    with open(output_path, 'w') as output_file:
        output_remaining_nodes(tree2, output_file)
    print(f"Output successfully written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py file1_path file2_path output_path threshold")
        sys.exit(1)

    file1_path, file2_path, output_path = sys.argv[1:4]
    threshold = int(sys.argv[4])
    main(file1_path, file2_path, output_path, threshold)


