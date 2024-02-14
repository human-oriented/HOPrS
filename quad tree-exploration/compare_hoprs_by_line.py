def read_quadtree_file(filepath):
    """
    Reads a quadtree file and returns a dictionary with the node path as the key
    and the entire line as the value.
    """
    imagehashes = {}
    lines = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            node_path = parts[0]
            # Store the entire line for output
            lines[node_path] =line.strip() 
            imagehashes[node_path] = parts[9]
    return imagehashes, lines

def compare_trees(imagehashes1, imagehashes2, lines1, lines2, num_bits):
    """
    Compares two quadtrees and returns a list of lines from tree1 that have different
    perceptual hashes in tree2 or do not exist in tree2.
    """
    differences = []
    for node_path, line in imagehashes1.items():
        if node_path not in imagehashes2 or \
        node_path in imagehashes2 and \
        hamming_distance(imagehashes1[node_path],imagehashes2[node_path] ) < num_bits:    
        #imagehashes1[node_path] != imagehashes2[node_path]):  #hamming distance check here
            differences.append(lines1[node_path])
            #differences.append(line)
    return differences

def main(file1_path, file2_path, output_path, number_bits):
    """
    Main function to read two quadtree files, compare them, and write out the differences.
    """
    imagehashes1, lines1 = read_quadtree_file(file1_path)
    imagehashes2, lines2 = read_quadtree_file(file2_path)
    
    differences = compare_trees(imagehashes1, imagehashes2, lines1, lines2,number_bits)

    with open(output_path, 'w') as output_file:
        for line in differences:
            output_file.write(line + '\n')

    print(f"Differences written to {output_path}")


def hamming_distance(hash1, hash2):
    # Convert hex strings to binary strings
    binary_hash1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    binary_hash2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

    # Calculate Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(binary_hash1, binary_hash2))

    return distance




if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python script.py file1_path file2_path number_bits output_path")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    number_bits = int(sys.argv[3])
    
    output_path = sys.argv[4]

    main(file1_path, file2_path, output_path, number_bits)

