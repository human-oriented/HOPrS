import sys

def validate_quadtree_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove newline characters and split by comma
    entries = [line.strip().split(',') for line in lines]

    # Extract only the path part of each entry
    paths = [entry[0] for entry in entries]

    # Check for completeness by ensuring all prefixes of each path exist
    missing_paths = set()
    for path in paths:
        # Check each prefix of the path, considering paths with and without trailing hyphens
        parts = path.strip('-').split('-')  # Temporarily remove trailing hyphen for prefix checking
        while parts:
            prefix_without_hyphen = '-'.join(parts)
            prefix_with_hyphen = prefix_without_hyphen + '-'
            # Check both representations of the prefix (with and without trailing hyphen)
            if prefix_without_hyphen not in paths and prefix_with_hyphen not in paths:
                missing_paths.add(prefix_without_hyphen)
            parts.pop()  # Remove the last part to check the next level prefix

    if missing_paths:
        print(f"Quadtree is incomplete. Missing nodes for paths: {', '.join(sorted(missing_paths))}")
        return False
    else:
        print("Quadtree is complete.")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_quadtree.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    validate_quadtree_file(filename)
