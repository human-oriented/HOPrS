import cv2
import sys

def parse_differences_file(filepath):
    """
    Parses the differences file to extract node levels and bounding boxes.
    Returns a list of tuples containing the node level and bounding box coordinates.
    """
    differences = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            node_path = parts[0]
            level = node_path.count('-')  # Level is determined by the number of dashes
            x0, y0, x1, y1 = map(int, parts[2:6])
            differences.append((level, (x0, y0, x1, y1)))
    return differences

def draw_rectangles(image_path, differences):
    """
    Draws rectangles on the image based on the differences extracted from the file.
    Each rectangle is colored based on the node level.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error opening the image file. Please check the path and try again.")
        return

    # Define colors for each level (as BGR tuples)
    colors = [
        (255, 0, 0),  # Level 0: Blue
        (0, 255, 0),  # Level 1: Green
        (0, 0, 255),  # Level 2: Red
        (255, 255, 0),# Level 3: Cyan
        # Add more colors as needed
    ]

    for level, (x0, y0, x1, y1) in differences:
        color = colors[level % len(colors)]  # Cycle through colors if there are more levels than colors
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)  # Draw rectangle

    # Display the result
    cv2.imshow('Differences Highlighted', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the result to a file
    # cv2.imwrite('highlighted_image.jpg', image)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py differences_file_path image_path")
        sys.exit(1)

    differences_file_path = sys.argv[1]
    image_path = sys.argv[2]

    differences = parse_differences_file(differences_file_path)
    draw_rectangles(image_path, differences)

