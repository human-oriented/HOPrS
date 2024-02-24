import csv
from PIL import Image, ImageDraw
import sys

def draw_quadtree_boxes(csv_filepath, output_image_filepath):
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        root_node = next(csv_reader)  # Assuming the first line is the root node
        # Create a white image with dimensions specified in the root node
        image_width, image_height = int(root_node[6]), int(root_node[7])
        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Process each node in the CSV file
        for node in csv_reader:
            x0, y0, x1, y1 = map(int, node[2:6])
            # Draw a black box for the node
            draw.rectangle([x0, y0, x1, y1], outline='black')
        
        # Save the image
        image.save(output_image_filepath)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_image_file>")
        sys.exit(1)

    csv_filepath = sys.argv[1]
    output_image_filepath = sys.argv[2]

    draw_quadtree_boxes(csv_filepath, output_image_filepath)
    print(f"Output image saved to {output_image_filepath}")
