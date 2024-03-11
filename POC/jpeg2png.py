import sys
from PIL import Image

def convert_jpeg_to_png(jpeg_file_path, png_file_path):
    try:
        # Open the JPEG file
        with Image.open(jpeg_file_path) as image:
            # Convert the image to PNG and save it
            image.save(png_file_path, 'PNG')
        print(f"Successfully converted {jpeg_file_path} to {png_file_path}")
    except Exception as e:
        print(f"Error converting {jpeg_file_path} to PNG: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jpeg2png.py <source_jpeg_file> <destination_png_file>")
    else:
        source_jpeg = sys.argv[1]
        destination_png = sys.argv[2]
        convert_jpeg_to_png(source_jpeg, destination_png)
