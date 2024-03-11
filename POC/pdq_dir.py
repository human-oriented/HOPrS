import sys
import cv2
import glob
import pdqhash
from PIL import Image as PILImage  # Make sure to import PIL.Image for conversion if needed

def bits_to_hex(bits):
    """
    Convert a NumPy array of bits to a hexadecimal string.
    
    Args:
    - bits: NumPy array of bits.
    
    Returns:
    - A string representing the hexadecimal representation of the bits.
    """
    binary_string = ''.join(str(bit) for bit in bits)
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string

def calc_hash(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error opening the image file {image_path}. Please check the path and try again.", file=sys.stderr)
            return None
    except IOError:
        print(f"Error opening the image file {image_path}. Please check the path and try again.", file=sys.stderr)
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vector, quality = pdqhash.compute(image_rgb)
    return bits_to_hex(vector), quality

if __name__ == "__main__":
    # Search for jpg and jpeg files in the current directory
    file_patterns = ['./**/*.jpg', './**/*.jpeg']
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    # Print CSV header
    print("filename,pdqhash")
    
    for image_path in files:
        hash_result, quality = calc_hash(image_path)
        if hash_result:
            print(f"{image_path},{hash_result},{quality}")
