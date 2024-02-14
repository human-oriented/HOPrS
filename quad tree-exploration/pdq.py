"""Portable version of single pdq hash

Outputs a 256bit perceptual PDQ hash.  See the Facebook ThreatExchange project
"""
import sys
from PIL import Image as PILImage  # Make sure to import PIL.Image for conversion if needed
import imagehash
import pdqhash
import cv2

def bits_to_hex(bits):
    """
    Convert a NumPy array of bits to a hexadecimal string.
    
    Args:
    - bits: NumPy array of bits.
    
    Returns:
    - A string representing the hexadecimal representation of the bits.
    """
    # Ensure the array is of type int, and then convert to a binary string without the '0b' prefix
    binary_string = ''.join(str(bit) for bit in bits)
    # Convert the binary string to an integer, and then format that integer as hex
    hex_string = format(int(binary_string, 2), 'x')
    return hex_string

def calc_hash(image_path):
    
    try:
        image = cv2.imread(image_path)
        
        # Ensure the image was loaded
        if image is None:
            print("Error opening the image file. Please check the path and try again.")
            sys.exit(1)
    except IOError:
        print("Error opening the image file. Please check the path and try again.")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Since pdqhash.compute expects a numpy.ndarray, pass the segment directly
    vector, quality = pdqhash.compute(image_rgb)

    print(bits_to_hex(vector))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdq.py image_path ")
        sys.exit(1)

    image_path = sys.argv[1]
    calc_hash(image_path)
