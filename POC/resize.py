import sys
from PIL import Image
import os

def resize_image(input_path, scale_percentage):
    """
    Resizes an image by a specified percentage and saves the new image with a meaningful filename
    and a high-quality setting.

    Parameters:
    - input_path: Path to the input image.
    - scale_percentage: The percentage to scale the image by.
    """
    try:
        # Load the image
        img = Image.open(input_path)

        # Calculate the new dimensions
        scale_factor = scale_percentage / 100.0
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Construct the new filename
        base_name, ext = os.path.splitext(os.path.basename(input_path))
        output_filename = f"{base_name}_resized_{scale_percentage:0>2}pc_{new_width:0>4}x{new_height:0>4}{ext}"
        output_path = os.path.join(os.path.dirname(input_path), output_filename)

        # Determine the format for saving based on the original extension
        if ext.lower() in ['.jpg', '.jpeg']:
            # Save the resized image with high quality for JPEG
            resized_img.save(output_path, quality=95)
        else:
            # Save other formats normally (consider using 'optimize=True' for PNG)
            resized_img.save(output_path)

        print(f"Resized image saved as {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize.py <image_path> <scale_percentage>")
        sys.exit(1)

    image_path = sys.argv[1]
    scale_percentage = float(sys.argv[2])

    resize_image(image_path, scale_percentage)
