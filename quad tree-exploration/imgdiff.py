import sys
from PIL import Image, ImageChops

def absolute_difference(image_path1, image_path2, output_path, exaggerate=False):
    """
    Calculates the absolute difference between two images, optionally exaggerates the difference,
    and saves it as a new image.

    Parameters:
    - image_path1: Path to the first image.
    - image_path2: Path to the second image.
    - output_path: Path where the output image will be saved.
    - exaggerate: Boolean indicating whether to exaggerate the difference.
    """
    try:
        # Load the images
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        # Check if the images are the same size
        if img1.size != img2.size:
            raise ValueError("Images are not the same size and cannot be processed.")

        # Calculate the absolute difference
        diff = ImageChops.difference(img1, img2)

        if exaggerate:
            # Apply a scaling factor to exaggerate the differences
            diff = diff.point(lambda x: x * 3)  # Adjust the multiplier as needed

        # Save the result
        diff.save(output_path)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python script.py <image1_path> <image2_path> <output_image_path> [exaggerate]")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    output_path = sys.argv[3]
    exaggerate = len(sys.argv) == 5 and sys.argv[4].lower() == "exaggerate"

    absolute_difference(image_path1, image_path2, output_path, exaggerate)
