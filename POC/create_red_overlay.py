import argparse
from PIL import Image

def create_red_overlay(original_image_path, mask_image_path, output_image_path, translucence=50):
    # Load the original photograph
    original_image = Image.open(original_image_path)

    # Load the second image (mask) and convert it to a transparency mask
    mask_image = Image.open(mask_image_path).convert('L')

    # Adjust the translucence of the mask
    # The translucence value affects the opacity; 255 is fully opaque, 0 is fully transparent
    translucence_value = int(255 * (translucence / 100))
    semi_transparent_mask = mask_image.point(lambda p: translucence_value if p > 0 else 0)

    # Create a red overlay image
    red_overlay = Image.new('RGB', original_image.size, color=(255, 0, 0))

    # Apply the semi-transparent mask to the red overlay to control the transparency
    red_overlay.putalpha(semi_transparent_mask)

    # Ensure the original image is in RGBA mode for alpha compositing
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')

    # Blend the red overlay with the original image
    final_image = Image.alpha_composite(original_image, red_overlay)

    # Save the final image
    final_image.save(output_image_path)
    print(f"Overlay image created successfully with {translucence}% translucence: {output_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a translucent red overlay on an image based on a mask with adjustable translucence.')
    parser.add_argument('original_image', help='Path to the original image file')
    parser.add_argument('mask_image', help='Path to the mask image file')
    parser.add_argument('output_image', help='Path for saving the output image with red overlay')
    parser.add_argument('--translucence', type=int, default=50, help='Translucence level of the red overlay (0-100, default 50)')

    args = parser.parse_args()

    create_red_overlay(args.original_image, args.mask_image, args.output_image, args.translucence)
