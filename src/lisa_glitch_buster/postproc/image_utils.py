from PIL import Image
import os
from PIL import Image
import os


def concat_images(image_paths, final_path):
    images = [Image.open(path) for path in image_paths]

    # Find the maximum width among all images
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # Create a new white image with the calculated dimensions
    new_image = Image.new('RGB', (max_width, total_height), color='white')

    # Paste each image centered horizontally
    y_offset = 0
    for img in images:
        x_offset = (max_width - img.width) // 2  # Calculate horizontal center
        new_image.paste(img, (x_offset, y_offset))
        y_offset += img.height

    # Save the concatenated image
    new_image.save(final_path)

    # Remove the old images
    for path in image_paths:
        os.remove(path)