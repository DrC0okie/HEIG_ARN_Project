from PIL import Image, ImageOps
import os

relative_input_directory = 'originals'
relative_output_directory = 'cropped_images'

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path of the target directories
input_dir = os.path.join(script_dir, relative_input_directory)
output_dir = os.path.join(script_dir, relative_output_directory)

os.makedirs(output_dir, exist_ok=True)

target_height = 1360

def crop_image(image_path, output_path, target_height):
    with Image.open(image_path) as img:
        width, height = img.size

        # If the height is greater than the target height, crop from the top
        if height > target_height:
            cropped_img = img.crop((0, 0, width, target_height))
        else:
            # If the height is less than the target height, pad the bottom with black pixels
            cropped_img = Image.new("RGB", (width, target_height), (0, 0, 0))
            cropped_img.paste(img, (0, 0))

        cropped_img.save(output_path)

def process_directory(input_dir, output_dir, target_height):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                crop_image(image_path, output_path, target_height)

# Run the script
process_directory(input_dir, output_dir, target_height)

print("Cropping completed.")
