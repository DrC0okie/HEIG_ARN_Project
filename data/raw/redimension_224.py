from PIL import Image, ImageOps
import numpy as np
import os
import cv2  # OpenCV is required for edge detection

relative_input_directory = 'originals'
relative_output_directory = 'ready'

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path of the target directories
input_dir = os.path.join(script_dir, relative_input_directory)
output_dir = os.path.join(script_dir, relative_output_directory)

os.makedirs(output_dir, exist_ok=True)

tile_size = (448, 448)
left_margin = 32

# Set personalized code density thresholds for each class
density_thresholds = {
    'cpp': 0.038,
    'hs': 0.012,
    'py': 0.027
}

# Counters for each programming language
cpp_count = 0
hs_count = 0
py_count = 0

def process_image(image_path, output_dir, tile_size, left_margin, density_thresholds):
    global cpp_count, hs_count, py_count
    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate the number of whole tiles that fit in the image height
        num_tiles = height // tile_size[1]

        for i in range(num_tiles):
            for j in range(2):  # Loop over the first two columns
                # Calculate the coordinates for the current tile
                left = left_margin + j * tile_size[0]
                upper = i * tile_size[1]
                right = left + tile_size[0]
                lower = upper + tile_size[1]

                if right > width or lower > height:
                    continue  # Skip if the tile is out of the image bounds

                # Crop the tile from the image
                tile = img.crop((left, upper, right, lower))

                # Convert tile to grayscale and then to a numpy array
                gray_tile = tile.convert("L")
                gray_array = np.array(gray_tile)

                # Apply Canny edge detection
                edges = cv2.Canny(gray_array, 100, 200)

                # Count the number of edge pixels
                edge_pixels = np.sum(edges > 0)
                total_pixels = gray_array.size
                density = edge_pixels / total_pixels

                # Determine the label of the image
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                if name.startswith('cpp'):
                    label = 'cpp'
                elif name.startswith('hs'):
                    label = 'hs'
                elif name.startswith('py'):
                    label = 'py'
                else:
                    continue

                # Save the tile only if the density exceeds the personalized threshold
                if density > density_thresholds[label]:
                    # Define the output path for the tile
                    tile_output_path = os.path.join(output_dir, f"{name}_tile_{i+1}_{j+1}{ext}")

                    # Resize the tile to the target size and save it
                    tile = tile.resize((224, 224), Image.LANCZOS)
                    tile.save(tile_output_path)

                    # Update counters
                    if label == 'cpp':
                        cpp_count += 1
                    elif label == 'hs':
                        hs_count += 1
                    elif label == 'py':
                        py_count += 1

def parse_directory(input_dir, output_dir, tile_size, left_margin, density_thresholds):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                process_image(image_path, output_dir, tile_size, left_margin, density_thresholds)

# Run the script
parse_directory(input_dir, output_dir, tile_size, left_margin, density_thresholds)

# Print the counts of generated images for each programming language
print(f"C++ images generated: {cpp_count}")
print(f"Haskell images generated: {hs_count}")
print(f"Python images generated: {py_count}")
