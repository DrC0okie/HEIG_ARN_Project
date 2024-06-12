from PIL import Image, ImageOps
import os

relative_input_directory = 'originals/py'
relative_output_directory = 'cropped_224'

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
    
# Create the absolute path of the target directory
input_dir = os.path.join(script_dir, relative_input_directory)
output_dir = os.path.join(script_dir, relative_output_directory)

os.makedirs(output_dir, exist_ok=True)

# Taille cible
target_size = (224, 224)

def resize_and_crop(image_path, output_path, size):
    with Image.open(image_path) as img:
        img = img.crop((32, 32, 704, 704))
        
        
        # Redimensionner la largeur à 512 pixels tout en conservant le ratio d'aspect
        img = img.resize((size[0], int(img.height * size[0] / img.width)), Image.Resampling.LANCZOS)
        
        # Ajouter des bordures si la hauteur est inférieure à 512 pixels
        if img.height < size[1]:
            padding = (0, (size[1] - img.height) // 2, 0, (size[1] - img.height + 1) // 2)
            img = ImageOps.expand(img, padding)
        
        # Rogner au centre si la hauteur est supérieure à 512 pixels
        if img.height > size[1]:
            left = 0
            top = (img.height - size[1]) / 2
            right = size[0]
            bottom = (img.height + size[1]) / 2
            img = img.crop((left, top, right, bottom))
        
        img.save(output_path)

# Parcourir les images dans le répertoire d'entrée
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # Filtrer les fichiers d'image
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        resize_and_crop(input_path, output_path, target_size)
