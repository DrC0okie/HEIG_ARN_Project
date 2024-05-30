import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
TF_ENABLE_ONEDNN_OPTS=0

# Répertoires d'entrée et de sortie
relative_input_directory = 'cropped'
relative_output_directory = 'cropped_augmented'

# Obtenir le chemin absolu du répertoire où se trouve le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Créer le chemin absolu des répertoires cible
input_dir = os.path.join(script_dir, relative_input_directory)
output_dir = os.path.join(script_dir, relative_output_directory)

os.makedirs(output_dir, exist_ok=True)

# Configurer les transformations de data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.6, 1.4],
)

# Générer et sauvegarder les images dérivées
def augment_and_save_image(image_path, output_dir, datagen, num_augmented_images=5):
    img = Image.open(image_path).convert('RGB')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    i = 0
    
    for batch in datagen.flow(x, batch_size=1):
        if i >= num_augmented_images:
            break
        
        # Déterminer le type de transformation
        transformation = []
        if datagen.rotation_range:
            transformation.append("rotation")
        if datagen.width_shift_range or datagen.height_shift_range:
            transformation.append("shift")
        if datagen.zoom_range:
            transformation.append("zoom")
        if datagen.brightness_range:
            transformation.append("brightness")
        
        transformation_suffix = "_".join(transformation)
        new_filename = f"{base_filename}_{transformation_suffix}_{i}.png"
        new_filepath = os.path.join(output_dir, new_filename)
        
        augmented_img = array_to_img(batch[0])
        augmented_img.save(new_filepath)
        i += 1

# Parcourir les images dans le répertoire d'entrée et générer les images dérivées
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        augment_and_save_image(input_path, output_dir, datagen)
