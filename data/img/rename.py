import os

def rename_png_files(relative_directory, base_name):
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the absolute path of the target directory
    target_dir = os.path.join(script_dir, relative_directory)
    
    # Get a list of all PNG files in the directory
    png_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    
    # Sort the files to maintain a consistent order
    png_files.sort()
    
    # Rename each file
    for i, filename in enumerate(png_files, start=1):
        new_name = f"{base_name}{i}.png"
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Example usage
relative_directory = 'py'
base_name = 'py'
rename_png_files(relative_directory, base_name)
