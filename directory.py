import os
import shutil
import random

# Define the paths
source_dir = r"C:\Users\asus\Downloads\CLIP\Dataset\CUB_200_2011\images"
destination_dir = r"C:\Users\asus\Downloads\CLIP\Dataset\CUB_200_2011\test"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over each bird category folder in the source directory
for bird_folder in os.listdir(source_dir):
    bird_folder_path = os.path.join(source_dir, bird_folder)

    if os.path.isdir(bird_folder_path):
        # Get all image files in the bird folder
        images = os.listdir(bird_folder_path)

        # Randomly select 4 images
        selected_images = random.sample(images, 4)

        # Create a corresponding subfolder in the destination directory
        dest_bird_folder_path = os.path.join(destination_dir, bird_folder)
        if not os.path.exists(dest_bird_folder_path):
            os.makedirs(dest_bird_folder_path)

        # Move the selected images to the destination folder
        for image in selected_images:
            source_image_path = os.path.join(bird_folder_path, image)
            dest_image_path = os.path.join(dest_bird_folder_path, image)

            shutil.move(source_image_path, dest_image_path)

print("Images have been moved successfully.")
