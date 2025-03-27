import os
import random
import shutil

# Path to the directory containing folders with images
base_dir = 'data/dash_cam/processed_frames'

# File to track processed folders
processed_folders_file = os.path.join(base_dir, 'processed_folders.txt')

# Load the list of already processed folders
if os.path.exists(processed_folders_file):
    with open(processed_folders_file, 'r') as f:
        processed_folders = set(f.read().splitlines())
else:
    processed_folders = set()

# Iterate through each subfolder in the base directory
for subfolder_name in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    
    # Skip if the folder is already processed
    if subfolder_name in processed_folders:
        print(f"Skipping already processed folder: {subfolder_name}")
        continue

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Get a list of all image files in the subfolder
        images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        print(f"Found {len(images)} images in {subfolder_path}")

        # Calculate the number of images to keep (1/3 of the total)
        num_to_keep = len(images) // 3
        print(f"Keeping {num_to_keep} images out of {len(images)}")

        # Randomly select images to keep
        images_to_keep = set(random.sample(images, num_to_keep))
        
        # Remove images not in the selected set
        for image in images:
            if image not in images_to_keep:
                image_path = os.path.join(subfolder_path, image)
                print(f"Deleting {image_path}")
                os.remove(image_path)
        
        # Mark the subfolder as processed after processing
        processed_folders.add(subfolder_name)

# Save the updated list of processed folders
with open(processed_folders_file, 'w') as f:
    f.write('\n'.join(processed_folders))

print("Random selection and deletion completed.")
