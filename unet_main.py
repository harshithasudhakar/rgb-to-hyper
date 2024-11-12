import os
import shutil
import zipfile
from main import load_model_and_predict  # Import the function

parent_dir = r"C:\Harshi\ECS-II\Dataset" 
rgb_dir = os.path.join(parent_dir, "rgb_micro")
mask_dir = os.path.join(parent_dir, "mask_micro")

"""# Define paths
dataset_path = r"C:\Harshi\ECS-II\Dataset\dataverse_files full.zip"
parent_dir = r"C:\Harshi\ECS-II\Dataset"  # The parent directory where dataverse_files_full will be extracted
extract_dir = os.path.join(parent_dir, "dataverse_files_full")  # Folder where the images will be extracted

# Create directories to store RGB and Mask images directly in the parent directory
rgb_dir = os.path.join(parent_dir, "rgb_micro")
mask_dir = os.path.join(parent_dir, "mask_micro")

# Unzip the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Create the directories if they don't exist
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Iterate through the files in the extracted folder and move them into the correct directories
for root, _, files in os.walk(extract_dir):
    for filename in files:
        if filename.endswith(".png"):
            # Check if it's a mask image (ends with '-1')
            if "-1" in filename:
                shutil.move(os.path.join(root, filename), os.path.join(mask_dir, filename))
            else:
                shutil.move(os.path.join(root, filename), os.path.join(rgb_dir, filename))

print("Images sorted into 'rgb_micro' and 'mask_micro'.")"""

# Call load_model_and_predict with the sorted images
load_model_and_predict(rgb_path=rgb_dir, hsi_path=mask_dir, checkpoint_path=r'C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper\checkpoints')
