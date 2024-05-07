import cv2
from PIL import Image
import numpy as np
import os

input_path = 'coronal_dataset/ControlNet/mri'
output_path = 'coronal_dataset/ControlNet/canny'

# Get a list of all files in the input directory
image_files = os.listdir(input_path)

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

from tqdm import tqdm 
# Set Canny edge detection thresholds
low_threshold = 100
high_threshold = 200

# Loop over each image file
for file_name in tqdm(image_files):
    # Read the image
    image_path = os.path.join(input_path, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Canny edge detection
    canny_image = cv2.Canny(image, low_threshold, high_threshold)
    
    # Convert the image to RGB format
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    
    # Create a PIL Image object
    canny_image_pil = Image.fromarray(canny_image)
    
    # Save the image to the output path
    output_file_path = os.path.join(output_path, file_name)
    canny_image_pil.save(output_file_path)
