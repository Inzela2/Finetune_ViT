import os
from PIL import Image
from transformers import CLIPProcessor
import torch
from tqdm import tqdm

# Define paths
dataset_dir = "C:/Users/asus/Downloads/CLIP/Dataset/CUB_200_2011"
images_dir = os.path.join(dataset_dir, "images")
image_labels_file = os.path.join(dataset_dir, "image_class_labels.txt")  # File containing image IDs and their labels
images_file = os.path.join(dataset_dir, "images.txt")  # File containing image IDs and their paths

# Initialize processor
processor = CLIPProcessor.from_pretrained('C:/Users/asus/Downloads/CLIP')


# Load and preprocess images and labels
def load_images_and_labels(images_file, image_labels_file):
    # Read image paths and labels
    image_paths = {}
    with open(images_file, "r") as f:
        for line in f:
            image_id, image_path = line.strip().split()
            image_paths[image_id] = os.path.join(images_dir, image_path)

    image_labels = {}
    with open(image_labels_file, "r") as f:
        for line in f:
            image_id, label = line.strip().split()
            image_labels[image_id] = label

    # Preprocess images and labels
    data = []
    for image_id in tqdm(image_paths.keys()):
        image_path = image_paths[image_id]
        label = image_labels[image_id]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess image and label (label as text for simplicity)
        inputs = processor(text=[label], images=[image], return_tensors="pt", padding=True)
        data.append(inputs)

    return data


# Load and preprocess the data
dataset = load_images_and_labels(images_file, image_labels_file)

print("Dataset preprocessing completed!")
