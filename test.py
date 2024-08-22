import os
from sklearn.metrics import accuracy_score
import json
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Paths to required files
label_file_path = "C:/Users/asus/Downloads/CLIP/Dataset/CUB_200_2011/image_class_labels"
image_file_path = "C:/Users/asus/Downloads/CLIP/Dataset/CUB_200_2011/images.txt"  # Add this path
model_path = "C:/Users/asus/Downloads/CLIP/Dataset/results/checkpoint-2064"  # Update this path if needed

# Checking if label file exists
if not os.path.exists(label_file_path):
    print(f"File not found: {label_file_path}")
else:
    with open(label_file_path, 'r') as f:
        true_labels = [int(line.split()[1]) - 1 for line in f.readlines()]  # Subtract 1 to match index starting from 0
        print("Label file opened successfully")

# Checking if image file exists
if not os.path.exists(image_file_path):
    print(f"File not found: {image_file_path}")
else:
    with open(image_file_path, 'r') as f:
        image_paths = [line.split()[1] for line in f.readlines()]
        print("Image file opened successfully")

# Load the model and processor
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Define the path to the test images
data_path = "C:/Users/asus/Downloads/CLIP/Dataset/CUB_200_2011/test"

# Initialize lists to store predictions and corresponding true labels
predicted_classes = []

# Predict and collect predictions
for img_path in image_paths:
    img_full_path = os.path.join(data_path, img_path)
    image = Image.open(img_full_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_classes.append(predicted_class_idx)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Output individual predictions with the image path
for img_path, pred_class in zip(image_paths, predicted_classes):
    print(f"Image: {img_path}, Predicted class index: {pred_class}")