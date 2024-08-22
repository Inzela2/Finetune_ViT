from transformers import CLIPModel, CLIPProcessor

# Load the CLIP model and processor
model = CLIPModel.from_pretrained('C:/Users/asus/Downloads/CLIP')
processor = CLIPProcessor.from_pretrained('C:/Users/asus/Downloads/CLIP')

print("CLIP model and processor loaded successfully!")
