import os
import torch
from torchvision import transforms
from PIL import Image
import timm
import argparse
import json
import torch.nn as nn

# Set directories
IMAGE_DIR = 'images'
UTILITIES_DIR = 'Utilities'  

def load_image(image_filename):
    """
    Load and preprocess the image.
    """
    # Combine the image directory path and the filename
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    # Define the image transformation for input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    return image

def load_pretrained_weights(model, weights_path, num_classes):
    """
    Load the pre-trained weights into the model and modify the output layer to match the number of classes.
    """
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)  

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes) 

    # Load the model weights from the 'model' key in the checkpoint
    model.load_state_dict(checkpoint['model'], strict=False)  
    print(f"Loaded pre-trained weights from {weights_path}\n")
    return model

def load_class_idx_to_species(species_mapping_path):
    """
    Load the class-to-species mapping from a JSON file.
    """
    with open(species_mapping_path, 'r') as f:
        class_idx_to_species = json.load(f)
    return class_idx_to_species

def main():
    model_name = 'vit_base_patch16_224'

    device = torch.device('cpu')

    # Define paths to weights and class map, adjusting for the new directory structure
    weights_path = os.path.join(UTILITIES_DIR, 'vit_base_patch16_224_weights_best_acc.tar')
    class_map_path = os.path.join(UTILITIES_DIR, 'class_idx_to_species_id.json')

    # Load the pre-trained Vision Transformer model
    print(f"Loading model: {model_name} on device {device}")
    model = timm.create_model(model_name, pretrained=False) 
    model = model.to(device)
    model.eval()  

    # Load the class index to species mapping
    class_idx_to_species = load_class_idx_to_species(class_map_path)
    num_classes = len(class_idx_to_species)  

    # Load pre-trained weights and modify the output layer to match the number of classes
    model = load_pretrained_weights(model, weights_path, num_classes)

    # Parse command-line arguments 
    parser = argparse.ArgumentParser(description='Run Vision Transformer Model with Fine-Tuning')
    parser.add_argument('image', type=str, help='Image filename (e.g. 1.jpg)')
    args = parser.parse_args()

    # Load and preprocess the image
    image = load_image(args.image)
    
    # Make prediction
    outputs = model(image)

    # Get the predicted class index
    predicted_class = torch.argmax(outputs, dim=1)
    print(f"Predicted class index: {predicted_class.item()}")

    # Map the predicted class index to the species ID
    predicted_species_id = class_idx_to_species.get(str(predicted_class.item()), "Unknown")
    print(f"Predicted species ID: {predicted_species_id}")

    # Map the species ID to the species name
    species_names_path = os.path.join(UTILITIES_DIR, 'plantnet300K_species_names.json')
    with open(species_names_path, 'r') as f:
        species_names = json.load(f)

    predicted_species_name = species_names.get(predicted_species_id, "Unknown species")
    print(f"Predicted species name: {predicted_species_name}")

if __name__ == "__main__":
    main()
