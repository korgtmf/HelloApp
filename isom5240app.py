from transformers import st
from PIL import Image
import os

# Title
print("Title: Age Classification using ViT")

# Load the age classification pipeline (this may take a moment the first time)
print("Loading the ViT age classifier model...")
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")

# Image path - make sure the file exists
image_name = "middleagedMan.jpg"

if not os.path.exists(image_name):
    print(f"Error: Image '{image_name}' not found in the current directory!")
    print("Current directory:", os.getcwd())
    exit()

# Open and convert image to RGB (important for consistency)
try:
    image = Image.open(image_name).convert("RGB")
    print(f"Successfully loaded image: {image_name} ({image.size[0]}x{image.size[1]})")
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# Classify age
print("Classifying age...")
age_predictions = age_classifier(image)

# Sort predictions by score (descending)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

# Display all predictions
print("\n" + "="*50)
print("PREDICTED AGE RANGES (sorted by confidence):")
print("="*50)
for i, pred in enumerate(age_predictions):
    print(f"{i+1}. {pred['label']:15} | Confidence: {pred['score']:.4f}")

# Show the top prediction
top_prediction = age_predictions[0]
print("\n" + "-"*50)
print(f"🎯 MOST LIKELY AGE RANGE: {top_prediction['label']}")
print(f"   Confidence: {top_prediction['score']:.2%}")
print("-"*50)
