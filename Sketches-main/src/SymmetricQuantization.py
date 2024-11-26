import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# נתיב בסיס
BASE_PATH = r"C:\Users\97253\OneDrive\שולחן העבודה\final project"
LABELS_FILE = os.path.join(BASE_PATH, "imagenet_class_index.json")

# טעינת labels
with open(LABELS_FILE, 'r') as f:
    labels = json.load(f)


def predict_image(model, image_path):
    """
    פונקציה שמבצעת ניבוי עבור תמונה על בסיס מודל נתון
    """
    try:
        model = model.float()  # ודא שהמודל בפורמט float32

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_index = torch.argmax(probabilities).item()
        return class_index, probabilities

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def quantize_weights_symmetric(weights, bits=8):
    """
    ביצוע קוונטיזציה סימטרית של משקולות
    """
    # מציאת טווח המשקולות
    min_val = weights.min()
    max_val = weights.max()

    # חישוב הטווח המקסימלי
    max_range = max(abs(min_val), abs(max_val))

    # יצירת גריד קוונטיזציה
    levels = 2 ** (bits - 1)
    scale = max_range / levels

    # קוונטיזציה
    quantized = torch.round(weights / scale) * scale

    return quantized


def quantize_model_symmetric(model):
    """
    קוונטיזציה סימטרית של כל המשקולות במודל
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if module.weight is not None:
                module.weight.data = quantize_weights_symmetric(module.weight.data)

            if module.bias is not None:
                module.bias.data = quantize_weights_symmetric(module.bias.data)

    return model


# טוענים את המודל
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# ביצוע קוונטיזציה סימטרית
model = quantize_model_symmetric(model)

# הגדרת נתיבים לתמונות
image_paths = [
    os.path.join(BASE_PATH, "dog.jpg"),
    os.path.join(BASE_PATH, "cat.jpeg"),
    os.path.join(BASE_PATH, "kite.jpg"),
    os.path.join(BASE_PATH, "lion.jpeg"),
    os.path.join(BASE_PATH, "paper.jpg"),
]

# Initialize a list to store results
results = []

# Iterate through the image paths and perform predictions
for image_path in image_paths:
    try:
        class_index, probabilities = predict_image(model, image_path)
        if class_index is not None:
            predicted_label = labels[str(class_index)][1]
            predicted_prob = probabilities[class_index].item() * 100
            print(f"Prediction for {os.path.basename(image_path)}: {predicted_label} ({predicted_prob:.2f}%)")

            results.append({
                "image": os.path.basename(image_path),
                "predicted_label": predicted_label,
                "confidence": predicted_prob
            })
        else:
            print(f"Prediction failed for {os.path.basename(image_path)}.")
            results.append({
                "image": os.path.basename(image_path),
                "predicted_label": "None",
                "confidence": 0
            })
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Save the results to a JSON file
results_file = "predictions_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_file}")