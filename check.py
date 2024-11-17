import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'C:\Users\97253\OneDrive\שולחן העבודה\final project\Sketches-main\src')
import Quantizer

# נתיב לקבצים הדרושים
ASSETS_PATH = r"C:\Users\97253\OneDrive\שולחן העבודה\final project"
LABELS_FILE = os.path.join(ASSETS_PATH, "imagenet_class_index.json")

# טעינת labels
with open(LABELS_FILE, 'r') as f:
    labels = json.load(f)


def predict_image(model, image_path):
    """
    Predict the class of an image using the given model.
    """
    # הגדרת טרנספורמציות
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # טעינת התמונה ויישום טרנספורמציות
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # הוספת ממד Batch

    # ביצוע ניבוי
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_index = torch.argmax(probabilities).item()

    return class_index, probabilities


def print_weights_after_quantization(model):
    """
    Print the weights of layer 0 after quantization in a flat array format.
    """
    layer_0_weights = model.layer1[0].conv1.weight.data.numpy()
    flattened_weights = layer_0_weights.flatten()
    print("Values after quantization for layer 0:\n", np.sort(flattened_weights))


def custom_quantize(weight_vector, grid):
    """
    Apply custom quantization to a weight vector using the specified grid.
    """
    quantized_weight, scale, extra_value = Quantizer.quantize(weight_vector, grid)
    return quantized_weight


def quantize_model(model, use_sign=True):
    """
    Apply custom quantization to all weights in the model.
    """
    # יצירת גריד
    cntr_size = 8
    if use_sign:
        grid = np.array([item for item in range(2 ** cntr_size)])
    else:
        grid = np.array([item for item in range(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1))])

    # קוונטיזציה למשקלים
    for name, param in model.named_parameters():
        if 'weight' in name:
            quantized_weight = custom_quantize(param.data.numpy().flatten(), grid)
            param.data = torch.tensor(quantized_weight, dtype=torch.float32).reshape(param.data.shape)
    return model


# טוענים את המודל ומבצעים קוונטיזציה
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = quantize_model(model)
print_weights_after_quantization(model)


#TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# ניבוי על התמונה עם המודל המקוונטז
image_path = os.path.join(ASSETS_PATH, "dog.jpg")
class_index, probabilities = predict_image(model, image_path)

# הדפסת תחזית עם אחוזים
predicted_label = labels[str(class_index)][1]
predicted_prob = probabilities[class_index].item() * 100
print(f"Prediction: {predicted_label} ({predicted_prob:.2f}%)")

# def quantizeModel(model, useSign=True):
#     """
#     Quantize the model using my quantization function
#     """
#     # קבלת וקטור המשקלים של שכבת הקונבולוציה הראשונה
#     conv1_weights = model.layer1[0].conv1.weight.data.numpy()  # גישה למשקלים של קונבולוציה הראשונה
#     flattened_weights = conv1_weights.flatten()  # שטח את המטריצה לווקטור חד-ממדי
#     print("Values before quantization:\n", np.sort(flattened_weights))  # הדפס לפני כימות
#
#     # יצירת גריד לקוונטיזציה
#     cntrSize = 8
#     if useSign:
#         grid = np.array([item for item in range(2 ** cntrSize)])
#     else:
#         grid = np.array([item for item in range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1))])
#
#     # תהליך הכימות
#     quantized_weights, scale, extra_value = Quantizer.quantize(flattened_weights, grid)
#     print("Values after quantization:\n", np.sort(quantized_weights))  # הדפס אחרי כימות
#
#     # תהליך דה-כימות
#     dequantized_weights = Quantizer.dequantize(quantized_weights, scale, z=1)
#     print("Values after dequantization:\n", np.sort(dequantized_weights))  # הדפס אחרי דה-כימות
#
#     # החלת המשקלים המקוונטזים על המודל
#     model.layer1[0].conv1.weight.data = torch.tensor(quantized_weights, dtype=torch.float32).reshape(
#         conv1_weights.shape)
#
#     return model
#
#
# # קריאה לפונקציה עם המודל
# model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# quantizeModel(model)