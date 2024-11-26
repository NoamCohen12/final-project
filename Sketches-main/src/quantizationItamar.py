import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sympy import true
from torchvision.models import resnet18, ResNet18_Weights

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'/Sketches-main/src')
import Quantizer

# נתיב לקבצים הדרושים
ASSETS_PATH = r"/"
LABELS_FILE = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\imagenet_class_index.json"

# טעינת labels
with open(LABELS_FILE, 'r') as f:
    labels = json.load(f)


def print_weight_dtypes(model):
    """
    מדפיס את השמות ואת סוגי הנתונים של משקולות (weights) בלבד במודל.
    """
    print("===== Weight Dtypes =====")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name}: {param.dtype}")


def verify_quantization(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
        else:
            print(f"No NaN in {name}")

def predict_image(model, image_path):
    """
    פונקציה שמבצעת ניבוי עבור תמונה על בסיס מודל נתון
    """
    model = model.float()  # ודא שהמודל בפורמט float32

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).float()

    if torch.isnan(input_tensor).any():
        print("Error: Input tensor contains NaN values")
        return None, None

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    if torch.isnan(output).any():
        print("Error: Output contains NaN values")
        return None, None

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    if torch.isnan(probabilities).any():
        print("Error: Probabilities contain NaN values")
        return None, None

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


# def quantize_model(model, use_sign=True):
#     """
#     Apply custom quantization to all weights in the model.
#     """
#     # יצירת גריד
#     cntr_size = 8
#     if use_sign:
#         grid = np.array([item for item in range(2 ** cntr_size)])
#     else:
#         grid = np.array([item for item in range(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1))])
#
#     # קוונטיזציה למשקלים
#
#     return model

def quantize_on_layer(tensor, use_sign=True):
    """
    מבצע קוונטיזציה מותאמת אישית עבור שכבה נתונה
    """
    print("Original tensor min:", tensor.min().item(), "max:", tensor.max().item())

    if tensor is None or torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values or is None")
        return tensor

    cntr_size = 8
    grid = (np.arange(2 ** cntr_size) if use_sign else
            np.arange(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1)))

    tensor_flat = tensor.cpu().numpy().flatten()

    try:
        quantized_vec, scale, extra_value = Quantizer.quantize(vec=tensor_flat, grid=grid)
        dequantized_vec = Quantizer.dequantize(quantized_vec, scale, z=extra_value)

        # חישוב סטיית השגיאה לאחר דה-קוונטיזציה
        mse_error = np.mean((tensor_flat - dequantized_vec) ** 2)
        print(f"Quantization MSE error: {mse_error:.6f}")

    except Exception as e:
        print(f"Quantization error: {e}")
        return tensor

    quantized_tensor = torch.from_numpy(dequantized_vec).view(tensor.shape).to(tensor.device, dtype=torch.float32)
    print("Quantized tensor min:", quantized_tensor.min().item(), "max:", quantized_tensor.max().item())

    if torch.isnan(quantized_tensor).any():
        print("Error: NaN values detected in quantized tensor")
        return tensor  # מחזירים את הטנסור המקורי במקרה של תקלה

    return quantized_tensor


# פונקציה שמבצעת קוונטיזציה על כל השכבות במודל
def quantize_all_layers(model):
    """
    מבצע קוונטיזציה על כל השכבות במודל
    """
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            layer.weight.data = quantize_on_layer(layer.weight.data)

            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:
                    layer.bias.data = layer.bias.data.float()
                layer.bias.data = quantize_on_layer(layer.bias.data)

    return model


# טוענים את המודל ומבצעים קוונטיזציה
model = resnet18(weights=ResNet18_Weights.DEFAULT)
print("original model")
print_weight_dtypes(model)  # 32
# # הדפסת המשקלים של השכבה הראשונה לפני הכימות
# print("Weights of the first Conv2d layer before quantization:")
# print(model.conv1.weight.data)

model = quantize_all_layers(model)
print("model after quantization")
print_weight_dtypes(model)  # 32

# for name, param in model.named_parameters():
#     print(name, param.dtype)

# verify_quantization(model)


# הגדרת נתיב לתמונה שאתה רוצה לנבא עליה
image_path_dog = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\dog.jpg"  # שם התמונה שלך
image_path_cat = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\cat.jpeg"  # שם התמונה שלך
image_path_kite = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\kite.jpg"  # שם התמונה שלך
image_path_lion = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\lion.jpeg"  # שם התמונה שלך
image_path_paper = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\paper.jpg"  # שם התמונה שלך
array_of_path = [image_path_dog]
for(image_path) in array_of_path:
    # מבצע את הניבוי על התמונה
    class_index, probabilities = predict_image(model, image_path)
    predicted_label = labels[str(class_index)][1]
    predicted_prob = probabilities[class_index].item() * 100
    print(f"Prediction: {predicted_label} ({predicted_prob:.2f}%)")
# # הדפסת המשקלים של השכבה הראשונה אחרי הכימות
# print("\nWeights of the first Conv2d layer after quantization:")
# print(model.conv1.weight.data)

# # משקלים של השכבה הראשונה (conv1)
# conv1_weights = model.conv1.weight
#
# # הפיכת המשקלים לווקטור חד-ממדי
# flattened_weights = conv1_weights.view(-1)  # או conv1_weights.flatten()

# # הדפסת המשקלים של השכבה הראשונה לפני הכימות
# print("Conv1 Weights:")
# np.set_printoptions(threshold=np.inf)  # הדפסת כל הערכים של המערך ללא קיצוץ

# print(np.sort(flattened_weights.detach().numpy()))  # הדפס אחרי המרה ל-NumPy

# # קריאה לפונקציה לביצוע כימות
# quantize_on_layer(model, use_sign=True, tensor=flattened_weights)

# model = quantize_model(model)


# print_weights_after_quantization(model)


# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# # ניבוי על התמונה עם המודל המקוונטז
# image_path = os.path.join(ASSETS_PATH, "dog.jpg")
# class_index, probabilities = predict_image(model, image_path)
#
# # הדפסת תחזית עם אחוזים
# predicted_label = labels[str(class_index)][1]
# predicted_prob = probabilities[class_index].item() * 100
# print(f"Prediction: {predicted_label} ({predicted_prob:.2f}%)")

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


import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'/Sketches-main/src')
import Quantizer

# נתיב בסיס
BASE_PATH = r"C:\Users\97253\OneDrive\שולחן העבודה\final project"
LABELS_FILE = os.path.join(BASE_PATH, "imagenet_class_index.json")

# טעינת labels
with open(LABELS_FILE, 'r') as f:
    labels = json.load(f)


def print_weight_dtypes(model):
    """
    מדפיס את השמות ואת סוגי הנתונים של משקולות (weights) בלבד במודל.
    """
    print("===== Weight Dtypes =====")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name}: {param.dtype}")


def verify_quantization(model):
    """
    פונקציה שבודקת אם יש ערכי NaN במשקולות המודל לאחר קוונטיזציה.
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
        else:
            print(f"No NaN in {name}")


def predict_image(model, image_path):
    """
    פונקציה שמבצעת ניבוי עבור תמונה על בסיס מודל נתון
    """
    model = model.float()  # ודא שהמודל בפורמט float32

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).float()

    if torch.isnan(input_tensor).any():
        print("Error: Input tensor contains NaN values")
        return None, None

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    if torch.isnan(output).any():
        print("Error: Output contains NaN values")
        return None, None

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    if torch.isnan(probabilities).any():
        print("Error: Probabilities contain NaN values")
        return None, None

    class_index = torch.argmax(probabilities).item()
    return class_index, probabilities


def quantize_on_layer(tensor, use_sign=True):
    """
    מבצע קוונטיזציה מותאמת אישית עבור שכבה נתונה.
    """
    print("Original tensor min:", tensor.min().item(), "max:", tensor.max().item())

    if tensor is None or torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values or is None")
        return tensor

    cntr_size = 8
    grid = (np.arange(2 ** cntr_size) if use_sign else
            np.arange(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1)))

    tensor_flat = tensor.cpu().numpy().flatten()

    try:
        quantized_vec, scale, extra_value = Quantizer.quantize(vec=tensor_flat, grid=grid)
        dequantized_vec = Quantizer.dequantize(quantized_vec, scale, z=extra_value)

        # חישוב סטיית השגיאה לאחר דה-קוונטיזציה
        mse_error = np.mean((tensor_flat - dequantized_vec) ** 2)
        print(f"Quantization MSE error: {mse_error:.6f}")

    except Exception as e:
        print(f"Quantization error: {e}")
        return tensor

    quantized_tensor = torch.from_numpy(dequantized_vec).view(tensor.shape).to(tensor.device, dtype=torch.float32)
    print("Quantized tensor min:", quantized_tensor.min().item(), "max:", quantized_tensor.max().item())

    if torch.isnan(quantized_tensor).any():
        print("Error: NaN values detected in quantized tensor")
        return tensor  # מחזירים את הטנסור המקורי במקרה של תקלה

    return quantized_tensor


def quantize_all_layers(model):
    """
    מבצע קוונטיזציה על כל השכבות במודל.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            layer.weight.data = quantize_on_layer(layer.weight.data)

            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:
                    layer.bias.data = layer.bias.data.float()
                layer.bias.data = quantize_on_layer(layer.bias.data)

    return model


# טוענים את המודל ומבצעים קוונטיזציה
model = resnet18(weights=ResNet18_Weights.DEFAULT)
print("Original model weights:")
print_weight_dtypes(model)

model = quantize_all_layers(model)
print("Model after quantization:")
print_weight_dtypes(model)

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
    class_index, probabilities = predict_image(model, image_path)
    if class_index is not None:
        predicted_label = labels[str(class_index)][1]
        predicted_prob = probabilities[class_index].item() * 100
        # Print the prediction
        print(f"Prediction for {os.path.basename(image_path)}: {predicted_label} ({predicted_prob:.2f}%)")

        # Append the results to the list
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

# Save the results to a JSON file
results_file = "predictions_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_file}")

