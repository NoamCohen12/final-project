import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from settings import *

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'/Sketches-main/src')

# נתיב לקבצים הדרושים
ASSETS_PATH = r"/"




def quantize_model_switch(model, method="None", num_bits=8):
    """
    Quantizes the given model based on the specified method.

    Args:
        model (torch.nn.Module): The model to be quantized.
        method (str): The quantization method. One of {"None", "INT8", "INT16", "F2P", "Morris"}.
        num_bits (int): The number of bits for quantization.

    Returns:
        torch.nn.Module: The quantized model.
    """
    quantization_methods = {
        "None": lambda m: m,  # Return the original model
        "INT8": lambda m: quantize_all_layers_uniform(m, num_bits=8),
        "INT16": lambda m: quantize_all_layers_uniform(m, num_bits=16),
        "F2P": lambda m: quantize_model_F2P(m),
        "Morris": lambda m: quantize_model_morris(m),
    }

    # Apply the selected quantization method
    quantized_model = quantization_methods.get(method, lambda m: m)(model)
    print(f"Model quantized with {method}")
    return quantized_model


# Dummy quantization functions for F2P and Morris (implement these as needed)
def quantize_model_F2P(model):
    print("Applying F2P quantization...")
    # Custom F2P quantization logic here
    return model


def quantize_model_morris(model):
    print("Applying Morris quantization...")
    # Custom Morris quantization logic here
    return model


# Testing the quantized models
def test_model_precision(model, image_path, device="cpu"):
    """
    Tests the precision of a given model on a single image.

    Args:
        model (torch.nn.Module): The model to be tested.
        image_path (str): Path to the image file.

    Returns:
        tuple: Predicted label, class index, and probability.
    """
    label, class_index, probabilities = predict_image(model, image_path, device)
    if label:
        predicted_label = label
        predicted_prob = probabilities[class_index].item() * 100
        print(f"Prediction: {predicted_label} (Class Index: {class_index}, Probability: {predicted_prob:.2f}%)")
    else:
        print("Prediction failed.")
    return predicted_label, predicted_prob


# Compare precision across quantization methods
def compare_quantization_methods(image_path, device="cpu"):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  # Load the original model

    quantization_methods = ["None", "INT8", "INT16", "F2P", "Morris"]
    results = {}

    for method in quantization_methods:
        quantized_model = quantize_model_switch(model, method=method)
        label, prob = test_model_precision(quantized_model, image_path, device)
        results[method] = (label, prob)

    print("\nComparison of Quantization Methods:")
    for method, result in results.items():
        print(f"Method: {method}, Label: {result[0]}, Probability: {result[1]:.2f}%")




def uniform_symmetric_quantize_with_checks(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """
    מבצע קוונטיזציה אחידה סימטרית לטנסור, כולל בדיקות איכות.

    Args:
        tensor (torch.Tensor): הטנסור שברצונך לקוונטז.
        num_bits (int): מספר הביטים לקוונטיזציה.

    Returns:
        torch.Tensor: הטנסור המקוונטז או הטנסור המקורי במקרה של בעיה.
    """
    # בדיקת קלט: אם הטנסור ריק, מכיל NaN או None
    if tensor is None or not torch.is_tensor(tensor):
        print("Error: Input is None or not a valid tensor.")
        return tensor

    if torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values.")
        return tensor

    try:
        # שלב 1: חישוב הטווח והסקלה
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (num_bits - 1) - 1)

        # שלב 2: קוונטיזציה וקלמפינג לטווח הביטים
        quantized = torch.clamp((tensor / scale).round(), -2 ** (num_bits - 1) + 1, 2 ** (num_bits - 1) - 1)

        # שלב 3: דה-קוונטיזציה (שחזור הערכים המקוריים)
        dequantized = quantized * scale

        # שלב 4: בדיקת פלט
        if torch.isnan(dequantized).any():
            print("Error: NaN values detected in quantized tensor.")
            return tensor

        return dequantized  # return the tensor after quantization

    except Exception as e:
        print(f"Error during quantization: {e}")
        return tensor  # return the tensor before quantization if there is an error


def quantize_all_layers_uniform(model, num_bits=8) -> torch.nn.Module:
    """
    מבצע קוונטיזציה אחידה סימטרית על כל השכבות במודל.

    Args:
        model (torch.nn.Module): המודל שברצונך לקוונטז.
        num_bits (int): מספר הביטים לקוונטיזציה.

    Returns:
        torch.nn.Module: המודל המקוונטז.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            # קוונטיזציה של המשקולות
            quantized_weights = uniform_symmetric_quantize_with_checks(layer.weight.data, num_bits=num_bits)
            layer.weight.data = quantized_weights
            print(f'{name}: {layer.weight.data.flatten()[:5]}')

            # קוונטיזציה של ה-bias אם קיים
            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:  # if the bias is in float64 then convert it to float32
                    layer.bias.data = layer.bias.data.float()
                quantized_bias = uniform_symmetric_quantize_with_checks(layer.bias.data, num_bits=num_bits)
                layer.bias.data = quantized_bias
                print(f'{name}.bias: {layer.bias.data.flatten()[:5]}')

    return model  # return the model after quantization


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


def predict_image(model, image_path, device='cpu'):
    """
    פונקציה שמבצעת ניבוי עבור תמונה על בסיס מודל נתון
    """
    model = model.to(device).float()  # ודא שהמודל על המכשיר הנכון ובפורמט float32

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device).float()  # ודא שהקלט על המכשיר הנכון

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

    # טעינת שמות הקטגוריות מתוך קובץ JSON
    with open(r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\imagenet_class_index.json") as labels_file:
        labels = json.load(labels_file)

    label = labels[str(class_index)]

    return label, class_index, probabilities


def print_weights_after_quantization(model):
    """
    Print the weights of layer 0 after quantization in a flat array format.
    """
    layer_0_weights = model.layer1[0].conv1.weight.data.numpy()
    flattened_weights = layer_0_weights.flatten()
    print("Values after quantization for layer 0:\n", np.sort(flattened_weights))


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
def print_all_5weights(model):
    """
      מבצע קוונטיזציה על כל השכבות במודל
      """
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            layer.weight.data = quantize_on_layer(layer.weight.data)
            print(f'{name}: {layer.weight.data.flatten()[:5]}')


def quantize_on_layer(tensor, use_sign=True) -> torch.Tensor:
    """
    מבצע קוונטיזציה מותאמת אישית עבור שכבה נתונה
    """
    # print("Original tensor min:", tensor.min().item(), "max:", tensor.max().item())

    if tensor is None or torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values or is None")
        return tensor

    cntr_size = 8
    grid = (np.arange(2 ** cntr_size) if use_sign else
            np.arange(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1)))

    tensor_flat = tensor.cpu().numpy().flatten()

    try:
        # קוונטיזציה
        quantized_vec, scale, extra_value = Quantizer.quantize(vec=tensor_flat, grid=grid)
        dequantized_vec = Quantizer.dequantize(quantized_vec, scale, z=extra_value)

        # חישוב סטיית השגיאה לאחר דה-קוונטיזציה
        mse_error = np.mean((tensor_flat - dequantized_vec) ** 2)
        print(f"Quantization MSE error: {mse_error:.6f}")

        # אם הסטייה גבוהה מדי, אפשר לבדוק את הסיבות
        if mse_error > 1e-4:
            print("Warning: High MSE error after quantization, might affect model accuracy.")

    except Exception as e:
        print(f"Quantization error: {e}")
        return tensor

    # הפיכת הטנזורים למערכת הנתונים של פייתון
    quantized_tensor = torch.from_numpy(dequantized_vec).view(tensor.shape).to(tensor.device, dtype=torch.float32)

    # אם יש NaN אחרי הקוונטיזציה, מחזירים את הטנזור המקורי
    if torch.isnan(quantized_tensor).any():
        print("Error: NaN values detected in quantized tensor")
        return tensor  # מחזירים את הטנסור המקורי במקרה של תקלה

    # החזרת הטנזור המקוונטז
    return quantized_tensor


def quantize_all_layers(model):
    """
    מבצע קוונטיזציה על כל השכבות במודל
    """
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for the first layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            layer.weight.data = quantize_on_layer(layer.weight.data)

            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:
                    layer.bias.data = layer.bias.data.float()
                layer.bias.data = quantize_on_layer(layer.bias.data)

    return model


def uniform_asymmetric_quantize(tensor, num_bits=8):
    """
    Quantize a tensor using uniform asymmetric quantization.

    Args:
        tensor (torch.Tensor): הטנסור שברצונך לקוונטז.
        num_bits (int): מספר הביטים לקוונטיזציה.

    Returns:
        torch.Tensor: הטנסור המקוונטז.
    """
    # חישוב הטווח המינימלי והמקסימלי
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (2 ** num_bits - 1)
    zero_point = (-min_val / scale).round()

    # קוונטיזציה
    quantized = torch.clamp(((tensor / scale).round() + zero_point), 0, 2 ** num_bits - 1)

    # דה-קוונטיזציה
    dequantized = (quantized - zero_point) * scale

    return dequantized


# verbose = -5  # -1, 0, 2, 3, 4, 5
#             {'print5weights',
#            'printFirstLayer5weights',
#            'printFirstLayer5weightsUniform',
#            'printAllLayersWeights',
#            'printAllLayersWeightsUniform',
#            '5ImagesTestQuantization',
#            'checkNanValuesWeights'}
VERBOSE = ''

image_path_dog = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\dog.jpg"  # שם התמונה שלך
image_path_cat = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\cat.jpeg"  # שם התמונה שלך
image_path_kite = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\kite.jpg"  # שם התמונה שלך
image_path_lion = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\lion.jpeg"  # שם התמונה שלך
image_path_paper = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\paper.jpg"  # שם התמונה שלך
array_of_path = [image_path_dog, image_path_cat, image_path_kite, image_path_lion, image_path_paper]

# ביצוע ניבוי על כל התמונות
for image_path in array_of_path:
    # מבצע את הניבוי על התמונה
    compare_quantization_methods(image_path)

# # Example usage
# image_path = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\dog.jpg"  # Specify your image path
# compare_quantization_methods(image_path)


# טוענים את המודל ומבצעים קוונטיזציה
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = quantize_all_layers(model)
# quanitizied with INT8, INT16, F2P, Morris.
# model = quantize_model(model)


if VERBOSE == 'print5weights':
    print("Original model")
    print_all_5weights(model)
    print("Quantized model")
    model = quantize_all_layers(model)
    print_all_5weights(model)
    error(1)

if VERBOSE == 'printFirstLayer5weights':  # print the first 5 weights of the first layer
    vec2quantize = model.conv1.weight.data.flatten()[:5]  # $$$
    print(f'b4 {vec2quantize}')  # $$$
    # quantized_vec = quantize_on_layer(vec2quantize)  # $$$
    quantized_vec = uniform_symmetric_quantize_with_checks(vec2quantize, 8)
    print(f'after {quantized_vec}')  # $$$
    error(1)

if VERBOSE == 'printFirstLayer5weightsUniform':  # print the all weights of the first layer
    vec2quantize = model.conv1.weight.data  # $$$
    print(f'b4 {vec2quantize}')  # $$$
    quantized_vec = quantize_on_layer(vec2quantize)  # $$$
    # quantized_vec = quantize_on_layer_uniform(vec2quantize, 8)
    print(f'after {quantized_vec}')  # $$$
    error(1)

if VERBOSE == 'printAllLayersWeights':  # print the all types of the  layers
    print("original types model")
    print_weight_dtypes(model)  # 32
    print("new types model")
    model = quantize_all_layers(model)
    print_weight_dtypes(model)  # 32
    error(1)

if VERBOSE == 'printAllLayersWeightsUniform':
    model = quantize_all_layers_uniform(model)

if VERBOSE == 'checkNanValuesWeights':
    print("new types model")
    verify_quantization(model)  # check if there are NaN values in the weights
    error(2)

if VERBOSE == '5ImagesTestQuantization':  # test the quantization on 5 images
    # הגדרת המכשיר (GPU אם זמין, אחרת CPU)
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")
        device = "cpu"
    model = model.to(device)
    # הגדרת נתיב לתמונה שאתה רוצה לנבא עליה
    image_path_dog = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\dog.jpg"  # שם התמונה שלך
    image_path_cat = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\cat.jpeg"  # שם התמונה שלך
    image_path_kite = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\kite.jpg"  # שם התמונה שלך
    image_path_lion = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\lion.jpeg"  # שם התמונה שלך
    image_path_paper = r"C:\Users\Bar Yechezkel\PycharmProjects\CS_final_project\paper.jpg"  # שם התמונה שלך
    array_of_path = [image_path_dog, image_path_cat, image_path_kite, image_path_lion, image_path_paper]

    # ביצוע ניבוי על כל התמונות
    for image_path in array_of_path:
        # מבצע את הניבוי על התמונה
        label, class_index, probabilities = predict_image(model, image_path, device)

        if label:
            predicted_label = label
            predicted_prob = probabilities[class_index].item() * 100
            print(f"Prediction: {predicted_label} (Class Index: {class_index}, Probability: {predicted_prob:.2f}%)")

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
