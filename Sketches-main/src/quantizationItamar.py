import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from settings import *
import Quantizer

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'/Sketches-main/src')

# נתיב לקבצים הדרושים
ASSETS_PATH = r"/"


def getImageList(folderPath) -> list:
    imageList = []
    for filename in os.listdir(folderPath):
        # if it is a folder - open it and loop over the images
        subfolderPath = os.path.join(folderPath, filename)
        if os.path.isdir(subfolderPath):
            for image_path in os.listdir(subfolderPath):
                imageList.append(os.path.join(subfolderPath, image_path))
    return imageList


def check_precision(prediction_list, prediction_list_quant) -> float:
    """
    check the precision of the quantization
    """
    # חישוב דיוק הניבוי
    correct = 0
    for label in prediction_list_quant:
        if label in prediction_list:
            correct += 1
    precision = correct / len(prediction_list_quant)
    return precision

def check_recall(prediction_list, prediction_list_quant) -> float:
    """
    Calculate the recall of the quantized predictions.

    Args:
        prediction_list (list): The ground truth labels (original model predictions).
        prediction_list_quant (list): The predicted labels from the quantized model.

    Returns:
        float: The recall value.
    """
    # Count true positives (labels in the ground truth that were correctly predicted)
    true_positives = sum(1 for label in prediction_list if label in prediction_list_quant)

    # Calculate recall
    recall = true_positives / len(prediction_list) if prediction_list else 0.0
    return recall

def get_precision_of_all_quotations(list_of_images, device="cpu") -> None:
    """
    Predict the labels of a list of images using ResNet18 and compare
    the precision of various quantization methods.

    Args:
        list_of_images (list): List of image file paths.
        device (str): Device to use ("cpu" or "cuda").

    Returns:
        None
    """
    if not list_of_images:
        raise ValueError("The list of images is empty.")

    # Load the original ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Generate predictions with the original model
    prediction_list = []
    for image_path in list_of_images:
        label, class_index, probabilities = predict_image(model, image_path, device)
        if label:
            prediction_list.append(label)
    # print(prediction_list)
    # error(1)

    # Iterate over quantization methods and evaluate precision
    for method in ["INT8", "INT16", "F2P", "uniform_symmetric", "Morris"]:
        # Quantize the model
        quantized_model = quantize_model_switch(model, method=method)
        quantized_model = quantized_model.to(device)
        quantized_model.eval()  # Set quantized model to evaluation mode

        # Generate predictions with the quantized model
        prediction_list_quant = []
        for image_path in list_of_images:
            label, class_index, probabilities = predict_image(quantized_model, image_path, device)
            if label:
                prediction_list_quant.append(label)

        # Compare predictions and calculate precision
        precision = check_precision(prediction_list, prediction_list_quant)
        recall = check_recall(prediction_list, prediction_list_quant)
        print(f"Precision of {method}: {precision:.3f}")
        print(f"Recall of {method}: {recall:.3f}")


# create a generator taht loop over all the images in the folder
def imageGenerator(folderPath):
    for filename in os.listdir(folderPath):
        # if it is a folder - open it and loop over the images
        if os.path.isdir(folderPath + filename):
            for image_path in os.listdir(folderPath + filename):
                yield image_path


def quantize_model_switch(model, method="None"):
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
        "INT8": lambda m: quantize_all_layers(m, quantization_type="symmetric", cntr_size=8),
        "INT16": lambda m: quantize_all_layers(m, quantization_type="symmetric", cntr_size=16),
        "uniform_symmetric": lambda m: quantize_all_layers(m, quantization_type="symmetric", cntr_size=8),
        "F2P": lambda m: quantize_model_F2P(m),
        "Morris": lambda m: quantize_all_layers(m, quantization_type="morris", cntr_size=8),
    }

    # Apply the selected quantization method
    quantized_model = quantization_methods.get(method, lambda m: m)(model)
    print(f"Model quantized with {method}")
    return quantized_model


def uniform_asymmetric_quantize(tensor, num_bits=8) -> torch.Tensor:
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


def uniform_symmetric_quantize(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
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


# Dummy quantization functions for F2P and Morris (implement these as needed)
def quantize_model_F2P(model, cntr_size=8) -> torch.nn.Module:  # same resent18 model stil $$$
    print("Applying F2P quantization...")
    # Custom F2P quantization logic here
    return model


def quantize_model_morris(tensor, num_bits=8) -> torch.Tensor:
    """
    Quantize a tensor using Morris quantization.

    Args:
        tensor (torch.Tensor): The tensor to quantize.
        num_bits (int): The number of bits for quantization.

    Returns:
        torch.Tensor: The quantized tensor.
    """
    # Number of quantization levels
    levels = 2 ** num_bits

    # Calculate percentiles (buckets)
    percentiles = torch.linspace(0, 100, levels + 1)
    buckets = torch.quantile(tensor.flatten(), percentiles / 100.0)

    # Assign each value to the nearest bucket
    quantized = torch.bucketize(tensor, buckets, right=True) - 1
    quantized = torch.clamp(quantized, 0, levels - 1)

    # Map quantized values back to original range
    dequantized = buckets[quantized]

    return dequantized


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


def print_weight_dtypes(model) -> None:
    """
    מדפיס את השמות ואת סוגי הנתונים של משקולות (weights) בלבד במודל.
    """
    print("===== Weight Dtypes =====")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name}: {param.dtype}")


def verify_quantization(model) -> None:  # Checks that the weights are not NaN.
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
        else:
            print(f"No NaN in {name}")  #


def predict_image(model, image_path, device='cpu') -> tuple:
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
    with open(r"C:\Users\97253\OneDrive\שולחן העבודה\final project\imagenet_class_index.json") as labels_file:
        labels = json.load(labels_file)

    label = labels[str(class_index)]

    return label, class_index, probabilities


def print_all_5weights(model) -> None:  # Prints the first 5 weights of each layer in the model.
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


def quantize_on_layer(tensor, use_sign=True) -> torch.Tensor:  # of itamar
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


def quantize_all_layers(model, quantization_type: str = "itamar", cntr_size=8) -> torch.nn.Module:
    """
    מבצע קוונטיזציה על כל השכבות במודל
    """
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            # print(f"Quantizing weights for the first layer: {name}")

            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()
            match quantization_type:
                case "itamar":
                    layer.weight.data = quantize_on_layer(layer.weight.data)
                case "symmetric":
                    layer.weight.data = uniform_symmetric_quantize(layer.weight.data, cntr_size)
                case "asymmetric":
                    layer.weight.data = uniform_asymmetric_quantize(layer.weight.data, cntr_size)
                case "morris":
                    layer.weight.data = quantize_model_morris(layer.weight.data, cntr_size)
                case "F2P":
                    layer.weight.data = quantize_model_F2P(layer.weight.data, cntr_size)
                case _:
                    print("Error: Invalid quantization type")

            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:
                    layer.bias.data = layer.bias.data.float()
                    match quantization_type:
                        case "itamar":
                            layer.bias.data = quantize_on_layer(layer.bias.data)
                        case "symmetric":
                            layer.bias.data = uniform_symmetric_quantize(layer.bias.data)
                        case "asymmetric":
                            layer.bias.data = uniform_asymmetric_quantize(layer.bias.data)
                        case "morris":
                            layer.bias.data = quantize_model_morris(layer.bias.data)
                        case "F2P":
                            layer.bias.data = quantize_model_F2P(layer.bias.data)
                        case _:
                            print("Error: Invalid quantization type")

    return model


def main():
    # check if the path is correct
    images_path = r"..\..\100 animals"  # 100 animals
    if os.path.exists(images_path):
        print("Path exists")
    lst = getImageList(images_path)  # test the imageGenerator function
    get_precision_of_all_quotations(lst)
    error(1)

    # 'print5weights',
    #    'printFirstLayer5weights',
    #    'printFirstLayerWeightsUniform',
    #    'printAllLayersTypes',
    #    'printAllLayersWeightsUniform',
    #    '5ImagesTestQuantization',
    #    'checkNanValuesWeights'

    VERBOSE = ["5ImagesTestQuantization"]

    image_path_dog = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\dog.jpg"  # שם התמונה שלך
    image_path_cat = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\cat.jpeg"  # שם התמונה שלך
    image_path_kite = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\kite.jpg"  # שם התמונה שלך
    image_path_lion = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\lion.jpeg"  # שם התמונה שלך
    image_path_paper = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\paper.jpg"  # שם התמונה שלך
    array_of_path = [image_path_dog, image_path_cat, image_path_kite, image_path_lion, image_path_paper]

    # # ביצוע ניבוי על כל התמונות
    # for image_path in array_of_path:
    #     # מבצע את הניבוי על התמונה
    #     compare_quantization_methods(image_path)

    # טוענים את המודל ומבצעים קוונטיזציה
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = quantize_all_layers(model, "morris", 8)
    # quanitizied with INT8, INT16, F2P, Morris.
    # model = quantize_model(model)

    if 'print5weights' in VERBOSE:
        print("Original model")
        print_all_5weights(model)
        print("Quantized model")
        model = quantize_all_layers(model)
        print_all_5weights(model)
        error(1)

    if 'printFirstLayer5weights' in VERBOSE:  # print the first 5 weights of the first layer
        vec2quantize = model.conv1.weight.data.flatten()[:5]  # $$$
        print(f'b4 {vec2quantize}')  # $$$
        quantized_vec = quantize_on_layer(vec2quantize, 8)
        print(f'after {quantized_vec}')  # $$$
        error(1)

    if 'printFirstLayerWeightsUniform' in VERBOSE:  # print the all weights of the first layer
        vec2quantize = model.conv1.weight.data  # $$$
        print(f'b4 {vec2quantize}')  # $$$
        quantized_vec = uniform_symmetric_quantize(vec2quantize)  # $$$
        print(f'after {quantized_vec}')  # $$$
        error(1)

    if 'printAllLayersTypes' in VERBOSE:  # print the all types of the  layers
        print("original types model")
        print_weight_dtypes(model)  # 32
        print("new types model")
        model = quantize_all_layers(model)
        print_weight_dtypes(model)  # 32
        error(1)

    if 'checkNanValuesWeights' in VERBOSE:  # check if there are NaN values in the weights
        print("new types model")
        verify_quantization(model)  # check if there are NaN values in the weights
        error(2)

    if '5ImagesTestQuantization' in VERBOSE:  # test the quantization on 5 images
        # הגדרת המכשיר (GPU אם זמין, אחרת CPU)
        if torch.cuda.is_available():
            device = "cuda"
            print("GPU is available")
        else:
            print("GPU is not available, using CPU")
            device = "cpu"
        model = model.to(device)
        # הגדרת נתיב לתמונה שאתה רוצה לנבא עליה
        image_path_dog = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\dog.jpg"  # שם התמונה שלך
        image_path_cat = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\cat.jpeg"  # שם התמונה שלך
        image_path_kite = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\kite.jpg"  # שם התמונה שלך
        image_path_lion = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\lion.jpeg"  # שם התמונה שלך
        image_path_paper = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\paper.jpg"  # שם התמונה שלך
        array_of_path = [image_path_dog, image_path_cat, image_path_kite, image_path_lion, image_path_paper]

        # ביצוע ניבוי על כל התמונות
        for image_path in array_of_path:
            # מבצע את הניבוי על התמונה
            label, class_index, probabilities = predict_image(model, image_path, device)

            if label:
                predicted_label = label
                predicted_prob = probabilities[class_index].item() * 100
                print(f"Prediction: {predicted_label} (Class Index: {class_index}, Probability: {predicted_prob:.2f}%)")


if __name__ == '__main__':
    main()
