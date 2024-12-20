import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import sys
from datetime import datetime
from settings import *
import Quantizer

# הוספת נתיב לקובץ Quantizer
sys.path.append(r'/Sketches-main/src')

# נתיב לקבצים הדרושים
ASSETS_PATH = r"/"


def random_tensor(tensor) -> torch.Tensor:
    """
    מבצע רנדומיזציה מלאה לטנזור נתון, כולל שלב דמוי 'dequantize'
    """
    if tensor is None or torch.isnan(tensor).any():
        print("Warning: Input tensor contains NaN values or is None")
        return tensor

    # הפיכת הטנזור למערך נומרי
    tensor_flat = tensor.cpu().numpy().flatten()

    try:
        # יצירת "קוונטיזציה" רנדומלית
        random_quantized = np.random.uniform(low=tensor_flat.min(), high=tensor_flat.max(), size=tensor_flat.shape)

        # "דה-קוונטיזציה" רנדומלית (מוסיפים סטייה רנדומלית לערכים הקיימים)
        randomized_dequantized = random_quantized + np.random.normal(loc=0, scale=0.1, size=random_quantized.shape)

        # חישוב סטיית שגיאה (רק לצורך הדגמה)
        mse_error = np.mean((tensor_flat - randomized_dequantized) ** 2)
        print(f"Randomization MSE error: {mse_error:.6f}")

    except Exception as e:
        print(f"Randomization error: {e}")
        return tensor

    # המרת הערכים האקראיים חזרה לטנזור של פייתון
    randomized_tensor = torch.from_numpy(randomized_dequantized).view(tensor.shape).to(tensor.device,
                                                                                       dtype=torch.float32)

    # בדיקה אם נוצרו NaN
    if torch.isnan(randomized_tensor).any():
        print("Error: NaN values detected in randomized tensor")
        return tensor  # מחזירים את הטנסור המקורי במקרה של תקלה

    # החזרת הטנזור המקוונטז (הרנדומלי)
    return randomized_tensor


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


import os


def write_results_to_file(file_path, method, precision, recall):
    """
    Write the precision and recall results of a quantization method to a file.

    Args:
        file_path (str): Path to the file where results will be saved.
        method (str): The quantization method used.
        precision (float): The precision of the method.
        recall (float): The recall of the method.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write results to the file
        with open(file_path, "a") as file:
            file.write("--------------------------------------------\n")
            file.write(f"Method: {method}\n")
            file.write(f"Precision: {precision:.3f}\n")
            file.write(f"Recall: {recall:.3f}\n")
            file.write("\n")
        print(f"Results for method {method} written to {file_path} successfully.")
    except Exception as e:
        print(f"Failed to write to file {file_path}: {e}")


def get_precision_of_all_quotations(list_of_images, device="cpu", test=True) -> None:
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
    if test:
        print("org:")
        printing_5(model)
        test_5pic(model)

    # Generate predictions with the original model
    prediction_list = []
    for image_path in list_of_images:
        label, class_index, probabilities = predict_image(model, image_path, device)
        if label:
            prediction_list.append(label)

    # Iterate over quantization methods and evaluate precision
    for method in ["INT8", "INT16", "F2P"]:
        try:
            # Quantize the model
            quantized_model = quantize_model_switch(model, method=method)
            if test:
                print("method:", method)
                printing_5(quantized_model)
                test_5pic(quantized_model)
            quantized_model = quantized_model.to(device)
            quantized_model.eval()  # Set quantized model to evaluation mode

            # Generate predictions with the quantized model
            prediction_list_quant = []
            for image_path in list_of_images:
                label, class_index, probabilities = predict_image(quantized_model, image_path, device)
                # print (label,"label111111111111111111111")
                if label:
                    prediction_list_quant.append(label)

            # Compare predictions and calculate precision
            precision = check_precision(prediction_list, prediction_list_quant)
            recall = check_recall(prediction_list, prediction_list_quant)
            print(f"Precision of {method}: {precision:.3f}")
            print(f"Recall of {method}: {recall:.3f}")

            # Save results to file
            path = "./report.txt"
            write_results_to_file(path, method, precision, recall)
        except Exception as e:
            print(f"Failed to process method {method}: {e}")


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
        "INT8": lambda m: quantize_all_layers(m, quantization_type="INT8", cntrSize=8),
        "INT16": lambda m: quantize_all_layers(m, quantization_type="INT16", cntrSize=16),
        "F2P": lambda m: quantize_all_layers(m, quantization_type="F2P", cntrSize=8),

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


import torch


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


def printing_5(model):
    print(model.conv1.weight.data.flatten()[:5])


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


# Quantizes the given model based on the specified method.
#
# Args:
# quantization type (str): (a)symmetric

def quantize_all_layers(
        model,
        quantization_type: str = "random",
        signed: bool = True,
        cntrSize=8,
        round1: bool = True) -> torch.nn.Module:
    """
    Perform quantization on all layers of the model.
    """
    print("quant_type:", quantization_type)
    for i, (name, layer) in enumerate(model.named_modules()):

        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            # Ensure weights are of type float32
            if layer.weight.dtype == torch.float64:
                layer.weight.data = layer.weight.data.float()

            # Quantization logic for weights
            match quantization_type:
                case "random":
                    layer.weight.data = random_tensor(layer.weight.data)
                case "INT8" | "INT16":
                    # Define the grid for quantization based on INT8 or INT16
                    print("cntrSize:", cntrSize)

                    grid = np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1)) if signed else np.array(
                        range(2 ** cntrSize))
                    print("grid-int8 or int16:", grid)
                    # Flatten the weight tensor for quantization
                    # מציאת המשקל המינימלי והמקסימלי לפני השטחה
                    print("before flat weight int8 or int16:", layer.weight.data)

                    # מציאת המשקל המינימלי והמקסימלי של המשקל בצורה רגילה
                    min_weight = layer.weight.data.min()
                    max_weight = layer.weight.data.max()

                    print(f"Min weight: {min_weight.item()}")
                    print(f"Max weight: {max_weight.item()}")

                    # שטיחת המשקל
                    flattened_weight = layer.weight.data.flatten()

                    # מציאת המשקל המינימלי והמקסימלי של הוקטור השטוח
                    minf_weight = flattened_weight.min()
                    maxf_weight = flattened_weight.max()

                    print(f"Min flattened weight: {minf_weight.item()}")
                    print(f"Max flattened weight: {maxf_weight.item()}")

                    quantized_weight, scale, z = Quantizer.quantize(flattened_weight, grid=grid)
                    print("quantized_weight int8 or int16:", quantized_weight)
                    print("scale int8 or int16:", scale)
                    print("z int8 or int16:", z)
                    # Dequantize the weight
                    quantized_weight = Quantizer.dequantize(quantized_weight, scale, z=z)
                    print("quantized_weight int8 or int16:", quantized_weight)

                    # מציאת המשקל המינימלי והמקסימלי של הוקטור השטוח
                    minD_weight = quantized_weight.min()
                    maxD_weight = quantized_weight.max()

                    print(f"MinD flattened weight: {minD_weight.item()}")
                    print(f"MaxD flattened weight: {maxD_weight.item()}")

                    # Reshape back to the original shape
                    layer.weight.data = torch.from_numpy(quantized_weight).view(layer.weight.data.shape).to(
                        layer.weight.data.device)
                    print("layer.weight.data int8 or int1611111:", layer.weight.data.flatten()[:5])

                case "F2P":

                    grid = Quantizer.getAllValsFxp(
                        fxpSettingStr='F2P_lr_h2',
                        cntrSize=cntrSize,
                        verbose=[],
                        signed=signed
                    )

                    print("grid-f2p:", grid)
                    # Flatten and quantize the weights
                    flat_weight = layer.weight.data.view(-1).numpy()
                    print("flat_weight f2p:", flat_weight)
                    quantized_weight, scale, z = Quantizer.quantize(flat_weight, grid=grid)
                    print("quantized_weight f2p:", quantized_weight)
                    print("scale f2p:", scale)
                    print("z f2p:", z)
                    quantized_weight = Quantizer.dequantize(quantized_weight, scale, z=z)
                    print("quantized_weight f2p:", quantized_weight)
                    layer.weight.data = torch.from_numpy(quantized_weight).view(layer.weight.data.shape).to(
                        layer.weight.data.device)
                    print("layer.weight.data f2p:", layer.weight.data.flatten()[:5])
                case _:
                    print(f"Error: Invalid quantization type '{quantization_type}' for weights")

            # Quantization logic for biases
            if layer.bias is not None:
                if layer.bias.dtype == torch.float64:
                    layer.bias.data = layer.bias.data.float()
                match quantization_type:
                    case "random":
                        layer.bias.data = random_tensor(layer.bias.data)
                    case "INT8" | "INT16":
                        # Define the grid for quantization based on INT8 or INT16
                        grid = np.array(
                            range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1)) if signed else np.array(
                            range(2 ** cntrSize))
                        print("grid_bias-int8 or int16:", grid)
                        # Flatten the bias tensor for quantization
                        flat_bias = layer.bias.data.flatten()
                        print("flat_bias int8 or int16:", flat_bias)
                        quantized_bias, scale, z = Quantizer.quantize(flat_bias, grid=grid)
                        print("quantized_bias int8 or int16:", quantized_bias)
                        # Dequantize the bias
                        quantized_bias = Quantizer.dequantize(quantized_bias, scale, z=z)
                        print("quantized_bias int8 or int16:", quantized_bias)
                        # Reshape back to the original shape
                        layer.bias.data = torch.from_numpy(quantized_bias).view(layer.bias.data.shape).to(
                            layer.bias.data.device)
                        print("layer.bias.data int8 or int16:", layer.bias.data.flatten()[:5])

                    case "F2P":
                        grid = Quantizer.getAllValsFxp(
                            fxpSettingStr=f"F2P_lr_h2",
                            cntrSize=cntrSize,
                            verbose=[],
                            signed=signed
                        )
                        print("grid_bias-f2p:", grid)
                        # Flatten and quantize the bias
                        flat_bias = layer.bias.data.view(-1).numpy()
                        print("flat_bias f2p:", flat_bias)
                        quantized_bias, scale, extra_value = Quantizer.quantize(flat_bias, grid=grid)
                        print("quantized_bias f2p:", quantized_bias)
                        quantized_bias = Quantizer.dequantize(quantized_bias, scale, z=extra_value)
                        print("quantized_bias f2p:", quantized_bias)
                        layer.bias.data = torch.from_numpy(quantized_bias).view(layer.bias.data.shape).to(
                            layer.bias.data.device)
                        print("layer.bias.data f2p:", layer.bias.data.flatten()[:5])
                    case _:
                        print(f"Error: Invalid quantization type '{quantization_type}' for bias")

    return model


# def quantize_tensor(tensor, quantization_type, signed, cntrSize, grid_settings=None):
#     """
#     Quantize and dequantize a tensor based on the given quantization type and parameters.
#     """
#     # Ensure the tensor is float32
#     if tensor.dtype == torch.float64:
#         tensor = tensor.float()
#
#     # Define the grid based on quantization type
#     if quantization_type in ["INT8", "INT16"]:
#         grid = np.arange(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1) if signed else np.arange(2 ** cntrSize)
#     elif quantization_type == "F2P":
#         grid = Quantizer.getAllValsFxp(fxpSettingStr=grid_settings, cntrSize=cntrSize, verbose=[], signed=signed)
#     elif quantization_type == "random":
#         tensor = random_tensor(tensor)
#         return tensor  # Return early for random quantization
#     else:
#         raise ValueError(f"Unsupported quantization type: {quantization_type}")
#
#     # Flatten, quantize, and dequantize
#     flat_tensor = tensor.view(-1).cpu().numpy()
#     quantized, scale, z = Quantizer.quantize(flat_tensor, grid=grid)
#     dequantized = Quantizer.dequantize(quantized, scale, z=z)

# # Reshape back to original shape
# return torch.from_numpy(dequantized).view(tensor.shape).to(tensor.device)


# def quantize_all_layers(model, quantization_type="random", signed=True, cntrSize=8, grid_settings="F2P_lr_h2"):
#     """
#     Perform quantization on all layers of a model (weights and biases).
#     Args:
#         model: torch.nn.Module
#         quantization_type: 'random', 'INT8', 'INT16', or 'F2P'
#         signed: Boolean for signed quantization
#         cntrSize: Bit width for INT-based quantization
#         grid_settings: String for F2P grid configuration
#     """
#     print("Quantization Type:", quantization_type)
#
#     for name, layer in model.named_modules():
#         if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
#             print(f"Quantizing Layer: {name}")
#
#             # Quantize weights
#             if layer.weight is not None:
#                 print(f"Original Weights (First 5): {layer.weight.data.flatten()[:5]}")
#                 layer.weight.data = quantize_tensor(
#                     tensor=layer.weight.data,
#                     quantization_type=quantization_type,
#                     signed=signed,
#                     cntrSize=cntrSize,
#                     grid_settings=grid_settings
#                 )
#                 print(f"Quantized Weights (First 5): {layer.weight.data.flatten()[:5]}")
#
#             # Quantize biases
#             if layer.bias is not None:
#                 print(f"Original Biases (First 5): {layer.bias.data.flatten()[:5]}")
#                 layer.bias.data = quantize_tensor(
#                     tensor=layer.bias.data,
#                     quantization_type=quantization_type,
#                     signed=signed,
#                     cntrSize=cntrSize,
#                     grid_settings=grid_settings
#                 )
#                 print(f"Quantized Biases (First 5): {layer.bias.data.flatten()[:5]}")
#
#     return model


def test_5pic(model, device="cpu"):
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
        else:
            print("Prediction failed.")


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

    VERBOSE = [""]

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
    # model = quantize_all_layers(model, "random", 8)
    # quanitizied with INT8, INT16, F2P, Morris.
    # model = quantize_model(model)

    if 'print5weights' in VERBOSE:
        print("Original model")
        print_all_5weights(model)
        print("Quantized model")
        model = quantize_all_layers(model, quantization_type="random", cntrSize=8)
        print_all_5weights(model)
        error(1)

    if 'printFirstLayer5weights' in VERBOSE:  # print the first 5 weights of the first layer
        vec2quantize = model.conv1.weight.data.flatten()[:5]  # $$$
        print(f'b4 {vec2quantize}')  # $$$
        quantized_vec = random_tensor(vec2quantize)
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
        test_5pic(model, image_path_dog)


if __name__ == '__main__':
    main()

# F2P flavors: sr, lr, si, li
# h
# Example: F2P_lr_h2, F2P_sr_h2,
