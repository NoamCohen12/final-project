import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import sys

from datetime import datetime
from settings import *
import Quantizer
from write_to_file_and_print_to_terminal import save_terminal_output_to_report

# הוספת נתיב לקובץ Quantizer
sys.path.append(r"/Sketches-main/src")

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


# TODO: create 3 files for each run
def write_results_to_file(method, precision, recall):
    """
    Write the precision and recall results of a quantization method to a new file for each run.

    Args:
        method (str): The quantization method used.
        precision (float): The precision of the method.
        recall (float): The recall of the method.
    """
    try:
        # Get the current date and time for the file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define the directory and ensure it exists
        report_directory = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)

        # Create a unique file path for each run using the timestamp
        file_path = os.path.join(report_directory, f"results_{timestamp}.txt")

        # Append results to the file
        with open(file_path, "a") as file:
            file.write("--------------------------------------------\n")
            file.write(f"Method: {method}\n")
            file.write(f"Precision: {precision:.3f}\n")
            file.write(f"Recall: {recall:.3f}\n")
            file.write("\n")

        print(f"Results for method {method} written to {file_path} successfully.")
    except Exception as e:
        print(f"Failed to write to file: {e}")


def get_precision_of_all_quotations(list_of_images, device="cpu", test=False) -> None:
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
    for method in ["INT8", "INT16", "F2P","random"]:
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

                if label:
                    prediction_list_quant.append(label)

            # Compare predictions and calculate precision
            precision = check_precision(prediction_list, prediction_list_quant)
            recall = check_recall(prediction_list, prediction_list_quant)
            print(f"Precision of {method}: {precision:.3f}")
            print(f"Recall of {method}: {recall:.3f}")

            # Save results to file
            write_results_to_file(method, precision, recall)
        except Exception as e:
            print(f"Failed to process method {method}: {e}")


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
        "INT8": lambda m: quantize_all_layers(m, quantization_type="INT", cntrSize=8),
        "INT16": lambda m: quantize_all_layers(m, quantization_type="INT", cntrSize=16),
        "F2P": lambda m: quantize_all_layers(m, quantization_type="F2P", cntrSize=8),
        "random": lambda m: quantize_all_layers(m, quantization_type="random", cntrSize=8),

    }

    # Apply the selected quantization method
    quantized_model = quantization_methods.get(method, lambda m: m)(model)
    print(f"Model quantized with {method}")
    return quantized_model


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
    with open("imagenet_class_index.json") as labels_file:
        labels = json.load(labels_file)

    label = labels[str(class_index)]

    return label, class_index, probabilities


# Quantizes the given model based on the specified method.
#
# Args:
# quantization type (str): (a)symmetric


def generate_grid(cntrSize, signed):
    """
    Generate the grid for quantization.
    """
    if signed:
        grid = np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1))
        return grid
    else:
        grid = np.array(range(2 ** cntrSize))
        return grid


def quantize_tensor(tensor, grid, verbose=False, type=""):
    """
    Quantize and dequantize a tensor using a given grid.
    """
    flattened = tensor.view(-1).numpy()
    quantized, scale, zero_point = Quantizer.quantize(flattened, grid=grid)
    if verbose:
        print(f"type {type},Quantized: {quantized}, Scale: {scale}, Zero Point: {zero_point}")
    dequantized = Quantizer.dequantize(quantized, scale, z=zero_point)
    return torch.from_numpy(dequantized).view(tensor.shape).to(tensor.device)


def process_layer_parameter(parameter, quantization_type, cntrSize, signed, verbose=False, type=""):
    """
    Process a single layer parameter (weight or bias) for quantization.
    #todo:change the parm name
    """
    if parameter is None:
        return
    if parameter.dtype == torch.float64:
        parameter.data = parameter.data.float()

    match quantization_type:
        case "random":
            parameter.data = random_tensor(parameter.data)
        case "INT":
            grid = generate_grid(cntrSize, signed)
            parameter.data = quantize_tensor(parameter.data, grid, verbose, type)

        case "F2P":
            grid = Quantizer.getAllValsFxp(
                fxpSettingStr='F2P_lr_h2',
                cntrSize=cntrSize,
                verbose=[],
                signed=signed
            )
            parameter.data = quantize_tensor(parameter.data, grid, verbose, type)
        case _:
            raise ValueError(f"Invalid quantization type '{quantization_type}'")


def quantize_all_layers(
        model,
        quantization_type: str = "random",
        signed: bool = False,
        cntrSize=8,
        verbose: bool = False) -> torch.nn.Module:
    """
    Perform quantization on all layers of the model.
    """
    print("quant_type:", quantization_type)
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            # Process weights
            if (verbose):
                print(f"Quantizing weights for layer: {name}")
            process_layer_parameter(layer.weight, quantization_type, cntrSize, signed, verbose, type="weight")

            # Process biases
            process_layer_parameter(layer.bias, quantization_type, cntrSize, signed, verbose, type="bias")

    return model


def test_5pic(model, device="cpu"):
    model = model.to(device)
    # הגדרת נתיב לתמונה שאתה רוצה לנבא עליה
    image_path_dog = r"5pics\dog.jpg"  # שם התמונה שלך
    image_path_cat = r"5pics\cat.jpeg"  # שם התמונה שלך
    image_path_kite = r"5pics\kite.jpg"  # שם התמונה שלך
    image_path_lion = r"5pics\lion.jpeg"  # שם התמונה שלך
    image_path_paper = r"5pics\paper.jpg"  # שם התמונה שלך

    array_of_path = [image_path_dog, image_path_cat, image_path_kite, image_path_lion, image_path_paper]

    # Check if the files exist
    for path in array_of_path:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

    print("All files found!")
    # ביצוע ניבוי על כל התמונות
    for image_path in array_of_path:
        # מבצע את הניבוי על התמונה
        label, class_index, probabilities = predict_image(model, image_path, device)

        if label:
            predicted_label = label
            predicted_prob = probabilities[class_index].item() * 100
            print(f"Prediction: {predicted_label[1]} (Class Index: {class_index}, Probability: {predicted_prob:.2f}%)")
        else:
            print("Prediction failed.")


def quantize_the_first_layer(model, quantization_type, cntrSize, signed):
    """
    Quantizes the first layer of the given model based on the specified method.
    """
    print("before")
    print(model.conv1.weight.data.flatten()[:5])
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing weights for layer: {name}")
            process_layer_parameter(layer.weight, quantization_type, cntrSize, signed, verbose=True)
            break
    print("after")
    print(model.conv1.weight.data.flatten()[:5])
    test_5pic(model)


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

    # # ביצוע ניבוי על כל התמונות
    # for image_path in array_of_path:
    #     # מבצע את הניבוי על התמונה
    #     compare_quantization_methods(image_path)

    # טוענים את המודל ומבצעים קוונטיזציה
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    VERBOSE = ["printFirstLayer5weights"]

    if 'printFirstLayerWeightsUniform' in VERBOSE:
        test_5pic(model)

    if 'printFirstLayer5weights' in VERBOSE:  # print the first 5 weights of the first layer
        # vec2quantize = model.conv1.weight.data.flatten()[:5]
        quantize_the_first_layer(model, "INT", 8, True)
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


if __name__ == '__main__':
    save_terminal_output_to_report(main)
    # main()

# F2P flavors: sr, lr, si, li
# h
# Example: F2P_lr_h2, F2P_sr_h2,
