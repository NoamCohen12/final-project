import torch
import torchvision.models as models
import numpy as np
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


def clamp(
        vec: np.array,
        lowerBnd: float,
        upperBnd: float) -> np.array:
    """
    Clamp a the input vector vec, as follows.
    For each item in vec:
    - if x<min(grid), assign x=lowrBnd
    - if x>max(grid), assign x=upperBnd
    """
    vec[vec < lowerBnd] = lowerBnd
    vec[vec > upperBnd] = upperBnd
    return vec


# כאן אתה יכול להגדיר את הפונקציות quantize ו-dequantize כפי שסיפקת קודם

def quantize(vec: np.array,  # The vector to quantize
             grid: np.array
             # The quantization grid (all the values that can be represented by the destination number representation
             ) -> [np.array, float]:  # [the_quantized_vector, the scale_factor (by which the vector was divided)]
    """
    Quantize an input vector, using symmetric Min-max quantization.
    This is done by:
    - Quantizing the vector, namely:
      - Clamping and scaling the vector. The scaling method is minMax.
      - Rounding the vector to the nearest values in the grid.
    """
    vec = np.sort(vec)
    upperBnd = vec[-1]  # The upper bound is the largest absolute value in the vector to quantize.
    lowerBnd = vec[0]  # The lower bound is the largest absolute value in the vector to quantize.
    scaledVec = clamp(vec, lowerBnd, upperBnd)
    if np.any(vec != scaledVec):
        error('in Quantizer.quantize(). vec!=clamped vec.')
    grid = np.sort(grid)
    scale = (vec[-1] - vec[0]) / (max(grid) - min(grid))
    z = -vec[0] / scale
    scaledVec = vec / scale + z  # The vector after scaling and clamping (still w/o rounding)
    quantVec = np.empty(len(vec))  # The quantized vector (after rounding scaledVec)
    idxInGrid = int(0)
    for idxInVec in range(len(scaledVec)):
        if idxInGrid == len(
                grid):  # already reached the max grid val --> all next items in q should be the last item in the grid
            quantVec[idxInVec] = grid[-1]
            continue
        quantVec[idxInVec] = grid[idxInGrid]
        minAbsErr = abs(scaledVec[idxInVec] - quantVec[idxInVec])
        while (idxInGrid < len(grid)):
            quantVec[idxInVec] = grid[idxInGrid]
            absErr = abs(scaledVec[idxInVec] - quantVec[idxInVec])
            if absErr <= minAbsErr:
                minAbsErr = absErr
                idxInGrid += 1
            else:
                idxInGrid -= 1
                quantVec[idxInVec] = grid[idxInGrid]
                break
    return [quantVec, scale, z]


def dequantize(vec, scale, z):
    return (vec - z) * scale


def quantize_model_first_layer(model, grid):
    """
    Quantizes the weights of the first layer in the model using the provided quantization grid.
    """
    quantized_model = model
    scale_factors = {}
    z_factors = {}

    # Get the first layer's name and parameter
    first_layer_name, first_layer_param = list(model.named_parameters())[0]

    if len(first_layer_param.shape) in {2, 4}:  # Fully connected or Conv2d weights
        vec = first_layer_param.data.cpu().numpy().flatten()  # Flatten the weights
        quantized_vec, scale, z = quantize(vec, grid)
        quantized_model.state_dict()[first_layer_name].copy_(
            torch.tensor(quantized_vec.reshape(first_layer_param.shape)))
        scale_factors[first_layer_name] = scale
        z_factors[first_layer_name] = z

    return quantized_model, scale_factors, z_factors


def dequantize_model(quantized_model, scale_factors, z_factors):
    """
    Dequantizes the weights of the model using the stored scale and z factors.
    """
    dequantized_model = quantized_model

    for name, param in dequantized_model.named_parameters():
        if name in scale_factors:
            scale = scale_factors[name]
            z = z_factors[name]
            dequantized_vec = dequantize(param.data.cpu().numpy().flatten(), scale, z)
            dequantized_model.state_dict()[name].copy_(torch.tensor(dequantized_vec.reshape(param.shape)))

    return dequantized_model


def printing_5(model):
    print("First 5 weights of the model:")
    print(model.conv1.weight.data.flatten()[:5])


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
    # טען את המודל ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')  # או אפשר גם weights=ResNet18_Weights.DEFAULT
    printing_5(model)
    print("orginal model:")
    test_5pic(model)
    cntr_size = 16
    bool = False
    grid = (np.arange(2 ** cntr_size) if bool else
            np.arange(-2 ** (cntr_size - 1) + 1, 2 ** (cntr_size - 1)))

    # הגדר את גריד הקוונטיזציה (לדוגמה: ערכים בין -10 ל-10 עם 50 ערכים)
    quantization_grid = grid

    # כעת נעשה קוונטיזציה על משקולות המודל
    quantized_model, scale_factors, z_factors = quantize_model_first_layer(model, quantization_grid)
    printing_5(quantized_model)
    # הצגת משקולות קוונטיזציה לדוגמה
    print("Quantized model weights:")
    for name, param in quantized_model.named_parameters():
        print(f"{name}: {param.shape}")

    # עתה נחזור על קוונטיזציה ונעשה דקוונטיזציה על המשקולות
    dequantized_model = dequantize_model(quantized_model, scale_factors, z_factors)
    printing_5(dequantized_model)
    print("demodel:")
    test_5pic(model)

    # הצגת משקולות דקוונטיזציה לדוגמה
    print("\nDequantized model weights:")
    for name, param in dequantized_model.named_parameters():
        print(f"{name}: {param.shape}")


if __name__ == "__main__":
    main()

# First 5 weights of the model:
# tensor([-0.0104, -0.0061, -0.0018,  0.0748,  0.0566])
# First 5 weights of the model:
# tensor([ 0.,  0.,  1., 12., 14.])
# First 5 weights of the model:
# tensor([-0.8434, -0.8434, -0.8361, -0.7559, -0.7413])

















