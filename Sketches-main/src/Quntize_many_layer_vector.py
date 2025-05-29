import os

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import Quantizer
import quantizationItamar
from PIL import Image
import torchvision.transforms as transforms

def save_quantized_weights(model, filename="quantized_weights.pth"):
    quantized_weights = {name: layer.weight.detach().cpu() for name, layer in model.named_modules() if isinstance(layer, (nn.Linear, nn.Conv2d))}
    torch.save(quantized_weights, filename)
    print(f"Quantized weights saved to {filename}")

def load_quantized_weights(model, filename="quantized_weights.pth"):
    if os.path.exists(filename):
        quantized_weights = torch.load(filename)
        for name, layer in model.named_modules():
            if name in quantized_weights:
                layer.weight.data = quantized_weights[name].to(layer.weight.device)
        print(f"Loaded quantized weights from {filename}")
    else:
        print(f"No saved quantized weights found at {filename}, running quantization again.")


def test_quantization(model, image_path, grid_type="int", cntr_size=14):
    torch.manual_seed(42)
    np.random.seed(42)

    image_tensor = load_and_preprocess_image(image_path)

    print(f"Image Tensor: shape={image_tensor.shape}, min={image_tensor.min()}, max={image_tensor.max()}")

    if torch.isnan(image_tensor).any():
        raise ValueError("NaN detected in the input image tensor!")


    if grid_type.startswith("int"):
        grid = quantizationItamar.generate_grid(cntr_size, signed=True)
    elif grid_type.startswith("F2P"):
        grid = Quantizer.getAllValsFxp(fxpSettingStr=grid_type, cntrSize=cntr_size, signed=True)
    else:
        raise ValueError("Invalid grid_type. Choose 'int' or 'F2P'")

    if not isinstance(grid, np.ndarray):
        raise TypeError("Generated grid must be a NumPy array!")
    if len(grid) == 0:
        raise ValueError("Generated grid is empty!")
    if np.isnan(grid).any():
        raise ValueError("NaN detected in the quantization grid!")

    print(f"Quantization Grid: min={np.min(grid)}, max={np.max(grid)}")

    weight_file = f"quantized_weights_{grid_type}_{cntr_size}.pth"
    if os.path.exists(weight_file):
        load_quantized_weights(model, weight_file)
    else:
        scale_factors = quantize_model(model, grid)
        save_quantized_weights(model, weight_file)

    model.eval()
    with torch.no_grad():
        original_output = model(image_tensor).detach().numpy()

    # if np.isnan(original_output).any():
    #     raise ValueError("NaN detected in model output!")

    print(f"Model Output for {image_path}: {np.sort(original_output)}")

    image_tensor_Q, scale_factor_image_Q, zero_point_image_Q = quantize(image_tensor.numpy().flatten(), grid=grid)

    image_tensor_Q = torch.tensor(image_tensor_Q.reshape(image_tensor.shape), dtype=torch.float32,
                                  device=image_tensor.device)

    quantize_output = model(image_tensor_Q).detach().numpy()

    diquantized_output = quantize_output * scale_factor_image_Q * np.prod(scale_factors)

    print(f"Dequantized Output: {np.sort(diquantized_output)}")

    print("--- End of Test ---\n")

def quantize(vec, grid, clamp_outliers=False, lower_bnd=None, upper_bnd=None):
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    if np.min(grid) >= 0:
        raise ValueError("Grid must be signed (contain negative values)")

    if clamp_outliers:
        if lower_bnd is None or upper_bnd is None:
            raise ValueError("Clamping requested but bounds not specified")
        vec = np.clip(vec, lower_bnd, upper_bnd)

    abs_max_vec = np.max(np.abs(vec))
    abs_max_grid = np.max(np.abs(grid))

    if abs_max_grid == 0:
        raise ValueError("Invalid quantization grid: max absolute value is 0!")

    scale = abs_max_vec / abs_max_grid if abs_max_grid != 0 else 1.0
    scaled_vec = vec / scale

    quantized_vec = np.array([grid[np.argmin(np.abs(grid - val))] for val in scaled_vec])

    return quantized_vec, scale, 0


def quantize_model(model, grid):
    scale_factors = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing layer: {name}")

            weights_np = layer.weight.detach().cpu().numpy().flatten()

            if np.isnan(weights_np).any():
                raise ValueError(f"NaN detected in original weights of {name}!")

            quantized_weights, scale_factor, zero_point = quantize(weights_np, grid)

            if np.isnan(quantized_weights).any():
                raise ValueError(f"NaN detected in quantized weights of {name}!")

            layer.weight.data = torch.tensor(quantized_weights.reshape(layer.weight.shape), dtype=torch.float32,
                                           )

            scale_factors.append(scale_factor)

    return np.array(scale_factors)


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def test_model_on_image(model, image_path):
    image_tensor = load_and_preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    if torch.isnan(output).any():
        raise ValueError("NaN detected in test model output!")

    print(f"Model output for {image_path}: {output.numpy()}")


if __name__ == '__main__':
    # image_path = r"C:\Users\97253\OneDrive\שולחן העבודה\final project\Sketches-main\src\5pics\cat.jpeg"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    list_of_grid_types = ["int"]
    list_cntSize = [8]
    for grid_type in list_of_grid_types:
        for cntr_size in list_cntSize:
            print(f"Testing grid: {grid_type}, counter size: {cntr_size}")
            image_path = r"C:\\Users\\97253\\OneDrive\\שולחן העבודה\\final project\\Sketches-main\\src\\5pics\\cat.jpeg"
            test_quantization(model, image_path, grid_type=grid_type, cntr_size=cntr_size)
            print("--------------------------------------------------")

    # image_tensor = load_and_preprocess_image(image_path)
    #
    # print(f"Image Tensor: shape={image_tensor.shape}, min={image_tensor.min()}, max={image_tensor.max()}")
    # original_output = model(image_tensor).detach().numpy()
    # print(f"Model Output for {image_path}: {original_output}")
