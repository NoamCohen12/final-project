import os

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision.transforms import ToPILImage

import Quantizer
import quantizationItamar
from PIL import Image
import torchvision.transforms as transforms

def register_nan_hooks(model):
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            raise RuntimeError(f"NaN detected after layer: {module}")

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.BatchNorm2d)):
            module.register_forward_hook(hook_fn)

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
    # torch.manual_seed(42)
    # np.random.seed(42)

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

    model.eval()
    with torch.no_grad():
        original_output = model(image_tensor).detach().numpy()

    # print(f"Model Output for {image_path}: {np.sort(original_output)}")

    image_tensor_Q, scale_factor_image_Q, zero_point_image_Q = quantize(image_tensor.numpy().flatten(), grid=grid)

    image_tensor_Q = torch.tensor(image_tensor_Q.reshape(image_tensor.shape), dtype=torch.float32,
                                  device=image_tensor.device)

    # ← בדיקה נוספת לוודא שאין NaNs בתמונה המקוונטת
    if torch.isnan(image_tensor_Q).any():
        raise ValueError("NaN found in quantized input image_tensor_Q!")

    scale_factor_model = quantize_model(model, grid)

    # cast to float64 for safe log/exp computation
    scale_factor_model = scale_factor_model.astype(np.float64)
    scale_factor_image_Q = float(scale_factor_image_Q)

    if np.any(scale_factor_model <= 0) or scale_factor_image_Q <= 0:
        raise ValueError("Scale factors must be strictly positive for logarithmic scaling.")

    log_prod = np.log(scale_factor_image_Q) + np.sum(np.log(scale_factor_model))
    final_scale = np.exp(log_prod)
    register_nan_hooks(model)

    quantize_output = model(image_tensor_Q).detach().numpy().astype(np.float64)

    # diquantized_output = quantize_output * final_scale

    print("Quantize output min/max:", np.nanmin(quantize_output), np.nanmax(quantize_output))
    print("Any NaN in quantize_output:", np.isnan(quantize_output).any())
    diquantized_output = quantize_output

    print("final scale:", final_scale)

    print("scale_factor_image_Q:", scale_factor_image_Q)
    print("scale_factor_model:", scale_factor_model)
    print(f"Final scaling factor: {final_scale}")
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

    if abs_max_vec == 0:
        return np.zeros_like(vec), 1.0, 0
    if abs_max_grid == 0:
        raise ValueError("Invalid quantization grid: max absolute value is 0!")

    # הגנה נגד scale קטן מדי
    raw_scale = abs_max_vec / abs_max_grid
    scale = max(raw_scale, 1e-3)
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

            layer.weight.data = torch.tensor(quantized_weights.reshape(layer.weight.shape), dtype=torch.float32)

            # ← הוספת בדיקה לאחר עדכון המשקלים
            if torch.isnan(layer.weight).any():
                raise ValueError(f"NaN present in weights of layer '{name}' after quantization!")

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

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    list_of_grid_types = ["int"]
    list_cntSize = [8]
    for grid_type in list_of_grid_types:
        for cntr_size in list_cntSize:
            print(f"Testing grid: {grid_type}, counter size: {cntr_size}")
            image_path = r"5pics/dog.jpg"
            test_quantization(model, image_path, grid_type=grid_type, cntr_size=cntr_size)
            print("--------------------------------------------------")

    # image_tensor = load_and_preprocess_image(image_path)
    #
    # print(f"Image Tensor: shape={image_tensor.shape}, min={image_tensor.min()}, max={image_tensor.max()}")
    # original_output = model(image_tensor).detach().numpy()
    # print(f"Model Output for {image_path}: {original_output}")
