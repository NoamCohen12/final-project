# import Quantizer
#
# grid = Quantizer.getAllValsFP(cntrSize=4, expSize=2, verbose=[], signed=False)
# print(grid)
# grid = Quantizer.getAllValsFP(cntrSize=4, expSize=2, verbose=[], signed=False)
# print(grid)
# grid = Quantizer.getAllValsFP(cntrSize=4, expSize=3, verbose=[], signed=False)
# print(grid)
# grid = Quantizer.getAllValsFP(cntrSize=5, expSize=4, verbose=[], signed=False)
# print(grid)
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
import matplotlib.pyplot as plt

def register_nan_hooks(model):
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Clamp output to a wider range
            output = torch.clamp(output, min=-500.0, max=500.0)
            print(f"Output of {module}: min={output.min().item()}, max={output.max().item()}")
            if torch.isnan(output).any():
                raise RuntimeError(f"NaN detected after layer: {module}")
        return output

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
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
    image_tensor = load_and_preprocess_image(image_path)
    print(f"Image Tensor: shape={image_tensor.shape}, min={image_tensor.min()}, max={image_tensor.max()}")

    if torch.isnan(image_tensor).any():
        raise ValueError("NaN detected in the input image tensor!")

    # Run original model for comparison
    model.eval()
    with torch.no_grad():
        original_output = model(image_tensor).detach().numpy()
        original_softmax = torch.softmax(torch.tensor(original_output), dim=1).numpy()
        original_top5_indices = np.argsort(original_softmax[0])[-5:][::-1]
        original_top5_scores = original_softmax[0][original_top5_indices]
        print(f"Original Top-5 predicted classes: {original_top5_indices}")
        print(f"Original Top-5 scores: {original_top5_scores}")

    if grid_type.startswith("int"):
        grid = quantizationItamar.generate_grid(cntr_size, signed=True)
        print(f"Grid shape: {grid.shape}, values: {grid[:5]}...{grid[-5:]}")

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



    # Collect activation ranges from the original model
    activation_ranges = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_ranges[name] = (output.min().item(), output.max().item())

        return hook

    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module.register_forward_hook(hook_fn(name))
        model(image_tensor)
        for name in activation_ranges:
            print(f"Activation range for {name}: min={activation_ranges[name][0]}, max={activation_ranges[name][1]}")

        # Use activation ranges to compute scale factors
        scale_factor_model = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                min_val, max_val = activation_ranges.get(name, (-1, 1))  # Default range if not found
                range_val = max(abs(min_val), abs(max_val))
                scale = range_val / 127 if cntr_size == 8 else range_val / 7  # Adjust based on grid
                scale_factor_model.append(scale if scale > 1e-6 else 1e-6)
        scale_factor_model = np.array(scale_factor_model)
        print(
            f"Activation-based scale_factor_model: min={np.min(scale_factor_model)}, max={np.max(scale_factor_model)}")

        # Quantize weights separately
        quantize_model(model, grid)  # This updates weights but we won't use its scale factors

        image_tensor_Q, scale_factor_image_Q, zero_point_image_Q = quantize(
            image_tensor.numpy().flatten(), grid, clamp_outliers=True, lower_bnd=np.min(grid), upper_bnd=np.max(grid)
        )
        image_tensor_Q = torch.tensor(image_tensor_Q.reshape(image_tensor.shape), dtype=torch.float32,
                                      device=image_tensor.device)

        if torch.isnan(image_tensor_Q).any():
            raise ValueError("NaN found in quantized input image_tensor_Q!")

        # Numerical stability for scale factors
        scale_factor_model = np.clip(scale_factor_model.astype(np.float64), 1e-6, None)
        scale_factor_image_Q = max(float(scale_factor_image_Q), 1e-6)

        print(f"scale_factor_image_Q: {scale_factor_image_Q}")
        print(f"scale_factor_model: min={np.min(scale_factor_model)}, max={np.max(scale_factor_model)}")

        register_nan_hooks(model)

        with torch.no_grad():
            x = image_tensor_Q
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4[0](x)
            x = model.layer4[1].conv1(x)
            x = model.layer4[1].bn1(x)
            x = model.layer4[1].relu(x)
            x = model.layer4[1].conv2(x)
            quantize_output = model(image_tensor_Q).detach().numpy().astype(np.float64)

        print(f"Quantize output min/max: {np.nanmin(quantize_output)}, {np.nanmax(quantize_output)}")
        print(f"Any NaN in quantize_output: {np.isnan(quantize_output).any()}")

        # חדש: חישוב final_scale לאחר quantize_output
        fc_range = activation_ranges.get('fc', (-1, 1))  # טווח הפלט של fc
        abs_max_fc = max(abs(fc_range[0]), abs(fc_range[1]))  # 14.447
        final_scale = abs_max_fc / np.max(np.abs(quantize_output))  # התאם את הפלט לטווח המקורי
        if np.isnan(final_scale) or np.isinf(final_scale):
            raise ValueError(f"Invalid final_scale: {final_scale}")
        print(f"Final scaling factor: {final_scale}")

        dequantized_output = quantize_output * final_scale
        # Apply softmax to dequantized output for classification
        softmax_output = torch.softmax(torch.tensor(dequantized_output), dim=1).numpy()
        top5_indices = np.argsort(softmax_output[0])[-5:][::-1]
        top5_scores = softmax_output[0][top5_indices]
        print(f"Top-5 predicted classes: {top5_indices}")
        print(f"Top-5 scores: {top5_scores}")

        print(f"Dequantized Output: {np.sort(dequantized_output)}")
        visualize_quantized_image(image_tensor, scale_factor_image_Q)
        compare_outputs(original_output.flatten(), dequantized_output.flatten())

        print("--- End of Test ---\n")



def visualize_quantized_image(image_tensor, scale_factor_image_Q):
    # קוונטיזציה של תמונת הקלט
    quantized_image = torch.round(image_tensor / scale_factor_image_Q) * scale_factor_image_Q

    # המרת התמונה המקורית והקוונטיזצית לפורמט תצוגה
    # הסרת מימד ה-batch והעברת ערוצי הצבע למקום הנכון: [1, 3, 224, 224] -> [224, 224, 3]
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    quantized_image_np = quantized_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # נרמול התמונות להצגה (אם הן לא כבר מנורמלות ל-[0, 1])
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    quantized_image_np = (quantized_image_np - quantized_image_np.min()) / (
                quantized_image_np.max() - quantized_image_np.min())

    # הצגת התמונות
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Dequantized Image")
    plt.imshow(quantized_image_np)
    plt.axis('off')

    plt.show()


def compare_outputs(original, dequantized):
    import matplotlib.pyplot as plt
    import numpy as np

    mae = np.mean(np.abs(original - dequantized))
    rmse = np.sqrt(np.mean((original - dequantized)**2))

    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(original.flatten(), label='Original')
    plt.plot(dequantized.flatten(), label='Dequantized')
    plt.title('Original vs Dequantized Outputs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(original.flatten() - dequantized.flatten(), color='red')
    plt.title('Error (Original - Dequantized)')

    plt.show()




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
            abs_max_weight = np.max(np.abs(weights_np))
            scale = abs_max_weight / 511  # שינוי מ-255 ל-511
            quantized_weights = np.round(weights_np / scale) * scale
            print(f"Layer {name} - Original weights max: {abs_max_weight}, Scale: {scale}, Quantized weights max: {np.max(np.abs(quantized_weights))}")
            layer.weight.data = torch.tensor(quantized_weights.reshape(layer.weight.shape), dtype=torch.float32)
            if torch.isnan(layer.weight).any():
                raise ValueError(f"NaN present in weights of layer '{name}' after quantization!")
            scale_factors.append(scale)
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
            image_path = r"5pics/kite.jpg"
            test_quantization(model, image_path, grid_type=grid_type, cntr_size=cntr_size)
            print("--------------------------------------------------")

    # image_tensor = load_and_preprocess_image(image_path)
    #
    # print(f"Image Tensor: shape={image_tensor.shape}, min={image_tensor.min()}, max={image_tensor.max()}")
    # original_output = model(image_tensor).detach().numpy()
    # print(f"Model Output for {image_path}: {original_output}")
