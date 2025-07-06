import torch
import torch.nn as nn
import torchvision.models as models
import Quantizer
import quantizationItamar
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def register_nan_hooks(model, log_only=False, verbose_channels=True):
    def hook_fn(name):
        def inner(module, input, output):
            if isinstance(output, torch.Tensor):
                input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else input

                # בדיקות כלליות
                if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                    print(f"[!] Invalid INPUT to layer '{name}'")
                    print(f"    input min={input_tensor.min().item():.5f}, max={input_tensor.max().item():.5f}")

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"\n[!] Invalid OUTPUT after layer: '{name}' ({module.__class__.__name__})")
                    print(f"    Output shape: {tuple(output.shape)}")

                    if verbose_channels and output.ndim == 4:  # Batch x Channels x H x W
                        for c in range(output.shape[1]):
                            ch = output[0, c]
                            if torch.isnan(ch).any() or torch.isinf(ch).any():
                                ch_min = ch.min().item() if not torch.isnan(ch).all() else float('nan')
                                ch_max = ch.max().item() if not torch.isnan(ch).all() else float('nan')
                                ch_mean = ch.mean().item() if not torch.isnan(ch).all() else float('nan')
                                print(
                                    f"    → Channel {c}: NaN/Inf detected | min={ch_min:.5f}, max={ch_max:.5f}, mean={ch_mean:.5f}")

                    elif output.ndim == 2:  # Linear layer (B, Features)
                        for i in range(output.shape[1]):
                            col = output[0, i]
                            if torch.isnan(col).any() or torch.isinf(col).any():
                                print(f"    → Feature {i}: NaN/Inf detected | val={col.item():.5f}")

                    if not log_only:
                        raise RuntimeError(f"[!] NaN or Inf detected after layer '{name}'")

        return inner

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
            module.register_forward_hook(hook_fn(name))


def generate_and_validate_grid(grid_type: str, cntr_size: int) -> np.ndarray:
    if grid_type.startswith("int"):
        grid = quantizationItamar.generate_grid_INT(cntr_size, signed=True)
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
    return grid


def test_quantization(model, image_path, grid_type="int", cntr_size=14):
    torch.manual_seed(42)
    np.random.seed(42)

    image_tensor = load_and_preprocess_image(image_path)
    validate_tensor(image_tensor, "image_tensor")

    # print("image_tensor:",image_tensor)

    grid = generate_and_validate_grid(grid_type, cntr_size)

    model.eval()
    with torch.no_grad():
        original_output = model(image_tensor).detach().numpy()

    print(f"Model Output for {image_path}: {original_output}")

    image_tensor_Q, scale_factor_image_Q, zero_point_image_Q = quantize(image_tensor.numpy().flatten(), grid=grid)

    image_tensor_Q = torch.tensor(image_tensor_Q.reshape(image_tensor.shape), dtype=torch.float32,
                                  device=image_tensor.device)

    validate_tensor(image_tensor_Q, "image_tensor_Q")

    scale_factor_model = quantize_model(model, grid)

    # cast to float64 for safe log/exp computation
    scale_factor_model = scale_factor_model.astype(np.float64)
    scale_factor_image_Q = float(scale_factor_image_Q)
    print("Scale factors:", scale_factor_model)
    print("Scale factor for image tensor:", scale_factor_image_Q)

    if np.any(scale_factor_model <= 0) or scale_factor_image_Q <= 0:
        raise ValueError("Scale factors must be strictly positive for logarithmic scaling.")

    log_prod = np.log(scale_factor_image_Q) + np.sum(np.log(scale_factor_model))
    final_scale = np.exp(log_prod)

    register_nan_hooks(model, log_only=False, verbose_channels=True)

    quantize_output = model(image_tensor_Q).detach().numpy().astype(np.float64)

    validate_tensor(quantize_output, "quantize_output")

    dequantized_output = quantize_output * final_scale

    validate_tensor(dequantized_output, "dequantized_output")

    print("scale_factor_image_Q:", scale_factor_image_Q)
    print("scale_factor_model:", scale_factor_model)
    print(f"Final scaling factor: {final_scale}")
    print(f"Dequantized Output: {dequantized_output}")
    print("--- End of Test ---\n")

    compare_outputs_full(original_output, dequantized_output, threshold=1e-3)


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

    abs_max_vec = np.percentile(np.abs(vec), 75)#PROBLEM
    abs_max_grid = np.max(np.abs(grid))

    if abs_max_vec == 0:
        raise ValueError("Input vector is all zeros — aborting quantization.")

    if abs_max_grid == 0:
        raise ValueError("Invalid quantization grid: max absolute value is 0!")

    raw_scale = abs_max_vec / abs_max_grid

    if raw_scale < 1e-3:
        raise ValueError(
            f"⚠️ Scale too small ({raw_scale:.2e}). Aborting quantization. Try clamping or using wider grid.")

    scale = raw_scale
    scaled_vec = vec / scale


    if np.isnan(scaled_vec).any() or np.isinf(scaled_vec).any():
        raise ValueError("scaled_vec contains NaN or Inf! scale might be too small or vec contains spikes.")

    print(f"→ quantize(): scale={scale:.5e}, max abs vec={abs_max_vec:.5f}, grid size={len(grid)}")

    quantized_vec = np.array([grid[np.argmin(np.abs(grid - val))] for val in scaled_vec])

    return quantized_vec, scale, 0


def quantize_model(model, grid) -> np.ndarray:
    scale_factors = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Quantizing layer: {name}")
            try:
                weights_np = layer.weight.detach().cpu().numpy().flatten()

                if np.isnan(weights_np).any() or np.isinf(weights_np).any():
                    raise ValueError(f"[!] Layer '{name}' contains NaN or Inf in original weights.")

                # שימוש בפונקציית quantize שלך
                quantized_weights, scale, zero_point = quantize(weights_np, grid)

                validate_tensor(quantized_weights, "quantized_weights")

                # # הגנה נוספת אחרי הקוונטיזציה
                # quantized_weights = np.nan_to_num(
                #     quantized_weights,
                #     nan=0.0,
                #     posinf=np.max(np.abs(grid)) * scale,
                #     neginf=-np.max(np.abs(grid)) * scale
                # )

#INT OR FLOAT?? TODO

                layer.weight.data = torch.tensor(quantized_weights.reshape(layer.weight.shape),
                                                 dtype=torch.float32
                                                 ).to(layer.weight.device)

                if torch.isnan(layer.weight).any() or torch.isinf(layer.weight).any():
                    raise ValueError(f"NaN or Inf found in quantized weights of layer {name}")

                scale_factors.append(scale)

            except Exception as e:
                print(f"[!] Error in layer {name}: {e}")
                continue

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


def validate_tensor(tensor, name="tensor"):
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray")

    print(f"🔍 Validating '{name}': shape={arr.shape}, dtype={arr.dtype}")

    if np.isnan(arr).any():
        raise ValueError(f"❌ NaN detected in {name}")
    if np.isinf(arr).any():
        raise ValueError(f"❌ Inf detected in {name}")
    if np.all(arr == 0):
        print(f"⚠️  Warning: All values in {name} are zero.")
    if np.max(arr) - np.min(arr) == 0:
        print(f"⚠️  Warning: {name} has no variation (constant tensor).")
    if arr.dtype not in [np.float32, np.float64]:
        print(f"⚠️  Warning: {name} has unexpected dtype: {arr.dtype}")
    if np.max(np.abs(arr)) > 1e6:
        print(f"⚠️  Warning: {name} contains very large values (>1e6)")
    if np.max(np.abs(arr)) < 1e-6 and not np.all(arr == 0):
        print(f"⚠️  Warning: {name} contains very small values (<1e-6)")

    print(f"max: {np.max(arr):.5f}, min: {np.min(arr):.5f}, mean: {np.mean(arr):.5f}")
    print(f"✅ '{name}' passed validation.\n")



def compare_outputs_full(original, dequantized, threshold=1e-3):
    """
    משווה בין original ו-dequantized שורה-שורה.
    מדפיס את ההפרשים ומציג גרף עם הערכים שחורגים מה-threshold.
    """
    original = np.array(original).flatten()
    dequantized = np.array(dequantized).flatten()

    if original.shape != dequantized.shape:
        raise ValueError("Original and dequantized outputs must have the same shape.")

    diffs = np.abs(original - dequantized)
    indices = np.arange(len(diffs))
    large_diff_mask = diffs > threshold
    count_large_diff = np.sum(large_diff_mask)

    print("\n📋 Differences per index:")
    for i, (orig, dq, diff) in enumerate(zip(original, dequantized, diffs)):
        print(f"[{i:03}] original={orig:.6f} | dequantized={dq:.6f} | diff={diff:.6f}")

    print(f"\n⚠️ {count_large_diff} entries had diff > {threshold}")

    # גרף
    plt.figure(figsize=(12, 5))
    plt.plot(indices, diffs, label="|original - dequantized|", linewidth=1)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold = {threshold}")
    plt.scatter(indices[large_diff_mask], diffs[large_diff_mask], color='red', s=20, label="Above threshold")
    plt.title("Difference Between Original and Dequantized Outputs")
    plt.xlabel("Index")
    plt.ylabel("Absolute Difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



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
