import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from SingleCntrSimulator import getAllValsFP



def generate_grid(cntrSize,signed, expSize=1, type='INT'):
    """
    Generate the grid for quantization.
    """
    if type == "INT":
        if signed:
            grid = np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1))
            return grid
        else:
            grid = np.array(range(2 ** cntrSize))
            return grid
    else:
        grid = getAllValsFP(cntrSize, expSize=1, signed=signed)
        return grid

def quantize(
    vec, grid,
    clamp_outliers=False,
    lower_bnd=None, upper_bnd=None,
    percentile_limits=(1, 99),
    std_multiplier=3
):
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    if np.min(grid) >= 0:
        raise ValueError("Grid must be signed (contain negative values)")

    if clamp_outliers:
        if clamp_outliers == "percentile":
            lower_bnd = np.percentile(vec, percentile_limits[0])
            upper_bnd = np.percentile(vec, percentile_limits[1])
        elif clamp_outliers == "std":
            mean = np.mean(vec)
            std = np.std(vec)
            lower_bnd = mean - std_multiplier * std
            upper_bnd = mean + std_multiplier * std
        elif lower_bnd is not None and upper_bnd is not None:
            pass  # Use provided bounds
        else:
            raise ValueError("Clamping requested but bounds not specified or mode invalid")

        vec = np.clip(vec, lower_bnd, upper_bnd)

    abs_max_vec = np.percentile(np.abs(vec), 99.9)
    abs_max_grid = np.max(np.abs(grid))

    if abs_max_vec == 0 or abs_max_grid == 0:
        raise ValueError("Invalid quantization input or grid")

    scale = abs_max_vec / abs_max_grid
    scaled_vec = vec / scale

    if np.isnan(scaled_vec).any() or np.isinf(scaled_vec).any():
        raise ValueError("scaled_vec contains NaN or Inf")

    quantized_vec = np.array([grid[np.argmin(np.abs(grid - val))] for val in scaled_vec])
    return quantized_vec, scale, 0



def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def quantize_model_weights(model, grid, debug=False, clamp_outliers=None, percentile_limits=(1, 99), std_multiplier=3):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            w = layer.weight.detach().cpu().numpy().flatten()
            w_q, scale, _ = quantize(w, grid,
                                     clamp_outliers=clamp_outliers,
                                     percentile_limits=percentile_limits,
                                     std_multiplier=std_multiplier)
            quant_info = {
                'w_q': torch.tensor(w_q.reshape(layer.weight.shape), dtype=torch.float32),
                'scale_w': scale
            }

            if layer.bias is not None:
                quant_info['bias'] = layer.bias.detach().cpu().numpy().flatten()

            layer._quant_info = quant_info

            if debug:
                print(
                    f"[Quantized] {name} ({type(layer).__name__}) - weight shape: {layer.weight.shape}, scale: {scale:.4g}")

        # Quantize downsample conv if exists
        elif hasattr(layer, 'downsample') and isinstance(layer.downsample, nn.Sequential):
            for i, sublayer in enumerate(layer.downsample):
                if isinstance(sublayer, nn.Conv2d):
                    w = sublayer.weight.detach().cpu().numpy().flatten()
                    w_q, scale, _ = quantize(w, grid)
                    sublayer._quant_info = {
                        'w_q': torch.tensor(w_q.reshape(sublayer.weight.shape), dtype=torch.float32),
                        'scale': scale
                    }
                    if debug:
                        print(
                            f"[Quantized] {name}.downsample[{i}] ({type(sublayer).__name__}) - weight shape: {sublayer.weight.shape}, scale: {scale:.4g}")


def q_layer(x, layer, grid):
    x_np = x.detach().cpu().numpy().flatten()
    x_q, s_x, _ = quantize(x_np, grid)
    x_q = torch.tensor(x_q.reshape(x.shape), dtype=torch.float32).to(x.device)

    w_q = layer._quant_info['w_q'].to(x.device)
    s_w = layer._quant_info['scale_w']

    if 'bias' in layer._quant_info:
        b_w = layer._quant_info['bias']
        b_w = torch.tensor(b_w.reshape(layer.bias.shape), dtype=torch.float32).to(x.device)
        print(f"Bias shape: {layer.bias.shape}, bias: {b_w.shape}")
    else:
        b_w = None


    if isinstance(layer, nn.Conv2d):
        z_q = F.conv2d(x_q, w_q, bias=b_w, stride=layer.stride,
                       padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
    elif isinstance(layer, nn.Linear):
        z_q = F.linear(x_q, w_q, bias=b_w)
    else:
        raise NotImplementedError

    return z_q * (s_x * s_w)


def run_quantized_forward(model, x, grid, debug=False):
    with torch.no_grad():
        if debug: print("Entering: conv1")
        x = q_layer(x, model.conv1, grid)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for block in model.layer1:
            residual = block.downsample(x) if block.downsample else x
            if isinstance(block.downsample, nn.Sequential):
                for ds_layer in block.downsample:
                    if isinstance(ds_layer, nn.Conv2d):
                        if debug: print("Layer1 - downsample conv")
                        residual = q_layer(x, ds_layer, grid)
                    elif isinstance(ds_layer, nn.BatchNorm2d):
                        residual = ds_layer(residual)

            if debug: print("Layer1 - conv1")
            x = q_layer(x, block.conv1, grid)
            x = block.bn1(x)
            x = model.relu(x)
            if debug: print("Layer1 - conv2")
            x = q_layer(x, block.conv2, grid)
            x = block.bn2(x)
            x += residual
            x = block.relu(x)

        for block in model.layer2:
            residual = block.downsample(x) if block.downsample else x
            if isinstance(block.downsample, nn.Sequential):
                for ds_layer in block.downsample:
                    if isinstance(ds_layer, nn.Conv2d):
                        if debug: print("Layer2 - downsample conv")
                        residual = q_layer(x, ds_layer, grid)
                    elif isinstance(ds_layer, nn.BatchNorm2d):
                        residual = ds_layer(residual)

            if debug: print("Layer2 - conv1")
            x = q_layer(x, block.conv1, grid)
            x = block.bn1(x)
            x = model.relu(x)
            if debug: print("Layer2 - conv2")
            x = q_layer(x, block.conv2, grid)
            x = block.bn2(x)
            x += residual
            x = block.relu(x)

        for block in model.layer3:
            residual = block.downsample(x) if block.downsample else x
            if isinstance(block.downsample, nn.Sequential):
                for ds_layer in block.downsample:
                    if isinstance(ds_layer, nn.Conv2d):
                        if debug: print("Layer3 - downsample conv")
                        residual = q_layer(x, ds_layer, grid)
                    elif isinstance(ds_layer, nn.BatchNorm2d):
                        residual = ds_layer(residual)

            if debug: print("Layer3 - conv1")
            x = q_layer(x, block.conv1, grid)
            x = block.bn1(x)
            x = model.relu(x)
            if debug: print("Layer3 - conv2")
            x = q_layer(x, block.conv2, grid)
            x = block.bn2(x)
            x += residual
            x = block.relu(x)

        for block in model.layer4:
            residual = block.downsample(x) if block.downsample else x
            if isinstance(block.downsample, nn.Sequential):
                for ds_layer in block.downsample:
                    if isinstance(ds_layer, nn.Conv2d):
                        if debug: print("Layer4 - downsample conv")
                        residual = q_layer(x, ds_layer, grid)
                    elif isinstance(ds_layer, nn.BatchNorm2d):
                        residual = ds_layer(residual)

            if debug: print("Layer4 - conv1")
            x = q_layer(x, block.conv1, grid)
            x = block.bn1(x)
            x = block.relu(x)
            if debug: print("Layer4 - conv2")
            x = q_layer(x, block.conv2, grid)
            x = block.bn2(x)
            x += residual
            x = block.relu(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        if debug: print("Entering: fc")
        x = q_layer(x, model.fc, grid)
        return x


def evaluate_quantization(output_fp, output_q):
    # Softmax probabilities
    prob_fp = torch.softmax(output_fp, dim=1)
    prob_q = torch.softmax(output_q, dim=1)

    # Top-1 prediction
    top1_fp = torch.argmax(prob_fp, dim=1)
    top1_q = torch.argmax(prob_q, dim=1)

    # Top-5 predictions
    top5_fp = torch.topk(prob_fp, 5).indices[0].tolist()
    top5_q = torch.topk(prob_q, 5).indices[0].tolist()
    top5_intersection = set(top5_fp) & set(top5_q)

    # Errors (pre-softmax)
    diff = output_fp - output_q
    mae = torch.mean(torch.abs(diff)).item()
    max_error = torch.max(torch.abs(diff)).item()


    # Relative errors (avoid divide-by-zero)
    denom = torch.abs(output_fp) + 1e-8
    rel_error = torch.abs(diff) / denom
    max_error_relative = torch.max(rel_error).item()

    # Confidence
    conf_fp = prob_fp[0, top1_fp].item()
    conf_q = prob_q[0, top1_q].item()

    # Mean Squared Error (MSE)
    mse = torch.mean(diff ** 2).item()

    # Print results
    print(f"Top-1 Match: {'✅' if top1_fp.item() == top1_q.item() else '❌'}")
    print(f"  • Float: {top1_fp.item()}, Quantized: {top1_q.item()}")
    print(f"Top-5 Match Count: {len(top5_intersection)} / 5")
    print(f"  • Float Top-5: {top5_fp}")
    print(f"  • Quantized Top-5: {top5_q}")
    print(f"  • Intersection: {list(top5_intersection)}")
    print(f"L2 Error: {torch.norm(diff).item():.6f}")
    print(f"Mean Absolute Error (mean_error): {mae:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    print(f"Max Relative Error (max_error_relative): {max_error_relative:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Confidence (FP): {conf_fp:.3f}")
    print(f"Confidence (Quantized): {conf_q:.3f}")
    print(f"Confidence Δ: {abs(conf_fp - conf_q):.3f}")


if __name__ == '__main__':
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    image_path = "5pics/dog.jpg"
    image_tensor = preprocess_image(image_path)
    # cntrSize = 16
    # # Generate a signed grid for quantization between -2^cntrSize and 2^cntrSize
    # grid = generate_grid(cntrSize, signed=True, type='INT')
    #
    # quantize_model_weights(model, grid, debug=True)
    # output = run_quantized_forward(model, image_tensor, grid, debug=True)
    #
    # output_fp = model(image_tensor)
    # print("for cntrSize:", cntrSize)
    # evaluate_quantization(output_fp, output)
    #


fp_configs = [
    {'cntrSize': 8,  'expSize': 3, 'name': 'FP8_E3M4', 'clamp': None},                  # 1 sign + 3 exponent + 4 mantissa, no clamping
    {'cntrSize': 8,  'expSize': 3, 'name': 'FP8_E3M4_clamped', 'clamp': "percentile"},  # 1 sign + 3 exponent + 4 mantissa, clamped to percentile
    {'cntrSize': 8, 'expSize': 3, 'name': 'FP8_E3M4_clamped', 'clamp': "std"},          # 1 sign + 3 exponent + 4 mantissa, clamped to std
    {'cntrSize': 8,  'expSize': 4, 'name': 'FP8_E4M3', 'clamp': None},                  # 1 sign + 4 exponent + 3 mantissa, no clamping
    {'cntrSize': 8,  'expSize': 4, 'name': 'FP8_E4M3_clamped', 'clamp': "percentile"},  # 1 sign + 4 exponent + 3 mantissa, clamped to percentile
    {'cntrSize': 8, 'expSize': 4, 'name': 'FP8_E4M3_clamped', 'clamp': "std"},          # 1 sign + 4 exponent + 3 mantissa, clamped to std
    {'cntrSize': 8,  'expSize': 5, 'name': 'FP8_E5M2', 'clamp': None},                  # 1 sign + 5 exponent + 2 mantissa, no clamping
    {'cntrSize': 8,  'expSize': 5, 'name': 'FP8_E5M2_clamped', 'clamp': "percentile"},  # 1 sign + 5 exponent + 2 mantissa, clamped to percentile
    {'cntrSize': 8, 'expSize': 5, 'name': 'FP8_E5M2_clamped', 'clamp': "std"},          # 1 sign + 5 exponent + 2 mantissa, clamped to std
    {'cntrSize': 8,  'expSize': 6, 'name': 'FP8_E6M1', 'clamp': None},                  # 1 sign + 6 exponent + 1 mantissa, no clamping
    {'cntrSize': 8,  'expSize': 6, 'name': 'FP8_E6M1_clamped', 'clamp': "percentile"},  # 1 sign + 6 exponent + 1 mantissa, clamped to percentile
    {'cntrSize': 8, 'expSize': 6, 'name': 'FP8_E6M1_clamped', 'clamp': "std"},          # 1 sign + 6 exponent + 1 mantissa, clamped to std
    {'cntrSize': 12, 'expSize': 5, 'name': 'FP12_E5M6', 'clamp': None},                 # 1 sign + 5 exponent + 6 mantissa, no clamping
    {'cntrSize': 12, 'expSize': 5, 'name': 'FP12_E5M6', 'clamp': "percentile"},         # 1 sign + 5 exponent + 6 mantissa, clamped to percentile
    {'cntrSize': 12, 'expSize': 5, 'name': 'FP12_E5M6_std', 'clamp': "std"},            # 1 sign + 5 exponent + 6 mantissa, clamped to std
    {'cntrSize': 16, 'expSize': 8, 'name': 'FP16_E8M7', 'clamp': None},                 # 1 sign + 8 exponent + 7 mantissa, no clamping
    {'cntrSize': 16, 'expSize': 8, 'name': 'FP16_E8M7', 'clamp': "percentile"},         # 1 sign + 8 exponent + 7 mantissa, clamped to percentile
    {'cntrSize': 16, 'expSize': 8, 'name': 'FP16_E8M7_std', 'clamp': "std"}             # 1 sign + 8 exponent + 7 mantissa, clamped to std
]


for cfg in fp_configs:
    print("\n==========================")
    print(f"Running for {cfg['name']} (cntr={cfg['cntrSize']}, exp={cfg['expSize']})")
    grid = generate_grid(cntrSize=cfg['cntrSize'], signed=True,expSize=cfg['expSize'],  type='FP')

    quantize_model_weights(
        model,
        grid,
        debug=True,
        clamp_outliers=cfg.get('clamp', None),
        percentile_limits=(1, 99),
        std_multiplier=3
    )
    output = run_quantized_forward(model, image_tensor, grid, debug=True)
    output_fp = model(image_tensor)

    evaluate_quantization(output_fp, output)
