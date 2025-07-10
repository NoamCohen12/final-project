import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from SingleCntrSimulator import getAllValsFP


def generate_grid(cntrSize, signed, expSize=1, type='INT'):
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
        grid = getAllValsFP(cntrSize, expSize=expSize, signed=signed)
        return grid


#
# def quantize(
#         vec, grid,
#         clamp_outliers=None,  # None / 'percentile' / 'std'
#         lower_bnd=None, upper_bnd=None,
#         percentile_limits=(1, 99),
#         std_multiplier=3
# ):
#     if not isinstance(vec, np.ndarray):
#         vec = np.array(vec)
#     if not isinstance(grid, np.ndarray):
#         grid = np.array(grid)
#
#     if np.min(grid) >= 0:
#         raise ValueError("Grid must be signed (contain negative values)")
#     if clamp_outliers is not None:
#         if clamp_outliers:
#             if clamp_outliers == "percentile":
#                 lower_bnd = np.percentile(vec, percentile_limits[0])
#                 upper_bnd = np.percentile(vec, percentile_limits[1])
#             elif clamp_outliers == "std":
#                 mean = np.mean(vec)
#                 std = np.std(vec)
#                 lower_bnd = mean - std_multiplier * std
#                 upper_bnd = mean + std_multiplier * std
#             elif lower_bnd is not None and upper_bnd is not None:
#                 pass  # Use provided bounds
#             else:
#                 raise ValueError("Clamping requested but bounds not specified or mode invalid")
#
#             vec = np.clip(vec, lower_bnd, upper_bnd)
#
#     abs_max_vec = np.percentile(np.abs(vec), 99.9)
#     abs_max_grid = np.max(np.abs(grid))
#
#     if abs_max_vec == 0 or abs_max_grid == 0:
#         raise ValueError("Invalid quantization input or grid")
#
#     scale = abs_max_vec / abs_max_grid
#     scaled_vec = vec / scale
#
#     if np.isnan(scaled_vec).any() or np.isinf(scaled_vec).any():
#         raise ValueError("scaled_vec contains NaN or Inf")
#
#     quantized_vec = np.array([grid[np.argmin(np.abs(grid - val))] for val in scaled_vec])
#     return quantized_vec, scale, 0

def quantize(
        vec, grid,
        clamp_outliers=None,  # None / 'percentile' / 'std'
        lower_bnd=None, upper_bnd=None,
        percentile_limits=(1, 99),
        std_multiplier=3,
        asymmetric=False
):
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    if not asymmetric and np.min(grid) >= 0:
        raise ValueError("Symmetric quantization requires signed grid (negative values)")

    # Clamping if requested
    if clamp_outliers is not None:
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

    if asymmetric:
        min_val = np.min(vec)
        max_val = np.max(vec)
        qmin = np.min(grid)
        qmax = np.max(grid)
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            raise ValueError("Scale computed as zero in asymmetric quantization")
        zero_point = round(-min_val / scale) + qmin
        scaled_vec = np.round(vec / scale + zero_point)
    else:
        abs_max_vec = np.percentile(np.abs(vec), 99.9)
        abs_max_grid = np.max(np.abs(grid))
        if abs_max_vec == 0 or abs_max_grid == 0:
            raise ValueError("Invalid quantization input or grid")
        scale = abs_max_vec / abs_max_grid
        scaled_vec = vec / scale
        zero_point = 0

    if np.isnan(scaled_vec).any() or np.isinf(scaled_vec).any():
        raise ValueError("scaled_vec contains NaN or Inf")

    # Find nearest value in grid
    quantized_vec = np.array([grid[np.argmin(np.abs(grid - val))] for val in scaled_vec])

    return quantized_vec, scale, zero_point


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
        # print(f"Bias shape: {layer.bias.shape}, bias: {b_w.shape}")
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


def classification_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return metrics


def quantization_error_metrics(output_fp, output_quant):
    metrics = {}
    metrics['MSE'] = mean_squared_error(output_fp, output_quant)
    metrics['MAE'] = mean_absolute_error(output_fp, output_quant)
    metrics['max_abs_error'] = np.max(np.abs(output_fp - output_quant))
    metrics['confidence_delta'] = np.abs(np.max(output_fp) - np.max(output_quant))
    return metrics


# Plot classification metrics
def plot_classification_metrics(metrics_dict, title="Classification Metrics"):
    labels = list(metrics_dict.keys())
    accuracy = [metrics_dict[fmt]['accuracy'] for fmt in labels]
    precision = [metrics_dict[fmt]['precision'] for fmt in labels]
    recall = [metrics_dict[fmt]['recall'] for fmt in labels]
    f1 = [metrics_dict[fmt]['f1_score'] for fmt in labels]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5 * width, precision, width, label='Precision')
    plt.bar(x + 0.5 * width, recall, width, label='Recall')
    plt.bar(x + 1.5 * width, f1, width, label='F1 Score')

    plt.xlabel('Quantization Format')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, labels, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot quantization error metrics
def plot_quantization_error(metrics_dict, title="Quantization Error Metrics"):
    labels = list(metrics_dict.keys())
    mse = [metrics_dict[fmt]['MSE'] for fmt in labels]
    mae = [metrics_dict[fmt]['MAE'] for fmt in labels]
    max_err = [metrics_dict[fmt]['max_abs_error'] for fmt in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, mse, width, label='MSE')
    plt.bar(x, mae, width, label='MAE')
    plt.bar(x + width, max_err, width, label='Max Abs Error')

    plt.xlabel('Quantization Format')
    plt.ylabel('Error Value')
    plt.title(title)
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_on_image_folder(model, run_quantized_forward, preprocess_fn, grid, image_dir):
    y_true_all = []
    y_pred_all = []

    mse_list = []
    mae_list = []
    max_err_list = []
    delta_list = []

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("⚠ No valid images were processed.")
        return {
            'accuracy': float('nan'),
            'mean_MSE': float('nan'),
            'mean_MAE': float('nan'),
            'mean_max_abs_error': float('nan'),
            'mean_confidence_delta': float('nan')
        }

    for img_path in tqdm(image_paths, desc="Evaluating"):
        image_tensor = preprocess_fn(img_path)
        with torch.no_grad():
            output_fp = model(image_tensor)
            output_q = run_quantized_forward(model, image_tensor, grid, debug=False)

            prob_fp = torch.softmax(output_fp, dim=1)
            prob_q = torch.softmax(output_q, dim=1)

            top1_fp = torch.argmax(prob_fp, dim=1).item()
            top1_q = torch.argmax(prob_q, dim=1).item()

            print("top1_fp:", top1_fp, "top1_q:", top1_q)

            y_true_all.append(top1_fp)
            y_pred_all.append(top1_q)

            mse_list.append(mean_squared_error(prob_fp.numpy(), prob_q.numpy()))
            mae_list.append(mean_absolute_error(prob_fp.numpy(), prob_q.numpy()))
            max_err_list.append(np.max(np.abs(prob_fp.numpy() - prob_q.numpy())))
            delta_list.append(np.abs(np.max(prob_fp.numpy()) - np.max(prob_q.numpy())))

    classification = classification_metrics(y_true_all, y_pred_all)

    result = {
        'accuracy': classification['accuracy'],
        'mean_MSE': np.mean(mse_list),
        'mean_MAE': np.mean(mae_list),
        'mean_max_abs_error': np.mean(max_err_list),
        'mean_confidence_delta': np.mean(delta_list)
    }
    return result


if __name__ == '__main__':
    image_dir = r"../../100 animals"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    configs = [
        {'cntrSize': 8, 'isAsymmetric': False, 'clamp': None, 'name': 'INT8_symmetric_noclamp'},
        {'cntrSize': 8, 'isAsymmetric': False, 'clamp': 'percentile', 'percentile_limits': (1, 99),'name': 'INT8_symmetric_clamp_p1'},
        {'cntrSize': 8, 'isAsymmetric': False, 'clamp': 'percentile', 'percentile_limits': (0.1, 99.9), 'name': 'INT8_symmetric_clamp_p0.1'},
        {'cntrSize': 8, 'isAsymmetric': False, 'clamp': 'std', 'std_multiplier': 3,'name': 'INT8_symmetric_clamp_std3'},
        {'cntrSize': 8, 'isAsymmetric': False, 'clamp': 'std', 'std_multiplier': 4,'name': 'INT8_symmetric_clamp_std4'},


        {'cntrSize': 16, 'isAsymmetric': False, 'clamp': None, 'name': 'INT16_symmetric_noclamp'},
        {'cntrSize': 16, 'isAsymmetric': False, 'clamp': 'percentile', 'percentile_limits': (1, 99),'name': 'INT16_symmetric_clamp_p1'},
        {'cntrSize': 16, 'isAsymmetric': False, 'clamp': 'percentile', 'percentile_limits': (0.1, 99.9),'name': 'INT16_symmetric_clamp_p0.1'},
        {'cntrSize': 16, 'isAsymmetric': False, 'clamp': 'std', 'std_multiplier': 3, 'name': 'INT16_symmetric_clamp_std3'},
        {'cntrSize': 16, 'isAsymmetric': False, 'clamp': 'std', 'std_multiplier': 4,'name': 'INT16_symmetric_clamp_std4'},


        {'cntrSize': 8, 'isAsymmetric': True, 'clamp': None, 'name': 'INT8_asymmetric_noclamp'},
        {'cntrSize': 8, 'isAsymmetric': True, 'clamp': 'percentile', 'percentile_limits': (1, 99),'name': 'INT8_asymmetric_clamp_p1'},
        {'cntrSize': 8, 'isAsymmetric': True, 'clamp': 'percentile', 'percentile_limits': (0.1, 99.9),'name': 'INT8_asymmetric_clamp_p0.1'},
        {'cntrSize': 8, 'isAsymmetric': True, 'clamp': 'std', 'std_multiplier': 3,'name': 'INT8_asymmetric_clamp_std3'},
        {'cntrSize': 8, 'isAsymmetric': True, 'clamp': 'std', 'std_multiplier': 4,'name': 'INT8_asymmetric_clamp_std4'},


        {'cntrSize': 16, 'isAsymmetric': True, 'clamp': None, 'name': 'INT16_asymmetric_noclamp'},
        {'cntrSize': 16, 'isAsymmetric': True, 'clamp': 'percentile', 'percentile_limits': (1, 99),'name': 'INT16_asymmetric_clamp_p1'},
        {'cntrSize': 16, 'isAsymmetric': True, 'clamp': 'percentile', 'percentile_limits': (0.1, 99.9),'name': 'INT16_asymmetric_clamp_p0.1'},
        {'cntrSize': 16, 'isAsymmetric': True, 'clamp': 'std', 'std_multiplier': 3,'name': 'INT16_asymmetric_clamp_std3'},
        {'cntrSize': 16, 'isAsymmetric': True, 'clamp': 'std', 'std_multiplier': 4,'name': 'INT16_asymmetric_clamp_std4'},

    ]
    for cfg in configs:
        if cfg['isAsymmetric']:
            continue

        print("\n==========================")
        print(f"Running for {cfg['name']} (cntrSize={cfg['cntrSize']}, isAsymmetric={cfg['isAsymmetric']})")
        cntrSize = cfg['cntrSize']
        grid = generate_grid(cntrSize, signed=True, type='INT')
        quantize_model_weights(
            model,
            grid,
            debug=True,
            clamp_outliers=cfg['clamp'],  # None / 'percentile' / 'std'
            percentile_limits=cfg.get('percentile_limits'),
            std_multiplier = cfg.get('std_multiplier')
        )
        metrics = evaluate_on_image_folder(model, run_quantized_forward, preprocess_image, grid, image_dir)
        print(f"Results for INT cntrSize={cntrSize}: {metrics}")
