import torch
import torch.nn as nn
import numpy as np
import Quantizer
import quantizationItamar
import matplotlib.pyplot as plt
import pandas as pd


class SimpleModel(nn.Module):
    def __init__(self, output_size=5):  # fixed from _init_
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=output_size)

    def forward(self, x):
        return self.fc(x)


def quantize(vec, grid, clamp_outliers=False, lower_bnd=None, upper_bnd=None, verbose=[]):
    grid = np.array(grid)
    vec = np.array(vec)

    if np.min(grid) >= 0:
        raise ValueError("Grid must be signed (contain negative values)")

    if clamp_outliers:
        if lower_bnd is None or upper_bnd is None:
            raise ValueError("Clamping requested but bounds not specified")
        vec = np.clip(vec, lower_bnd, upper_bnd)

    abs_max_vec = np.max(np.abs(vec))
    abs_max_grid = np.max(np.abs(grid))
    scale = abs_max_vec / abs_max_grid if abs_max_grid != 0 else 1.0
    scale = scale if np.isfinite(scale) and scale != 0 else 1.0

    scaled_vec = vec / scale
    quantized_vec = np.zeros_like(vec)
    for i, val in enumerate(scaled_vec):
        idx = np.argmin(np.abs(grid - val))
        quantized_vec[i] = grid[idx]

    return quantized_vec, scale, 0


def print_stats(original_output, quantized_output, Dquantized_output, error_data):
    diff = np.abs(original_output - Dquantized_output)
    relative_error = np.abs(diff / original_output)
    mean_relative_error = np.mean(relative_error)
    max_relative_error = np.max(relative_error)
    error_data['mean_relative_error'].append(mean_relative_error)
    error_data['max_relative_error'].append(max_relative_error)
    print(f"🔹 Mean relative error: {mean_relative_error}")
    print(f"🔹 Max relative error: {max_relative_error}")


def test_quantization(grid_type="int", cntr_size=14, X_array=None, verbose=[]):
    torch.manual_seed(42)
    np.random.seed(42)
    model = SimpleModel(output_size=15)
    X = torch.tensor(X_array, dtype=torch.float32)
    original_output = np.array(model(X).tolist())
    print(f"Original output: {original_output}")

    if grid_type.startswith("int"):
        grid = quantizationItamar.generate_grid_INT(cntr_size, signed=True)
    elif grid_type.startswith("F2P"):
        grid = Quantizer.getAllValsFxp(fxpSettingStr=grid_type, cntrSize=cntr_size, verbose=[], signed=True)
        print(f"Quantization grid ({grid_type}): {grid[:10]} ... {grid[-10:]}")
    else:
        raise ValueError("Invalid grid_type. Choose 'int' or 'F2P'")

    original_shape = model.fc.weight.data.shape
    weights_np = model.fc.weight.data.numpy().flatten()
    quantized_weights, scale_W, zero_point = quantize(weights_np, grid=grid)
    model.fc.weight.data = torch.tensor(quantized_weights.reshape(original_shape))

    error_data = {"mean_relative_error": [], "max_relative_error": []}
    X_Q, scale_X, zero_point_X = quantize(X.numpy().flatten(), grid=grid)
    X_Q_tensor = torch.tensor(X_Q, dtype=torch.float32)
    quantized_output = np.array(model(X_Q_tensor).tolist())
    Dquantized_output = quantized_output * scale_W * scale_X
    mean_error, max_error = compute_relative_error(original_output, Dquantized_output)
    print_error_stats(mean_error, max_error)

    error_data = {
        "mean_relative_error": mean_error,
        "max_relative_error": max_error
    }

    return error_data


def print_error_stats(mean_error, max_error):
    print(f"🔹 Mean relative error: {mean_error}")
    print(f"🔹 Max relative error: {max_error}")


def compute_relative_error(original_output, quantized_output):
    diff = np.abs(original_output - quantized_output)
    relative_error = np.abs(diff / original_output)
    mean_relative_error = np.mean(relative_error)
    max_relative_error = np.max(relative_error)
    return mean_relative_error, max_relative_error


def visual(X_array,print_stats=True):
    data = {
        "grid_type": [],
        "bit_width": [],
        "mean_relative_error": [],
        "max_relative_error": []
    }

    list_of_grid_types = ["int", "F2P_lr_h2", "F2P_sr_h2", "F2P_si_h2", "F2P_li_h2"]
    list_cntSize = [8, 10, 12, 14, 16]

    for grid_type in list_of_grid_types:
        for cntr_size in list_cntSize:
            print(f"Testing grid: {grid_type}, counter size: {cntr_size}")
            error_data = test_quantization(grid_type=grid_type, cntr_size=cntr_size, X_array=X_array)

            data["grid_type"].append(grid_type)
            data["bit_width"].append(cntr_size)
            data["mean_relative_error"].append(error_data["mean_relative_error"])
            data["max_relative_error"].append(error_data["max_relative_error"])


    df = pd.DataFrame(data)

    if print_stats:
        rows = list(zip(
            data["grid_type"],
            data["bit_width"],
            data["mean_relative_error"],
            data["max_relative_error"]
        ))

        print("\nGrid Type       Bit Width    Mean Rel Error     Max Rel Error")
        print("--------------------------------------------------------------")
        for row in rows:
            print(f"{row[0]:<15} {row[1]:<12} {row[2]:<18.6f} {row[3]:<.6f}")

        # === Graph 1: Line plot for Mean Error
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v']
    linestyles = ['-', '--', '-.', ':', '-']
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i, grid_type in enumerate(df['grid_type'].unique()):
        sub_df = df[df['grid_type'] == grid_type]
        plt.plot(sub_df['bit_width'],
                 sub_df['mean_relative_error'],
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 color=colors[i % len(colors)],
                 linewidth=2,
                 label=grid_type)

    plt.title("Mean Relative Error vs Bit Width")
    plt.xlabel("Bit Width")
    plt.ylabel("Mean Relative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Graph 2: Line plot for Max Error
    plt.figure(figsize=(8, 6))
    for i, grid_type in enumerate(df['grid_type'].unique()):
        sub_df = df[df['grid_type'] == grid_type]
        plt.plot(sub_df['bit_width'],
                 sub_df['max_relative_error'],
                 marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)],
                 color=colors[i % len(colors)],
                 linewidth=2,
                 label=grid_type)

    plt.title("Max Relative Error vs Bit Width")
    plt.xlabel("Bit Width")
    plt.ylabel("Max Relative Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X_array = np.array([0.656565, 2, 1000, -4, -0.5, 6.11, 7, 8, -1000, 0.45])
    visual(X_array)
