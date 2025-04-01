import torch
import torch.nn as nn
import numpy as np
import Quantizer
import quantizationItamar


def quantize(
        vec: np.array,  # Vector to quantize
        grid: np.array,  # Quantization grid (signed)
        clamp_outliers: bool = False,  # Clamp outliers flag
        lower_bnd: float = None,  # Lower bound for clamping
        upper_bnd: float = None,  # Upper bound for clamping
        verbose: list = []  # Verbose output
) -> tuple:  # Returns (quantized_vector, scale)
    """
    Quantize an input vector using symmetric min-max quantization with a signed grid.

    Steps:
    1. Optional clamping if requested
    2. Calculate scale factor based on input range and grid range
    3. Scale the vector
    4. Find nearest grid values for each element
    5. Return quantized vector and scale factor
    """
    grid = np.array(grid)
    vec = np.array(vec)

    # Verify we have a signed grid
    if np.min(grid) >= 0:
        raise ValueError("Grid must be signed (contain negative values)")

    # Optional clamping
    if clamp_outliers:
        if lower_bnd is None or upper_bnd is None:
            raise ValueError("Clamping requested but bounds not specified")
        vec = np.clip(vec, lower_bnd, upper_bnd)

    # For symmetric quantization with signed grid, use the max absolute value
    abs_max_vec = np.max(np.abs(vec))
    abs_max_grid = np.max(np.abs(grid))

    try:
        # Symmetric scaling factor
        scale = abs_max_vec / abs_max_grid
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
    except Exception as e:
        raise ValueError(f"Error calculating scale: {e}")

    # Scale the vector (symmetric around zero)
    scaled_vec = vec / scale

    # For integer grids, we could just cast to int, but better to find nearest
    quantized_vec = np.zeros_like(vec)
    for i, val in enumerate(scaled_vec):
        # Find index of nearest grid value
        idx = np.argmin(np.abs(grid - val))
        quantized_vec[i] = grid[idx]

    return quantized_vec, scale, 0  # Zero-point is always 0 for symmetric quantization


# יצירת המודל
class SimpleModel(nn.Module):
    def __init__(self, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=output_size)  # שכבה Fully Connected

    def forward(self, x):
        return self.fc(x)


def print_stats(original_output, quantized_output, Dquantized_output):
    print(f"Original Output: {np.sort(original_output)}")
    print(f"Quantized Output: {np.sort(quantized_output)}")
    print(f"Dequantized Output: {np.sort(Dquantized_output)}")

    # Absolute difference for each element
    diff = np.abs(original_output - Dquantized_output)

    # 1. Maximum difference
    max_diff = np.max(diff)

    # 2. Minimum difference
    min_diff = np.min(diff)

    # 3. Relative error for each element
    relative_error = np.abs(diff / original_output)  # Computing relative error

    # 4. Mean absolute error
    mean_error = np.mean(diff)

    # Display results
    print(f"🔹 Maximum difference: {max_diff}")
    print(f"🔹 Minimum difference: {min_diff}")
    print(f"🔹 Relative error (for each element): {relative_error}")
    print(f"🔹 Mean relative error: {np.mean(relative_error)}")
    print(f"🔹 Mean absolute error: {mean_error}")


def test_quantization(grid_type="int", cntr_size=14, X_array=None, verbose=[]):
    torch.manual_seed(42)
    np.random.seed(42)
    model = SimpleModel(output_size=15)
    X = torch.tensor(X_array, dtype=torch.float32)

    original_output = np.array(model(X).tolist())
    print("original_output:", np.sort(original_output))

    if grid_type.startswith("int"):
        grid = quantizationItamar.generate_grid(cntr_size, signed=True)
    elif grid_type.startswith("F2P"):
        grid = Quantizer.getAllValsFxp(fxpSettingStr=grid_type, cntrSize=cntr_size, verbose=[], signed=True)
    else:
        raise ValueError("Invalid grid_type. Choose 'int' or 'F2P'")

    original_shape = model.fc.weight.data.shape
    weights_np = model.fc.weight.data.numpy().flatten()

    if 0 in verbose:
        print(f"Original Weights: {weights_np[:10]}........{weights_np[-10:]}")
        Quantizer.testQuantOfSingleVec(vec2quantize=weights_np, grid=grid, verbose=[2])

    quantized_weights, scale_factor, zero_point = quantize(weights_np, grid=grid)

    model.fc.weight.data = torch.tensor(quantized_weights.reshape(original_shape))

    if 3 in verbose:
        lower_bnd, upper_bnd = np.percentile(X, 1), np.percentile(X, 99)
        X_Q, scale_factor_X, zero_point_X = quantize(X, grid, clamp_outliers=True,
                                                               lower_bnd=lower_bnd, upper_bnd=upper_bnd)
    else:
        X_Q, scale_factor_X, zero_point_X = quantize(X.numpy().flatten(), grid=grid)

    X_Q_tensor = torch.tensor(X_Q).reshape(X.shape)

    quantized_output = np.array(model(X_Q_tensor).tolist())
    Dquantized_output = quantized_output *scale_factor * scale_factor_X

    print_stats(original_output, quantized_output, Dquantized_output)


if __name__ == '__main__':
    X_array = np.array([0.656565, 2, 1000, -4, -0.5, 6.11, 7, 8, -1000, 0.45])

    list_of_grid_types = ["int", "F2P_lr_h2", "F2P_sr_h2", "F2P_si_h2", "F2P_li_h2"]
    list_cntSize = [8, 10, 12, 14, 16]
    list_verbose = [0, 3]
    # Example usage:
    for grid_type in list_of_grid_types:
        for cntr_size in list_cntSize:
            for verbose in list_verbose:
                print(
                    f"Testing grid: {grid_type}, counter size: {cntr_size}, verbose: {'clamp' if verbose == 3 else 'without clamp'}")
                test_quantization(grid_type=grid_type, cntr_size=cntr_size, X_array=X_array, verbose=[0,verbose])
                print("--------------------------------------------------")
