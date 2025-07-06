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
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=1)  # שכבה Fully Connected

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    # 1. Create model, input, calculate original output (keep this part the same)
    model = SimpleModel()
    X = torch.randn(1, 10)
    original_output = model(X).item()

    # 2. Generate quantization grid
    grid = quantizationItamar.generate_grid_INT(8, signed=True)

    # 3. Get original shape BEFORE flattening (MOVE THIS LINE UP!)
    original_shape = model.fc.weight.data.shape

    # 4. Get weights and flatten them
    weights_np = model.fc.weight.data.numpy().flatten()

    # 5. Run tests and validation checks (keep this part the same)
    Quantizer.testQuantOfSingleVec(vec2quantize=weights_np, grid=grid, verbose=[2])
    print("-------------------------------------------------------------------")
    assert not np.isnan(weights_np).any(), "Error: Found NaN in model weights!"
    assert not np.isinf(weights_np).any(), "Error: Found Inf in model weights!"
    assert len(grid) > 1, "Error: Quantization grid is empty or too small!"
    print(f"Original Weights:\n{np.sort(weights_np)}")
    print(f"Quantization Grid - Max: {np.max(grid)}, Min: {np.min(grid)}")

    # 6. Perform quantization
    quantized_weights, scale_factor, zero_point = quantize(weights_np, grid=grid)

    # 8. Validation checks
    assert np.all(quantized_weights >= np.min(grid)), "Error: Some quantized weights are below the allowed grid range!"
    assert np.all(quantized_weights <= np.max(grid)), "Error: Some quantized weights exceed the allowed grid range!"
    print(f"Quantized Weights:\n{np.sort(quantized_weights)}")

    # 10. Reshape and assign back to model
    model.fc.weight.data = torch.tensor(quantized_weights.reshape(original_shape))
    X_np = X.numpy().flatten()  # Flatten the input tensor

    X_Q, scale_factor_X, zero_point_X = quantize(X_np, grid=grid)
    X_Q_tensor = torch.tensor(X_Q).reshape(X.shape)  # Reshape back to original input shape

    # 11. Test and validate (keep this part the same)
    quantized_output = model(X_Q_tensor).item()
    print(f"Original Output: {original_output}")
    print(f"Quantized Output: {quantized_output}")
    Dquantized_output =  quantized_output * scale_factor * scale_factor_X
    print(f"Dequantized Output: {Dquantized_output}")

    # 12. Possibly relax this assertion or add a relative error check
    assert abs(original_output - Dquantized_output) < abs(
        original_output) * 0.5, "Error: Quantized output is too far from original output!"
    print("All tests passed successfully! ✅")
