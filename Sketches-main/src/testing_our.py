import numpy as np


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


#kind of grids to check
#a lot of numbers in the edge and few in the middle
#a lot of numbers in the midlle and few in the edge also check the clamp
# and compare to int16
#
#