
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

def main():
    # 1. וקטור רנדומלי עם 100 ערכים בטווח -1000 עד 1000
    vec_random = np.random.randint(-1000, 1001, size=100)

    # 2. וקטור שבו רוב הערכים קרובים למינימום והמקסימום
    vec_extreme = np.concatenate([np.random.randint(-1000, -500, size=45),
                                  np.random.randint(500, 1001, size=45),
                                  np.random.randint(-10, 10, size=10)])

    # 3. וקטור שבו רוב הערכים קרובים ל-0
    vec_center = np.concatenate([np.random.randint(-10, 10, size=90),
                                 np.random.randint(-1000, -500, size=5),
                                 np.random.randint(500, 1001, size=5)])

    # יצירת גריד ל-int8 (גריד בין -128 ל-127)
    grid_int8 = np.linspace(-128, 127, 256)  # גריד עם 256 ערכים בין -128 ל-127

    # יצירת גריד ל-int16 (גריד בין -32768 ל-32767)
    grid_int16 = np.linspace(-32768, 32767, 65536)  # גריד עם 65536 ערכים בין -32768 ל-32767

    # קריאה לפונקציית quantize על כל אחד מהווקטורים
    for vec, description in [(vec_random, "Random Vector"),
                             (vec_extreme, "Extreme Values Vector"),
                             (vec_center, "Centered Vector")]:
        # הדפסת הווקטור המקורי ממוין
        print(f"\nOriginal {description} (sorted):")
        print(np.sort(vec))  # הדפס את הווקטור המקורי ממוין

        # הדפסה עם גריד של int8
        print(f"\n{description} after quantization with int8 grid:")
        quantized_vec, scale, zero_point = quantize(vec, grid_int8)

        # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
        dequantized_vec = quantized_vec * scale

        # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
        print(f"Dequantized {description} with int8 grid (sorted):")
        print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

        # הדפסת המידע הנוסף
        print(f"Scale: {scale}")
        print(f"Zero-point: {zero_point}")

        # הדפסה עם גריד של int16
        print(f"\n{description} after quantization with int16 grid:")
        quantized_vec, scale, zero_point = quantize(vec, grid_int16)

        # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
        dequantized_vec = quantized_vec * scale

        # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
        print(f"Dequantized {description} with int16 grid (sorted):")
        print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

        # הדפסת המידע הנוסף
        print(f"Scale: {scale}")
        print(f"Zero-point: {zero_point}")

    # עבור הווקטור בו הערכים קרובים ל-0, נוסיף את האפשרות של חיתוך (clamping)
    vec_center_clamped = np.clip(vec_center, -10, 10)  # חיתוך בטווח של -10 עד 10

    # הדפסת הווקטור המקורי אחרי חיתוך
    print(f"\nOriginal Centered Vector after Clamping (sorted):")
    print(np.sort(vec_center_clamped))  # הדפס את הווקטור אחרי חיתוך

    # הדפסה עם גריד של int8 אחרי חיתוך
    print(f"\nCentered Vector after quantization with int8 grid and clamping:")
    quantized_vec, scale, zero_point = quantize(vec_center_clamped, grid_int8, clamp_outliers=True, lower_bnd=-10, upper_bnd=10)

    # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
    dequantized_vec = quantized_vec * scale

    # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
    print(f"Dequantized Centered Vector with int8 grid and clamping (sorted):")
    print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

    # הדפסת המידע הנוסף
    print(f"Scale: {scale}")
    print(f"Zero-point: {zero_point}")

    # הדפסה עם גריד של int16 אחרי חיתוך
    print(f"\nCentered Vector after quantization with int16 grid and clamping:")
    quantized_vec, scale, zero_point = quantize(vec_center_clamped, grid_int16, clamp_outliers=True, lower_bnd=-10, upper_bnd=10)

    # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
    dequantized_vec = quantized_vec * scale

    # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
    print(f"Dequantized Centered Vector with int16 grid and clamping (sorted):")
    print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

    # הדפסת המידע הנוסף
    print(f"Scale: {scale}")
    print(f"Zero-point: {zero_point}")


# קריאה לפונקציית main
if __name__ == "__main__":
    main()
'''
without clamp
# פונקציית main להדגמה עם וקטור רנדומלי גדול יותר
def main():
    # 1. וקטור רנדומלי עם 100 ערכים בטווח -1000 עד 1000
    vec_random = np.random.randint(-1000, 1001, size=100)

    # 2. וקטור שבו רוב הערכים קרובים למינימום והמקסימום
    vec_extreme = np.concatenate([np.random.randint(-1000, -500, size=45),
                                  np.random.randint(500, 1001, size=45),
                                  np.random.randint(-10, 10, size=10)])

    # 3. וקטור שבו רוב הערכים קרובים ל-0
    vec_center = np.concatenate([np.random.randint(-10, 10, size=90),
                                 np.random.randint(-1000, -500, size=5),
                                 np.random.randint(500, 1001, size=5)])

    # יצירת גריד ל-int8 (גריד בין -128 ל-127)
    grid_int8 = np.linspace(-128, 127, 256)  # גריד עם 256 ערכים בין -128 ל-127

    # יצירת גריד ל-int16 (גריד בין -32768 ל-32767)
    grid_int16 = np.linspace(-32768, 32767, 65536)  # גריד עם 65536 ערכים בין -32768 ל-32767

    # קריאה לפונקציית quantize על כל אחד מהווקטורים
    for vec, description in [(vec_random, "Random Vector"),
                             (vec_extreme, "Extreme Values Vector"),
                             (vec_center, "Centered Vector")]:
        # הדפסת הווקטור המקורי ממוין
        print(f"\nOriginal {description} (sorted):")
        print(np.sort(vec))  # הדפס את הווקטור המקורי ממוין

        # הדפסה עם גריד של int8
        print(f"\n{description} after quantization with int8 grid:")
        quantized_vec, scale, zero_point = quantize(vec, grid_int8)

        # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
        dequantized_vec = quantized_vec * scale

        # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
        print(f"Dequantized {description} with int8 grid (sorted):")
        print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

        # הדפסת המידע הנוסף
        print(f"Scale: {scale}")
        print(f"Zero-point: {zero_point}")

        # הדפסה עם גריד של int16
        print(f"\n{description} after quantization with int16 grid:")
        quantized_vec, scale, zero_point = quantize(vec, grid_int16)

        # דה-קוונטיזציה: החזרת הווקטור לממד המקורי על ידי הכפלה ב-scale
        dequantized_vec = quantized_vec * scale

        # הדפסת הווקטור אחרי דה-קוונטיזציה וממוין
        print(f"Dequantized {description} with int16 grid (sorted):")
        print(np.sort(dequantized_vec))  # הדפס את הווקטור אחרי דה-קוונטיזציה וממוין

        # הדפסת המידע הנוסף
        print(f"Scale: {scale}")
        print(f"Zero-point: {zero_point}")


# קריאה לפונקציית main
if __name__ == "__main__":
    main()


'''
