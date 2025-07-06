import numpy as np
import quantizationItamar
import Quantizer
import testing_our


# we create vector of vectors for testing the quantisation

# 10 simple vectors
# 10 numbers from 1 to 10
vec1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 100 numbers from 1 to 100
vec2 = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
     61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
     90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
# 1000 numbers from 1 to 1000
vec3 = np.array(range(1, 1001))

# random positive numbers from 1 to 1000
vec4 = np.random.randint(1, 1001, size=1000)

# random positive numbers from 1 to 10000
vec5 = np.random.randint(1, 1001, size=10000)

# random positive numbers from 1 to 100,000
vec6 = np.random.randint(1, 1001, size=100000)

# random positive numbers from 1 to 1,000,000
vec7 = np.random.randint(1, 1001, size=1000000)

# positive numbers from 1 to 1,000,000
vec8 = np.array(range(1, 1000000))

# vector with 100 numbers: 30 from 1 to 5, 30 from 10 to 60, and 40 from 80 to 100
vec9 = np.concatenate([
    np.random.randint(0, 1, size=30),
    np.random.randint(10, 61, size=40),
    np.random.randint(100, 101, size=30)
])

# vector with 1000 numbers: 10 '1', 980 from 100 to 800, and 10 '1000'

vec10 = np.concatenate([
    np.random.randint(0, 1, size=10),
    np.random.randint(100, 801, size=980),
    np.random.randint(1000, 1001, size=10)
])

# vector with 1000 numbers
vec11 = np.concatenate([
    np.random.randint(0, 20, size=40),
    np.random.randint(100, 150, size=920),
    np.random.randint(236, 257, size=40)
])

# vectors with negative numbers
# 256 negative vectors from -256 to -1
vec12 = np.array(range(-256, 0))

# 1000 negative vectors from -256 to -1
vec13 = np.array(range(-1000, 0))

vec14 = np.random.randint(-10000, 0, size=1000)

# vectors with negative and positive numbers
# 256 numbers from -127 to 128
vec15 = np.array(range(-127, 129))

# 256 numbers from -127 to 128: 10 '-127' , 236 from -100 to 100, 10 '128'
vec16 = np.concatenate([
    np.random.randint(-127, -126, size=10),
    np.random.randint(-70, 70, size=236),
    np.random.randint(128, 129, size=10)
])

# 1000 random numbers from -1000 to 1000
vec17 = np.random.randint(-127, 129, size=1000)

# 1000 random numbers from -1000 to 1000
vec18 = np.random.randint(-10000, 10000, size=1000)

# numbers with floating point

# 100 numbers from 0.1 to 1.0
vec19 = np.linspace(0.1, 1.0, num=100)

# 1000 random numbers from -127 to 128
vec20 = np.random.uniform(-127, 128, size=1000)

# 1000 random numbers from -127 to 128
vec21 = np.random.uniform(-127, 128, size=1000)

# 100 random numbers from -10000 to 10000
vec22 = np.random.uniform(-10000, 10000, size=100)

# 100 random numbers from -10000 to 10000
vec23 = np.random.uniform(-10, 10, size=100)

# 256 floating numbers from -127 to 128: 10 '-127' , 236 from -100 to 100, 10 '128'
vec24 = np.concatenate([
    np.random.uniform(-127, -126, size=10),
    np.random.uniform(-70, 70, size=236),
    np.random.uniform(128, 129, size=10)
])

# 1000 floating numbers from -127 to 128
vec25 = np.concatenate([
    np.random.uniform(-127, -122, size=200),
    np.random.uniform(-80, 80, size=600),
    np.random.uniform(124, 129, size=200)
])

# 100 floating numbers from -127 to 128
vec26 = np.concatenate([
    np.random.uniform(-127, -127, size=20),
    np.random.uniform(-80, 80, size=60),
    np.random.uniform(128, 128, size=20)
])

# 100 floating numbers from -127 to 128
vec27 = np.concatenate([
    np.random.uniform(-127, -80, size=50),
    np.random.uniform(0, 0, size=3),
    np.random.uniform(80, 128, size=47)
])

# 100,000 floating numbers from −32,768 to 32,767
vec28 = np.concatenate([
    np.random.uniform(-32768, -32000, size=50000),
    np.random.uniform(0, 0, size=3),
    np.random.uniform(32000, 32767, size=49997)
])

# 1000 floating numbers from −32,768 to 32,767
vec29 = np.concatenate([
    np.random.uniform(-32768, -32000, size=450),
    np.random.uniform(0, 100, size=100),
    np.random.uniform(32000, 32767, size=450)
])

# שמירת כל הווקטורים במילון
vectors = {
    "vec1": vec1, "vec2": vec2, "vec3": vec3, "vec4": vec4, "vec5": vec5, "vec6": vec6, "vec7": vec7, "vec8": vec8,
    "vec9": vec9, "vec10": vec10, "vec11": vec11, "vec12": vec12, "vec13": vec13, "vec14": vec14, "vec15": vec15,
    "vec16": vec16,
    "vec17": vec17, "vec18": vec18, "vec19": vec19, "vec20": vec20, "vec21": vec21, "vec22": vec22, "vec23": vec23,
    "vec24": vec24,
    "vec25": vec25, "vec26": vec26, "vec27": vec27, "vec28": vec28, "vec29": vec29
}




def print_vector(name, vec):
    """ פונקציה להדפסת וקטור - מדפיסה הכל אם קטן מ-100, אחרת 10 ראשונים ואחרונים """
    vec = np.sort(vec)
    if len(vec) > 100:
        print(f"{name}: {vec[:10]} ... {vec[-10:]}")
    else:
        print(f"{name}: {vec}")


def test(func, vec2quantize, grid,vec_name):
    print(f"Testing {vec_name}")
    print_vector("Vector original", vec2quantize)

    print(f"Grid: {grid[:10]} ... {grid[-10:]}")
    print(f"Grid : max: {np.max(grid)}, min: {np.min(grid)}")

    quantized_vec, scale, zero_point = func(vec2quantize, grid)

    print_vector("Quantized vector", quantized_vec)

    print(f"Scale: {scale}")
    print(f"Zero-point: {zero_point}")

    dequantized_vec = quantized_vec * scale

    print_vector("Dequantized vector", dequantized_vec)

    # Calculate average absolute error between original and dequantized vector
    avg_abs_error = np.mean(np.abs(vec2quantize - dequantized_vec))

    # Calculate average relative error
    #for avoiding division by zero
    safe_vec2quantize = np.where(vec2quantize == 0, 1e-10, vec2quantize)

    avg_rel_error = np.mean(np.abs((vec2quantize - dequantized_vec) / safe_vec2quantize)) * 100
    print(f"Average Absolute Error: {avg_abs_error}")
    print(f"Average Relative Error (%): {avg_rel_error}")

    print("---------------------------------------------------------------------------")


if __name__ == '__main__':
    func1 = Quantizer.quantize
    func2 = testing_our.quantize
    grid1 = quantizationItamar.generate_grid_INT(8, True)
    flavor ="F2P_li_h2"
    grid2 = Quantizer.getAllValsFxp(
        fxpSettingStr=flavor,
        cntrSize=8,
        verbose=[],
        signed=True
    )
    for vec_name, vec_data in vectors.items():
        test(func1, vec_data, grid2, vec_name)
