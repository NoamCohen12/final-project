import numpy as np
import matplotlib.pyplot as plt


#we create vector of vectors for testing the quantisation

# 10 simple vectors
#10 numbers from 1 to 10
vec1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#100 numbers from 1 to 100
vec2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
#1000 numbers from 1 to 1000
vec3 = np.array(range(1, 1001))

#random positive numbers from 1 to 1000
vec4 = np.random.randint(1, 1001, size=1000)

#random positive numbers from 1 to 10000
vec5 = np.random.randint(1, 1001, size=10000)

#random positive numbers from 1 to 100,000
vec6 = np.random.randint(1, 1001, size=100000)

#random positive numbers from 1 to 1,000,000
vec7 = np.random.randint(1, 1001, size=1000000)

#positive numbers from 1 to 1,000,000
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


#vectors with negative and positive numbers
# 256 numbers from -127 to 128
vec15 = np.array(range(-127, 129))

# 256 numbers from -127 to 128: 10 '-127' , 236 from -100 to 100, 10 '128'
vec16 = np.concatenate([
    np.random.randint(-127, -126, size=10),
    np.random.randint(-70, 70, size=236),
    np.random.randint(128, 129, size=10)
])

#1000 random numbers from -1000 to 1000
vec17 = np.random.randint(-127, 129, size=1000)

#1000 random numbers from -1000 to 1000
vec18 = np.random.randint(-10000, 10000, size=1000)



# numbers with floating point

#100 numbers from 0.1 to 1.0
vec19 = np.linspace(0.1, 1.0, num=100)

#1000 random numbers from -127 to 128
vec20 = np.random.uniform(-127, 128, size=1000)

#1000 random numbers from -127 to 128
vec21 = np.random.uniform(-127, 128, size=1000)

#100 random numbers from -10000 to 10000
vec22 = np.random.uniform(-10000, 10000, size=100)

#100 random numbers from -10000 to 10000
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



# יצירת גרף לכל וקטור בנפרד
for name, vec in vectors.items():
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(vec)), vec, alpha=0.5, color='b', s=5)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Scatter Plot of {name}")

    # חישוב נתונים סטטיסטיים
    min_val, max_val = np.min(vec), np.max(vec)
    mean_val = np.mean(vec)
    median_val = np.median(vec)
    std_val = np.std(vec)
    count = len(vec)

    # הוספת טקסט לגרף
    stats_text = f"Count: {count}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
    plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

    # הגדרת גריד דינמי
    if min_val >= 0 and max_val <= 256:
        plt.yticks(np.arange(0, 257, step=16))  # רווחים של 16 בין כל קו
    elif min_val >= -127 and max_val <= 128:
        plt.yticks(np.arange(-127, 129, step=16))  # רווחים של 16 בין כל קו
    elif min_val >= -32768 and max_val <= 32767:
        plt.yticks(np.arange(min_val, max_val, step=5000))  # רווחים של 5000 בין כל קו
    else:
        plt.yticks(np.linspace(min_val, max_val, num=10))  # 10 קווים אחידים בין המינימום למקסימום

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()