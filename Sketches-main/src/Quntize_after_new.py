import torch
import torch.nn as nn
import numpy as np
import Quantizer
import quantizationItamar


# יצירת המודל
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(in_features=10, out_features=1)  # שכבה Fully Connected

    def forward(self, x):
        return self.fc(x)


# יצירת אובייקט מודל
model = SimpleModel()

# יצירת קלט אקראי
X = torch.randn(1, 10)

# חישוב הפלט מהמודל המקורי
original_output = model(X).item()

# קבלת המשקלים של השכבה כ-numpy array
weights_np = model.fc.weight.data.numpy().flatten()

# בדיקות תקינות למשקלים לפני קוונטיזציה
assert not np.isnan(weights_np).any(), "Error: Found NaN in model weights!"
assert not np.isinf(weights_np).any(), "Error: Found Inf in model weights!"

# יצירת רשת קוונטיזציה (גריד) עם 256 ערכים (לדוגמה, 8 ביט)
grid = quantizationItamar.generate_grid(16, signed=False)

# בדיקה שהגריד לא ריק
assert len(grid) > 1, "Error: Quantization grid is empty or too small!"

# הדפסות לבדיקת ערכים
print(f"Original Weights:\n{np.sort(weights_np)}")  # השתמש ב- numpy.sort()
print(f"Quantization Grid - Max: {np.max(grid)}, Min: {np.min(grid)}")

# ביצוע קוונטיזציה למשקלים
try:
    quantized_weights, scale_factor, zero_point = Quantizer.quantize(weights_np, grid=grid)
except Exception as e:
    print(f"Error in quantization: {e}")
    exit()

print(f"Scale Factor: {scale_factor}")
print(f"Max weights_np: {np.max(weights_np)}, Min weights_np: {np.min(weights_np)}")
print(f"quantized_weights:{quantized_weights}")

# בדיקות תקינות אחרי קוונטיזציה
assert np.all(quantized_weights >= np.min(grid)), "Error: Some quantized weights are below the allowed grid range!"
assert np.all(quantized_weights <= np.max(grid)), "Error: Some quantized weights exceed the allowed grid range!"

# הדפסה לבדיקה
print(f"Quantized Weights:\n{np.sort(quantized_weights)}")  # השתמש ב- numpy.sort()
print(f"Scale Factor: {scale_factor}")

# עדכון המשקלים במודל לאחר קוונטיזציה
model.fc.weight.data = torch.tensor(quantized_weights, dtype=torch.float32)

# חישוב הפלט מהמודל לאחר קוונטיזציה
quantized_output = model(X).item()

# בדיקה שהתוצאה אחרי קוונטיזציה לא רחוקה מדי מהתוצאה המקורית
assert abs(original_output - quantized_output) < abs(
    original_output) * 0.5, "Error: Quantized output is too far from original output!"

# הצגת התוצאות
print(f"Original Output: {original_output}")
print(f"Quantized Output: {quantized_output}")

print("All tests passed successfully! ✅")
