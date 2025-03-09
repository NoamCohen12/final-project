import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import Quantizer
import quantizationItamar
import matplotlib.pyplot as plt


# 1. יצירת שכבה מותאמת אישית (Custom Layer)
class QuantizedConvLayer(nn.Module):
    def __init__(self, quantized_weights, original_shape):
        super().__init__()
        # המרת המשקלים המכווצים ל-tensor
        self.weights = torch.from_numpy(quantized_weights).float().view(original_shape)
        # אין Bias בשכבה זו
        self.bias = None

    def forward(self, x):
        # ביצוע קונבולוציה
        return nn.functional.conv2d(x, self.weights, bias=self.bias, stride=1, padding=1)


# 2. טוענים את המודל ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()  # מוד evaluation

# חילוץ משקלים מהשכבה הראשונה של המודל
first_layer_weights = model.conv1.weight.detach().cpu().numpy()
original_shape = first_layer_weights.shape  # שמירת הצורה המקורית של המשקלים
vec = first_layer_weights.flatten()  # הפיכת המשקלים לוקטור

# 3. כימות המשקלים
grid = quantizationItamar.generate_grid(8, False)  # ייצור גריד לכימות
quantized_weights, scale, z = Quantizer.quantize(vec=vec, grid=grid)  # כימות המשקלים

# יצירת שכבת Conv מותאמת אישית
quntizer_layer = QuantizedConvLayer(quantized_weights, original_shape)
# 4. טעינת תמונה ועיבוד מקדים
image_path = "5pics/cat.jpeg"  # נתיב התמונה
image = Image.open(image_path).convert("RGB")

# שינוי גודל ונרמול התמונה
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # שינוי גודל התמונה
    transforms.ToTensor(),  # המרה ל-tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # נרמול
])
input_tensor = transform(image).unsqueeze(0)  # הוספת מימד Batch

# 5. העברת התמונה דרך השכבה הראשונה
with torch.no_grad():
    output_tensor = quntizer_layer(input_tensor)

# 6. המרת התוצאה למערך NumPy להצגה
output_array = output_tensor.squeeze(0).cpu().numpy()  # הסרת מימד ה-Batch
print("Shape of output:", output_array.shape)

# ביצוע הפיכת כימות (Dequantization)
deq_output_array = Quantizer.dequantize(output_array, scale, z)  # הפיכת המערך למקור

# כתיבת כותרת ומידע לקובץ
with open("output_array.txt", "a") as f:
    f.write("Output after quantize:\n")  # כותב את הכותרת לכימות
    np.savetxt(f, output_array.reshape(output_array.shape[0], -1), fmt='%.6f')  # שמירת הערכים המכווצים

with open("output_array11.txt", "a") as g:
    g.write("\nDequantized Output:\n")  # כותב כותרת ל-Dequantized
    np.savetxt(g, deq_output_array.reshape(deq_output_array.shape[0], -1), fmt='%.6f')  # שמירת הערכים המקוריים
print("Quantized and dequantized arrays saved to output_array.txt")


# נירמול התוצאה לתצוגה
output_array -= output_array.min()
output_array /= output_array.max()
output_array *= 255
output_array = output_array.astype(np.uint8)

# 7. הצגת התוצאות
num_filters = output_array.shape[0]
fig, axes = plt.subplots(1, min(5, num_filters), figsize=(15, 15))  # עד 5 פילטרים להצגה
for i in range(min(5, num_filters)):
    ax = axes[i] if num_filters > 1 else axes
    ax.imshow(output_array[i], cmap="gray")
    ax.axis("off")

plt.show()

original_layer = model.conv1
original_output = original_layer(input_tensor)
# כתיבת כותרת ומידע לקובץ
with open("output_array22.txt", "a") as h:
    h.write("Output original:\n")  # כותב את הכותרת לכימות
    np.savetxt(h, original_output.detach().cpu().reshape(original_output.shape[0], -1).numpy(), fmt='%.6f')

