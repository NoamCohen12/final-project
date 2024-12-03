import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from resnet10 import ResNet10  # ייבוא המודל מהקובץ resnet10.py

# בחר את המכשיר (GPU אם קיים, אחרת CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.device("cuda")
    print("use gpu")
else:
    print("use cpu")

# יצירת המודל
model = ResNet10(num_classes=10).to(device)

# ---- שלב 1: הגדרת פעולות עיבוד לתמונות ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # שינוי גודל התמונות ל-224x224
    transforms.ToTensor(),          # המרת התמונות לטנסור
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # נורמליזציה לערכים סטנדרטיים
])

# ---- שלב 2: הורדה והגדרת מאגר הנתונים ----
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# ---- שלב 3: הגדרת DataLoader לאימון ולבדיקה ----
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---- שלב 4: בדיקת מספר דוגמאות במאגר ----
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images: {len(test_dataset)}")

# ---- שלב 5: הצגת תמונה לדוגמה ----
image, label = train_dataset[553]  # תמונה לדוגמה מהאינדקס 553
classes = train_dataset.classes    # שמות הקטגוריות

print(f"Label: {label} ({classes[label]})")  # הצגת התיוג של התמונה
plt.imshow(image.permute(1, 2, 0))  # שינוי סדר הצירים לפורמט מתאים להצגה
plt.title(f"Class: {classes[label]}")
plt.show()

# ---- שלב 6: יצירת המודל ResNet10 ----
model = ResNet10(num_classes=10).to(device)  # יצירת מודל עם 10 קטגוריות (של CIFAR-10)

# ---- שלב 7: בדיקת המודל עם תמונה לדוגמה ----
model.eval()  # מצב הערכה (ללא אימון)
image, label = test_dataset[503]  # טעינת תמונה לדוגמה ממאגר הבדיקה

# הוספת ממד batch לתמונה כדי להתאים אותה לקלט המודל
image = image.unsqueeze(0).to(device)  # התמונה כעת בגודל [1, 3, 224, 224]

# העברת התמונה דרך המודל
output = model(image)  # הפעלת המודל
_, predicted = torch.max(output, 1)  # חיזוי הקטגוריה בעלת ההסתברות הגבוהה ביותר

# הדפסת התוצאה
print(f"True Label: {label} ({classes[label]})")
print(f"Predicted Class: {predicted.item()} ({classes[predicted.item()]})")

# ---- שלב 8: שמירת המשקלות של המודל ----
torch.save(model.state_dict(), "resnet10_cifar10.pth")  # שמירת משקלות המודל לקובץ
print("Model weights saved!")

# ---- שלב 9: חישוב דיוק המודל ----
correct = 0
total = 0

with torch.no_grad():  # לא לחשב גרדיאנט במהלך הבדיקה
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # שלח את התמונות והתיוגים למכשיר
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
