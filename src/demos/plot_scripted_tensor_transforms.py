from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as v1
from torchvision.io import decode_image
from torchvision.models import resnet18, ResNet18_Weights

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

BASE = Path(__file__).resolve().parent.parent  # goes from demos → src → project root
ASSETS_PATH = BASE / "data" / "demo_images"

animal1 = decode_image(str(ASSETS_PATH / 'dog.jpg'))
animal2 = decode_image(str(ASSETS_PATH / 'lion.jpeg'))

resize_transform = v1.Resize((224, 224))
animal1 = resize_transform(animal1)
animal2 = resize_transform(animal2)

transforms = torch.nn.Sequential(
    v1.RandomCrop(224),
    v1.RandomHorizontalFlip(p=0.3),
)

scripted_transforms = torch.jit.script(transforms)


def plot(images):
    fig, axs = plt.subplots(1, len(images), figsize=(12, 4))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0)  # העבר לממדים (H, W, C) להצגה
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.show()


try:
    plot([animal1, scripted_transforms(animal1), animal2, scripted_transforms(animal2)])
except Exception as e:
    print("Error while plotting images:", e)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms(antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            probabilities = F.softmax(y_pred, dim=1)  # מחשבים הסתברויות
            return probabilities


device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = Predictor().to(device)
scripted_predictor = torch.jit.script(predictor).to(device)

batch = torch.stack([animal1, animal2]).to(device)
res = predictor(batch)
res_scripted = scripted_predictor(batch)

with open(ASSETS_PATH / 'imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)

for i, (probs, probs_scripted) in enumerate(zip(res, res_scripted)):
    pred_idx = probs.argmax().item()
    pred_prob = probs[pred_idx].item() * 100
    assert pred_idx == probs_scripted.argmax().item()
    print(f"Prediction for Image {i + 1}: {labels[str(pred_idx)]} ({pred_prob:.2f}%)")

save_path = "scripted_model.pt"
scripted_predictor.save(save_path)

dumped_scripted_predictor = torch.jit.load(save_path)
res_scripted_dumped = dumped_scripted_predictor(batch)

assert (res_scripted_dumped.argmax(dim=1) == res_scripted.argmax(dim=1)).all()
