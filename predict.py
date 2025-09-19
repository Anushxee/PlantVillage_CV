import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os

#---------- Load metadata ----------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("remedies.json", "r") as f:
    remedies = json.load(f)

#---------- Define model architecture ----------
num_classes = len(class_names)   # should be 12
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

#---------- Load trained weights ----------
model.load_state_dict(torch.load("model/model_12class.pth", map_location="cpu"))
model.eval()

#---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(img_path):
    """Predict disease + remedy for a single image"""
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        class_name = class_names[str(pred.item())]
        remedy = remedies.get(class_name, "No remedy found")

    return class_name, remedy

#---------- Predict for all images in dataset folder ----------
dataset_folder = "dataset"

for root, _, files in os.walk(dataset_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            disease, remedy = predict_image(img_path)
            print(f"{file} → Disease: {disease} | Remedy: {remedy}")
