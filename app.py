from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io
import json

#--------------------------
#Load remedies.json
#--------------------------
with open("remedies.json", "r") as f:
    remedies = json.load(f)

#Load class names (saved during training)
with open("class_names.json", "r") as f:
    class_names = json.load(f)
    idx_to_class = {v: k for k, v in class_names.items()}  # reverse mapping

#--------------------------
#FastAPI app setup
#--------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------------------------
#Load trained model
#--------------------------
num_classes = len(class_names)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model/model_12class.pth", map_location="cpu"))
model.eval()

#--------------------------
#Image preprocessing
#--------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "SIH CV API is running ✅"}

#--------------------------
#Prediction endpoint
#--------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        #Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        #Run model
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        #Map prediction
        class_name = idx_to_class[class_idx]
        remedy_info = remedies.get(class_name, "No remedy found.")

        return JSONResponse(content={
            "predicted_class": class_name,
            "remedy": remedy_info
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
