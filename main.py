from fastapi.responses import FileResponse
import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# loading the saved weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("vision_resnet18.pt", map_location=device))
model.to(device)
model.eval()

# has to match the resnet transform used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI()



@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()

    return {
        "predicted_class": classes[pred],
        "confidence": round(probs[pred].item(), 3),
    }