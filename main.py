from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import joblib
from PIL import Image, ImageOps
import io
import numpy as np

app = FastAPI()

model = joblib.load('vision-upload-pipeline')


@app.get('/')
async def root():
    return FileResponse('index.html')
    
label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

@app.post('/predict')
async def predict_image(file : UploadFile = File(...)):
    img = await file.read()
    load_img = Image.open(io.BytesIO(img))

    trans_image = load_img.convert('L')
    inverted_img = ImageOps.invert(trans_image).resize((28,28))
    final_img = np.array(inverted_img).reshape(1,784)
    pred = model.predict(final_img)
    return {'prediction' : label_map[int(pred[0])]}
