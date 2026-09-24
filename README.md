# Vision Upload — Fashion Classifier

Hey there! Welcome to **Vision Upload**, a fast and sleek web application that uses Machine Learning to identify items of clothing. 

Just drop an image of a clothing item (like a shirt, sneaker, or bag), and the app will instantly tell you what it is using a custom-trained model based on the famous **Fashion-MNIST** dataset.

## Features

- **Instant Classification**: Get real-time predictions for 10 different clothing categories.
- **Sleek Interface**: A stunning, responsive, dark-mode-first glassmorphism UI built with plain HTML/CSS/JS (no heavy frontend frameworks required!).
- **Drag & Drop**: Seamlessly upload images by dragging them right onto the page.
- **FastAPI Backend**: A lightning-fast Python backend serving the ML model and frontend.

## Getting Started

Want to run this locally? It's super easy.

### Prerequisites

Make sure you have Python installed, along with the required libraries:
- `fastapi`
- `uvicorn`
- `joblib`
- `Pillow`
- `numpy`
- `scikit-learn` (for the pipeline)

You can install them via pip:
```bash
pip install fastapi uvicorn joblib Pillow numpy scikit-learn python-multipart
```

### Running the App

1. Clone this repository (or download the files).
2. Open your terminal in the project folder.
3. Start the server using uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
4. Open your browser and head over to `http://127.0.0.1:8000`.
5. Drop an image and see the magic happen! ✨

## How it Works

1. **Frontend**: The user drops an image onto the glowing drop zone. The image is previewed locally and then sent to the backend via a POST request to `/predict`.
2. **Backend**: FastAPI receives the image. We use `Pillow` to convert it to grayscale, invert the colors, and resize it to 28x28 pixels to match the format the model was trained on.
3. **Model**: The processed image is flattened and passed into a pre-trained `joblib` model (a scikit-learn pipeline) which predicts the category.
4. **Result**: The result is sent back to the frontend and displayed with a satisfying animation!

## Categories Supported

The model can recognize the following Fashion-MNIST categories:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

