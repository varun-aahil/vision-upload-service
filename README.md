# Vision Upload: Fashion Classifier

Drop an image of a clothing item and get its category back. A scikit-learn pipeline trained on Fashion-MNIST runs behind a FastAPI endpoint with a drag-and-drop web page.

![alt text](<Screenshot 2026-09-30 141801.png>)

## Model

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 70,000 grayscale 28x28 images, 10 classes
- **Approach:** scikit-learn pipeline saved with `joblib`
- **Test accuracy:** TODO
- **Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot


## How it works

1. The page sends the dropped image to `POST /predict`.
2. FastAPI uses Pillow to convert it to grayscale, invert the colors, and resize it to 28x28 so it matches the training data.
3. The flattened pixels go through the saved pipeline, which predicts the class.
4. The label is returned and shown in the UI.

## Known limitation

The model is trained on clean Fashion-MNIST images, so real photos with busy backgrounds can be misclassified. A CNN with data augmentation would handle this better.

## Run locally

```bash
git clone https://github.com/varun-aahil/vision-upload-service.git
cd vision-upload-service
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 and drop an image.

## Tech stack

Python, scikit-learn, FastAPI, Pillow, NumPy, joblib, vanilla HTML/CSS/JS

## Next steps

- Retrain as a small CNN in PyTorch and compare it against this pipeline
- Add confidence scores and the top 3 predictions to the response
- Add tests for the preprocessing step
