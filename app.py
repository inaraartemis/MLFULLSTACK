from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load models
reg_model = joblib.load("models/regression.pkl")
clf_model = joblib.load("models/classifier.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/regression", response_class=HTMLResponse)
def regression_page(request: Request):
    return templates.TemplateResponse("regression.html", {"request": request})


@app.get("/classification", response_class=HTMLResponse)
def classification_page(request: Request):
    return templates.TemplateResponse("classification.html", {"request": request})


@app.post("/predict_regression")
def predict_regression(features: str = Form(...)):
    try:
        values = np.array([float(i) for i in features.split(",")]).reshape(1, -1)
        if values.shape[1] != 10:
            return {"error": f"Expected 10 features, got {values.shape[1]}"}
        prediction = reg_model.predict(values)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_digit_image")
async def predict_digit_image(image_data: str = Form(...)):
    try:
        # data:image/png;base64,...
        header, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(data)).convert("L")
        
        # Smart Inversion detection
        # Digits dataset is high-intensity on low-intensity (white digit on black)
        # We check the mean. If mean > 127, it's likely black on white -> Invert.
        curr_mean = np.mean(np.array(img))
        if curr_mean > 127:
            img = Image.eval(img, lambda x: 255 - x)
            print(f"DEBUG: Image inverted (mean was {curr_mean})")

        # Resize to 8x8 like the digits dataset
        # Using Resampling.LANCZOS for better downsampling
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize to 0-16 (digits dataset range)
        pixel_array = np.array(img).astype(float)
        
        # Ensure range is 0-16
        # The digits dataset values are integers 0 to 16.
        pixel_array = (pixel_array / 255.0) * 16.0
        
        pixel_array = pixel_array.reshape(1, -1)
        
        prediction = clf_model.predict(pixel_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}


@app.post("/predict_classification")
def predict_classification(features: str = Form(...)):
    try:
        values = np.array([float(i) for i in features.split(",")]).reshape(1, -1)
        if values.shape[1] != 64:
            return {"error": f"Expected 64 features, got {values.shape[1]}"}
        prediction = clf_model.predict(values)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}