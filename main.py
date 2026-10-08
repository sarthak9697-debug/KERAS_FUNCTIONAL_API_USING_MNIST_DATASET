import os
os.environ["KERAS_BACKEND"] = "torch"

import io
import pickle
import base64
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Classifier API", 
    description="Production API using Keras Functional API model to predict handwritten digits (0-9)."
)

MODEL_PATH = Path(__file__).with_name("model.pkl")

# Verify and Load the model globally at startup so uvicorn runs smoothly
if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl was not found. Keep main.py and model.pkl in the same folder.")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    if not hasattr(model, "predict"):
        raise TypeError("model.pkl does not contain a usable Keras model.")
except Exception as exc:
    raise RuntimeError(f"Unable to load model.pkl: {exc}")


# ---------------------------------------------------------------------
# Pydantic Input/Output Schemas
# ---------------------------------------------------------------------
class ImageInput(BaseModel):
    # Option 1: Send a raw Base64 image string (Recommended for production)
    base64_image: Optional[str] = Field(None, description="Base64 encoded string of the PNG/JPG image.")
    # Option 2: Directly send a pre-flattened 784-length float array
    raw_array: Optional[List[float]] = Field(None, description="Pre-flattened array of 784 normalized pixel values.")

class PredictionOutput(BaseModel):
    predicted_digit: int = Field(..., description="The predicted digit from 0 to 9.")
    confidence: float = Field(..., description="Confidence score of the highest probability class.")
    probabilities: List[float] = Field(..., description="Probability array across all 10 classes.")


# ---------------------------------------------------------------------
# Core Preprocessing Logic (Directly from your Streamlit code)
# ---------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = ImageOps.contain(image, (28, 28), method=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))
    image = canvas

    array = np.asarray(image, dtype="float32") / 255.0

    # Invert color space if image has a light background
    if array.mean() > 0.55:
        array = 1.0 - array

    return array.reshape(1, 784)


# ---------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_digit(payload: ImageInput):
    # Route A: Handle Direct Array Processing
    if payload.raw_array is not None:
        if len(payload.raw_array) != 784:
            raise HTTPException(status_code=400, detail="Raw array must contain exactly 784 elements.")
        image_array = np.array(payload.raw_array, dtype="float32").reshape(1, 784)
        
    # Route B: Handle Base64 Image Processing
    elif payload.base64_image is not None:
        try:
            # Clean base64 header data if provided (e.g. data:image/png;base64,...)
            b64_data = payload.base64_image.split(",")[-1]
            raw_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(raw_bytes))
            image_array = preprocess_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="You must provide either 'base64_image' or 'raw_array'.")

    # Run Prediction (Uses global model loaded on startup)
    try:
        raw_preds = np.asarray(model.predict(image_array, verbose=0))[0]
        digit = int(np.argmax(raw_preds))
        confidence = float(raw_preds[digit])
        probabilities_list = [float(p) for p in raw_preds]
        
        return {
            "predicted_digit": digit,
            "confidence": confidence,
            "probabilities": probabilities_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")
