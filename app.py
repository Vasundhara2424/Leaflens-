import os
from pathlib import Path
from urllib.request import urlopen, Request
from flask import Flask, send_from_directory, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from io import BytesIO

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "plant_disease_efficientnet.keras"
CLASSES_PATH = MODEL_DIR / "class_names.txt"
MODEL_URL = "https://huggingface.co/Nefflymicn/PlantVillage-plant-disease-detection/resolve/main/plant_disease_efficientnet.keras"
CLASSES_URL = "https://huggingface.co/Nefflymicn/PlantVillage-plant-disease-detection/resolve/main/class_names.txt"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
_model = None
_classes = None

def download_if_missing(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        return
    req = Request(url, headers={"User-Agent": "LeafLens-student-project/1.0"})
    temp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(req, timeout=120) as response, open(temp, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        temp.replace(destination)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise

def load_model_once():
    global _model, _classes
    if _model is None:
        download_if_missing(MODEL_URL, MODEL_PATH)
        download_if_missing(CLASSES_URL, CLASSES_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        parsed = {}
        for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                idx_text, name = line.split(":", 1)
                try:
                    parsed[int(idx_text.strip())] = name.strip()
                except ValueError:
                    continue
        if not parsed:
            raise RuntimeError("The model class list could not be parsed.")
        output_count = int(_model.output_shape[-1])
        if len(parsed) != output_count or any(i not in parsed for i in range(output_count)):
            raise RuntimeError(f"Model has {output_count} outputs but the class-name file does not map every output.")
        _classes = [parsed[i] for i in range(output_count)]
    return _model, _classes

def friendly_name(label):
    return label.replace("___", " — ").replace("__", " — ").replace("_", " ")

def guidance_for(label):
    low = label.lower()
    if "healthy" in low:
        return "The model classified this image as healthy. Continue regular monitoring and good watering, nutrition, and sanitation practices."
    if any(word in low for word in ("rust", "blight", "spot", "scab", "mildew", "rot", "mosaic", "virus", "bacterial")):
        return "Remove heavily affected plant material where appropriate, avoid handling wet foliage, clean tools, and improve airflow. Confirm the disease locally before choosing any pesticide or treatment."
    return "Inspect nearby plants and confirm the cause with a local agricultural extension officer before applying treatments. Avoid spraying chemicals based only on an image prediction."

@app.get("/")
def home():
    return send_from_directory(APP_DIR, "index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="Please choose a leaf image first."), 400
    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify(error="The uploaded image has no filename."), 400
    try:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file could not be read as an image. Try a JPG or PNG photo."), 400
    try:
        model, classes = load_model_once()
        array = np.asarray(image, dtype=np.float32)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        # Match the model author's documented EfficientNet preprocessing.
        array = preprocess_input(array)
        scores = np.asarray(model.predict(np.expand_dims(array, axis=0), verbose=0))[0]
        index = int(np.argmax(scores))
        confidence = float(scores[index]) * 100
        label = friendly_name(classes[index])
        return jsonify(prediction=label, confidence=round(confidence, 2),
                       low_confidence=confidence < 60, guidance=guidance_for(classes[index]))
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error="The AI model could not be loaded or run. Check internet access for the first model download and see the terminal error. Details: " + str(exc)[:240]), 503

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Please choose an image smaller than 10 MB."), 413

if __name__ == "__main__":
    # Local development only. For public hosting, use a production WSGI server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
