"""
app.py

Backend Flask: sirve la página web y expone un endpoint /predict que recibe
una imagen (capturada desde la cámara del navegador) y devuelve la categoría
de desecho predicha junto con la confianza del modelo.
"""

import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "waste_classifier.h5"
CLASSES_PATH = "class_names.json"

model = None
class_names = None


def load_model_if_needed():
    """Carga el modelo entrenado una sola vez (lazy loading)."""
    global model, class_names

    if model is not None:
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            "No se encontró 'waste_classifier.h5' o 'class_names.json'. "
            "Primero corre: python train_model.py"
        )

    # Import diferido para que la app arranque rápido aunque el modelo pese
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Las llaves quedan como strings al leer JSON; las volvemos a int
    class_names = {int(k): v for k, v in raw.items()}

    app.config["preprocess_input"] = preprocess_input


def decode_base64_image(data_url: str) -> Image.Image:
    """Convierte un data URL 'data:image/...;base64,XXXX' a imagen PIL."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model_if_needed()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    try:
        image = decode_base64_image(data["image"])
    except Exception:
        return jsonify({"error": "No se pudo procesar la imagen enviada."}), 400

    # Preprocesamiento igual que en el entrenamiento
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32")
    arr = app.config["preprocess_input"](arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    # Top 3 para mostrar contexto además de la predicción principal
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"clase": class_names[i], "confianza": round(float(preds[i]) * 100, 1)}
        for i in top3_idx
    ]

    return jsonify(
        {
            "clase": class_names[top_idx],
            "confianza": round(confidence * 100, 1),
            "top3": top3,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
