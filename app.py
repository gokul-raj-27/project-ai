from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load trained model
model = tf.keras.models.load_model("best_isl_model_old.keras")

# Class names (A-Z)
CLASS_NAMES = [
    "1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    if confidence < 0.80:
        return jsonify({
        "letter": "No Sign",
        "confidence": round(confidence * 100, 2)
    })

    return jsonify({
        "letter": CLASS_NAMES[index],
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)