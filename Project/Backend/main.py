from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os
import random

# =========================================
# APP SETUP
# =========================================

app = Flask(__name__)

CORS(app)

# =========================================
# CONFIGURATION
# =========================================

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================
# FAKE AI DATABASE
# Replace later with real AI model
# =========================================

FISH_SPECIES = [
    "Goldfish",
    "Betta Fish",
    "Clownfish",
    "Guppy",
    "Koi Fish",
    "Angelfish",
    "Salmon",
    "Tilapia",
    "Catfish",
    "Tuna"
]

# =========================================
# CHECK FILE TYPE
# =========================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():

    return render_template("index.html")

# =========================================
# RESULTS PAGE
# =========================================

@app.route("/results")
def results():

    return render_template("index.html")

# =========================================
# PREDICTION ROUTE
# =========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================================
        # CHECK IMAGE EXIST
        # =========================================

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "message": "No image uploaded"
            })

        file = request.files["image"]

        # =========================================
        # CHECK EMPTY FILE
        # =========================================

        if file.filename == "":

            return jsonify({
                "success": False,
                "message": "No file selected"
            })

        # =========================================
        # CHECK FILE TYPE
        # =========================================

        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "message": "Invalid file type"
            })

        # =========================================
        # SAVE IMAGE
        # =========================================

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # =========================================
        # FAKE AI PREDICTION
        # Replace this later with TensorFlow model
        # =========================================

        predicted_fish = random.choice(
            FISH_SPECIES
        )

        confidence = round(
            random.uniform(85, 99),
            2
        )

        # =========================================
        # RETURN RESPONSE
        # =========================================

        return jsonify({

            "success": True,

            "fish": predicted_fish,

            "confidence": confidence,

            "image": filename,

            "message": "Prediction successful"

        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({

            "success": False,

            "message": "Server error occurred"

        })

# =========================================
# START SERVER
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )