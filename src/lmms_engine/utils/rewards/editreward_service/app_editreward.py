"""
Flask application for EditReward service.

Usage:
    # Start with gunicorn (recommended for production):
    gunicorn -c gunicorn.conf.py 'app_editreward:create_app()'
    
    # Or run directly for debugging:
    python app_editreward.py
"""
import base64
import pickle
import traceback
from io import BytesIO

import numpy as np
from flask import Blueprint, Flask, request
from PIL import Image

from lmms_engine.utils.rewards.editreward_service.editreward import load_editreward

root = Blueprint("root", __name__)

INFERENCE_FN = None


def create_app():
    """Create and configure the Flask application."""
    global INFERENCE_FN
    INFERENCE_FN = load_editreward()

    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/", methods=["POST"])
def inference():
    """
    Main inference endpoint.

    Expects POST data (pickled dict):
        - prompts: List[str] - editing instructions
        - source_images: List[bytes] - PNG/JPEG encoded source images
        - edited_images: List[bytes] - PNG/JPEG encoded edited images

    Returns (pickled dict):
        - scores: List[float] or List[List[float]] - reward scores
    """
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        data = pickle.loads(data)

        # Decode images (supports PNG, JPEG, etc.)
        source_images = [Image.open(BytesIO(d)).convert("RGB") for d in data["source_images"]]
        edited_images = [Image.open(BytesIO(d)).convert("RGB") for d in data["edited_images"]]
        prompts = data["prompts"]

        print(f"Got {len(source_images)} source images, {len(edited_images)} edited images, {len(prompts)} prompts")

        scores = INFERENCE_FN(prompts, source_images, edited_images)

        response = {"scores": scores}
        response = pickle.dumps(response)
        returncode = 200

    except Exception as e:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


@root.route("/json", methods=["POST"])
def inference_json():
    """
    JSON inference endpoint for easier integration.

    Expects JSON data:
        - prompts: List[str] - editing instructions
        - source_images: List[str] - base64 encoded source images
        - edited_images: List[str] - base64 encoded edited images

    Returns JSON:
        - scores: List[float] or List[List[float]] - reward scores
    """
    from flask import jsonify

    print(f"Received JSON POST request from {request.remote_addr}")

    try:
        data = request.get_json()

        # Decode base64 images
        source_images = []
        for img_b64 in data["source_images"]:
            img_bytes = base64.b64decode(img_b64)
            source_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        edited_images = []
        for img_b64 in data["edited_images"]:
            img_bytes = base64.b64decode(img_b64)
            edited_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

        prompts = data["prompts"]

        print(f"Got {len(source_images)} source images, {len(edited_images)} edited images")

        scores = INFERENCE_FN(prompts, source_images, edited_images)

        return jsonify({"scores": scores})

    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500


@root.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return "OK", 200


HOST = "127.0.0.1"
PORT = 18087  # Default port for EditReward

if __name__ == "__main__":
    app = create_app()
    app.run(HOST, PORT, debug=True)
