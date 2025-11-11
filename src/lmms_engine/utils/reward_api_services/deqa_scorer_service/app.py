from PIL import Image
from io import BytesIO
import pickle
import traceback
from deqa_scorer import load_deqascore 
import numpy as np
import os

from flask import Flask, request, Blueprint
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import deserialize_images

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading DeQA Scorer model...")
    INFERENCE_FN = load_deqascore() 
    print("DeQA Scorer model loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        # expects a dict with "images", "prompts", and optionally "metadata"
        payload = pickle.loads(data)

        images_bytes = payload["images"]
        prompts = payload.get("prompts", []) # Not strictly used by DeQA but for API consistency
        metadata = payload.get("metadata", {})

        images = deserialize_images(images_bytes) # Use common deserialize_images
        print(f"Got {len(images)} images for DeQA Scorer")

        outputs = INFERENCE_FN(images) # DeQA returns list of floats

        response = {"scores": outputs} # Ensure consistent key "scores"
        response = pickle.dumps(response)

        return response, 200
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in DeQA Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18084, debug=True) 