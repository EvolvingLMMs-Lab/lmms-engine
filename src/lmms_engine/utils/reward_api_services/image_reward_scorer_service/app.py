import sys
import os
import torch
from PIL import Image
import pickle
import traceback
from io import BytesIO

from flask import Flask, request, Blueprint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import deserialize_images

from imagereward_scorer import ImageRewardScorer

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading ImageReward Scorer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    INFERENCE_FN = ImageRewardScorer(device=device, dtype=torch.float32)
    INFERENCE_FN.eval()
    print("ImageReward Scorer model loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        payload = pickle.loads(data)
        images_bytes = payload["images"]
        prompts = payload.get("prompts", [])
        metadata = payload.get("metadata", {})

        images = deserialize_images(images_bytes)
        print(f"Got {len(images)} images for ImageReward Scorer")

        if not prompts:
            raise ValueError("ImageRewardModel requires prompts.")

        with torch.no_grad():
            rewards = INFERENCE_FN(prompts, images) 
            scores = rewards

        response = {"scores": scores}
        response = pickle.dumps(response)
        return response, 200

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in ImageReward Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18081, debug=True) 