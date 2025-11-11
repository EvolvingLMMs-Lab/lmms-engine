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

from pickscore_scorer import PickScoreScorer

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading PickScore Scorer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    INFERENCE_FN = PickScoreScorer(device=device, dtype=torch.float32)
    INFERENCE_FN.eval()
    print("PickScore Scorer model loaded.")

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
        print(f"Got {len(images)} images for PickScore Scorer")

        if not prompts:
            raise ValueError("PickScore Scorer requires prompts.")

        scores = []
        with torch.no_grad():
            for i, img in enumerate(images):
                p = prompts[i] if i < len(prompts) else prompts[0] 
                single_score = INFERENCE_FN([p], [img]).cpu().tolist() 
                scores.extend(single_score)

        response = {"scores": scores}
        response = pickle.dumps(response)
        return response, 200

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in PickScore Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18083, debug=True) # 不同的端口