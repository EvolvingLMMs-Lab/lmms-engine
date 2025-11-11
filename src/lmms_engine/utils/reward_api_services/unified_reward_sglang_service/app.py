import sys
import os
import pickle
import traceback
from PIL import Image

from flask import Flask, request, Blueprint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import deserialize_images

from unifiedreward_score_sglang import unifiedreward_score_sglang

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading UnifiedReward Scorer client (for sglang server)...")
    INFERENCE_FN = unifiedreward_score_sglang(device=None)
    print("UnifiedReward Scorer client loaded.")

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
        print(f"Got {len(images)} images for UnifiedReward Scorer client")

        scores_tuple, _ = INFERENCE_FN(images, prompts, metadata)
        scores = scores_tuple

        response = {"scores": scores}
        response = pickle.dumps(response)
        return response, 200

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in UnifiedReward Scorer client service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18086, debug=True) 