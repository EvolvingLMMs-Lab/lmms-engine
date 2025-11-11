from PIL import Image
from io import BytesIO
import pickle
import traceback
from gen_eval import load_geneval 
import numpy as np
import os

from flask import Flask, request, Blueprint
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))
from utils import deserialize_images
import torch

INFERENCE_FN = None

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    print("Loading GenEval Scorer model...")
    os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmdetection_base_path = os.path.join(script_dir, "mmdetection")
    
    os.environ["MY_CONFIG_PATH"] = os.path.join(mmdetection_base_path, "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
    os.environ["MY_CKPT_PATH"] = os.path.join(mmdetection_base_path, "weights/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth")

    INFERENCE_FN = load_geneval()
    print("GenEval Scorer model loaded.")

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        # expects a dict with "images", "prompts", "meta_datas", "only_strict"
        payload = pickle.loads(data)

        images_bytes = payload["images"]
        prompts = payload.get("prompts", []) # Not strictly used by GenEval but for API consistency
        # GenEval 需要 meta_datas 和 only_strict 参数
        meta_datas = payload.get("metadata", {}).get("meta_datas", []) # 从 metadata 中提取
        only_strict = payload.get("metadata", {}).get("only_strict", False)

        images = deserialize_images(images_bytes)
        print(f"Got {len(images)} images for GenEval Scorer")

        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = INFERENCE_FN(images, meta_datas, only_strict)

        # 返回 GenEval 的所有结果
        response = {
            "scores": scores, 
            "rewards": rewards, 
            "strict_rewards": strict_rewards, 
            "group_rewards": group_rewards, 
            "group_strict_rewards": group_strict_rewards
        }
        response = pickle.dumps(response)

        return response, 200
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in GenEval Scorer service: {error_msg}")
        response = {"error": error_msg}
        response = pickle.dumps(response)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18085, debug=True) # 使用你原始 gunicorn.conf.py 中定义的端口