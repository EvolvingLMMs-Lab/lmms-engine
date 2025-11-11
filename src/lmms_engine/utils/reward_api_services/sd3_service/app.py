import sys
import os
import torch
from PIL import Image
import pickle
import traceback
from io import BytesIO
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import load_image

from flask import Flask, request, Blueprint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common')))

PIPELINE = None

root = Blueprint("root", __name__)

def create_app():
    global PIPELINE
    print("Loading Stable Diffusion 3 Medium model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    PIPELINE = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=dtype,
        # variant="bf16" if device == "cuda" else None,
    )
    
    PIPELINE = PIPELINE.to(device)
    
    PIPELINE.enable_model_cpu_offload() 
    
    # if hasattr(torch, 'compile'):
    #     PIPELINE.unet = torch.compile(PIPELINE.unet, mode="reduce-overhead", fullgraph=True)
    
    PIPELINE.safety_checker = None 
    
    print("Stable Diffusion 3 Medium model loaded successfully.")
    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def generate_image():
    print(f"Received POST request from {request.remote_addr}")
    data = request.get_data()
    
    try:
        payload = pickle.loads(data)
        prompts = payload.get("prompts", "")
        negative_prompts = payload.get("negative_prompts", None)
        width = payload.get("width", 512)
        height = payload.get("height", 512)
        num_inference_steps = payload.get("num_inference_steps", 10)
        guidance_scale = payload.get("guidance_scale", 3.5)
        num_images = payload.get("num_images", 1)
        
        if not prompts or len(prompts) == 0:
            raise ValueError("Prompt cannot be empty")
        
        print(f"Generating {num_images} image(s) for prompt: '{prompts[0]}...'")
        print(f"Resolution: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        with torch.inference_mode():
            if num_images > 1:
                images = PIPELINE(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                ).images
            else:
                images = PIPELINE(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                ).images
        
        image_bytes_list = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            image_bytes_list.append(img_bytes)
        
        response_data = {
            "success": True,
            "images": image_bytes_list,
            "prompt": prompts,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": len(images)
        }
        
        response = pickle.dumps(response_data)
        print(f"Successfully generated {len(images)} image(s)")
        return response, 200
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error in Stable Diffusion service: {error_msg}")
        response_data = {
            "success": False,
            "error": str(e),
            "traceback": error_msg
        }
        response = pickle.dumps(response_data)
        return response, 500

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=18099, debug=True)