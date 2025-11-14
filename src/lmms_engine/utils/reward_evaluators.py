import os
import torch
import requests
import pickle
from PIL import Image
from typing import List, Dict, Any, Union
import sys
import os
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor



class RewardEvaluatorClient:
    def __init__(self, scorer_urls: Dict[str, str]):
        self.scorer_urls = scorer_urls

    def evaluate(self, 
                 model_name: str, 
                 images: List[Image.Image], 
                 prompts: List[str], 
                 metadata: Dict[str, Any] = None
        ) -> Union[List[float], Dict[str, Any]]:
        url = self.scorer_urls.get(model_name)
        if not url:
            raise ValueError(f"Reward model '{model_name}' URL not configured.")

        payload_bytes = create_payload(images, prompts, metadata)

        try:
            response = requests.post(url, data=payload_bytes, timeout=600) 
            response.raise_for_status() 

            result = parse_response(response.content)
            
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"Scorer '{model_name}' service returned error: {result['error']}")
            
            return result

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request to '{model_name}' failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to process response from '{model_name}': {e}")

    def evaluate_multiple(self, 
                          model_weights: Dict[str, float], 
                          images: List[Image.Image], 
                          prompts: List[str], 
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        all_results = {}
        for model_name, weight in model_weights.items():
            if weight == 0: 
                continue
            try:
                if model_name in ["gen_eval", "unifiedreward_sglang"]:
                    specific_metadata = metadata.get(model_name, {})
                    result = self.evaluate(model_name, images, prompts, specific_metadata)
                else:
                    result = self.evaluate(model_name, images, prompts, metadata.get(model_name, {}))
                all_results[model_name] = result
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        return all_results

    def pickscore(self, images, prompts):
        return self.evaluate("pickscore", images, prompts)

    def deqa(self,images, prompts):
        return self.evaluate("deqa", images, prompts)

    def gen_eval(self, images, prompts, meta_files):
        return self.evaluate("gen_eval", images, prompts, meta_files)

    def unifiedreward_sglang(self, images, prompts):
        return self.evaluate("unifiedreward_sglang", images, prompts)

    def ocr(self, images, prompts):
        return self.evaluate("ocr", images, prompts)

    def image_reward(self, images, prompts):
        return self.evaluate("image_reward", images, prompts)

    def aesthetic(self, images, prompts):
        return self.evaluate("aesthetic", images, prompts)

    def hps(self, images, prompts):
        return self.evaluate("hps", images, prompts)

def serialize_images(images: List[Image.Image]) -> List[bytes]:
    images_bytes = []
    for img in images:
        img_byte_arr = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save(img_byte_arr, format="JPEG", quality=95)
        img.save(img_byte_arr, format="JPEG")
        images_bytes.append(img_byte_arr.getvalue())
    return images_bytes

def deserialize_images(images_bytes: List[bytes]) -> List[Image.Image]:
    images = [Image.open(BytesIO(d)) for d in images_bytes]
    return images

def create_payload(images: List[Image.Image], prompts: List[str], metadata: Dict[str, Any] = None) -> bytes:
    serialized_images = serialize_images(images)
    payload = {
        "images": serialized_images,
        "prompts": prompts,
        "metadata": metadata if metadata is not None else {}
    }
    return pickle.dumps(payload)

def parse_response(response_content: bytes) -> Union[List[float], Dict[str, Any]]:
    return pickle.loads(response_content)


def recon_reward(orig_images, recon_images):
    """
    Compute rewards based on PSNR between original and reconstructed tensor images.
    Images are assumed to have values normalized between 0 and 1, with maximum value exactly 1.0.
    
    Args:
        orig_images (torch.Tensor): Batch of original images [B, C, H, W], values in [0, 1].
        recon_images (torch.Tensor): Batch of reconstructed images [B, C, H, W], values in [0, 1].
    
    Returns:
        list: List of PSNR values as rewards for each image pair.
    """
    def calculate_psnr(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Image dimensions must match")
        
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        perfect_match = mse == 0
        psnr = torch.zeros_like(mse)
        valid = ~perfect_match
        psnr[valid] = -10 * torch.log10(mse[valid])
        psnr[perfect_match] = float('inf')
        return psnr
    
    orig_images = orig_images.float().clamp(0, 1)
    recon_images = recon_images.float().clamp(0, 1)
    rewards = calculate_psnr(orig_images, recon_images)
    return rewards

def jpeg_incompressibility(images):
    buffers = [io.BytesIO() for _ in images]
    for image, buffer in zip(images, buffers):
        image.save(buffer, format="JPEG", quality=95)
    sizes = [buffer.tell() / 1000 for buffer in buffers]
    return torch.tensor(sizes)

def jpeg_compressibility(images):
    return jpeg_incompressibility(images) * -1.0 / 500

def sim_direction(orig_images, edited_images, original_caption, edited_caption, clip_model, clip_processor):
    """
    Compute sim_direction using HuggingFace CLIP by calculating cosine similarity 
    between the difference of image features and the difference of text features.
    
    Args:
        orig_images: Original image or list of images (PIL.Image)
        edited_images: Edited image or list of images (PIL.Image)
        original_caption: List of original text prompts
        edited_caption: List of edited text prompts
        clip_model: Preloaded HuggingFace CLIP model
        clip_processor: Preloaded HuggingFace CLIP processor
    
    Returns:
        torch.Tensor: Cosine similarity of image and text feature differences (sim_direction)
    """
    device = clip_model.device
    inputs_orig = clip_processor(images=orig_images, text=original_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_edited = clip_processor(images=edited_images, text=edited_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
    inputs_edited = {k: v.to(device) for k, v in inputs_edited.items()}
    
    with torch.no_grad():
        image_features_orig = clip_model(**inputs_orig)
        image_features_edited = clip_model(**inputs_edited)
    
    image_features_orig, text_features_orig = image_features_orig.image_embeds, image_features_orig.text_embeds
    image_features_edited, text_features_edited = image_features_edited.image_embeds, image_features_edited.text_embeds
    
    image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)
    image_features_edited = image_features_edited / image_features_edited.norm(dim=-1, keepdim=True)
    text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
    text_features_edited = text_features_edited / text_features_edited.norm(dim=-1, keepdim=True)
    
    sim_direction = F.cosine_similarity(
        image_features_edited - image_features_orig, 
        text_features_edited - text_features_orig
    )
    return sim_direction

def clip_similarity(orig_images, recon_images, clip_model, clip_processor):
    """
    Compute cosine similarity between images and reference images using CLIP embeddings.
    
    Args:
        orig_images (list[PIL.Image]): List of original PIL images.
        recon_images (list[PIL.Image]): List of reconstructed PIL images.
        clip_model (CLIPModel): Preloaded CLIP model for computing embeddings.
        clip_processor (CLIPProcessor): Preloaded CLIP processor for image preprocessing.
    
    Returns:
        torch.Tensor: Cosine similarity scores for each image pair.
    """
    ref_inputs = clip_processor(images=orig_images, return_tensors="pt")
    recon_inputs = clip_processor(images=recon_images, return_tensors="pt")
    ref_inputs = {k: v.to(clip_model.device) for k, v in ref_inputs.items()}
    recon_inputs = {k: v.to(clip_model.device) for k, v in recon_inputs.items()}
    
    with torch.no_grad():
        ref_embeddings = clip_model.get_image_features(**ref_inputs)
        recon_embeddings = clip_model.get_image_features(**recon_inputs)
    
    similarity = F.cosine_similarity(ref_embeddings, recon_embeddings, dim=-1)
    return similarity

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<answer>.*?</answer>'
    extracted_contents = [
        s[s.find('\nassistant') + len('\nassistant'):].strip() if s.find('\nassistant') != -1 else ""
        for s in completions
    ]
    matches = list(map(lambda x: re.match(pattern, x), extracted_contents))
    return [1.0 if match else 0.0 for match in matches]