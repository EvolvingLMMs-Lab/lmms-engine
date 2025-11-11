import os
import requests
from clint.textui import progress
from PIL import Image
import torch
import torch.nn as nn 
# from .open_clip import create_model_and_transforms, get_tokenizer
import open_clip
import numpy as np
from huggingface_hub import hf_hub_download



HPSV2_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_checkpoints')


class Selector():
    def __init__(self, device):
        self.device = device

        model, preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        if not os.path.exists(HPSV2_ROOT_PATH):
            os.makedirs(HPSV2_ROOT_PATH)
        checkpoint_path = os.path.join(HPSV2_ROOT_PATH, 'HPS_v2.1_compressed.pt')
        if not os.path.exists(checkpoint_path):
            print('Downloading HPS_v2_compressed.pt from Hugging Face...')
            checkpoint_path = hf_hub_download(
                repo_id="xswu/HPSv2",
                filename="HPS_v2.1_compressed.pt",
                cache_dir=HPSV2_ROOT_PATH
            )
            print(f'Downloaded to {checkpoint_path}')
        print('Loading HPSv2 model ...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        

        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        
        model = model.to(device)
        model.eval()
        self.model = model
        print('HPSv2 model loaded successfully!')

    def score(self, images, prompt):
        result = []
        with torch.no_grad():
            for one_img in images:
                if isinstance(one_img, Image.Image):
                    image = self.preprocess_val(one_img).unsqueeze(0).to(device=self.device, non_blocking=True)
                else:
                    raise TypeError('Input images must be PIL.Image objects.')

                text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
                
                outputs = self.model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])
        return result


class HPSv2Scorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        print(f"Initializing HPSv2 Selector on device: {self.device}")
        self.selector = Selector(device=self.device)
        print("HPSv2 Selector initialized.")
        
    @torch.no_grad()
    def __call__(self, images, prompts, metadata=None):
        if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
            raise TypeError("Images must be a list of PIL.Image objects.")
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("Prompts must be a list of strings.")
        if len(images) != len(prompts):
            if len(prompts) != 1:
                raise ValueError("When using HPSv2Scorer, if images and prompts length differ, prompts must contain exactly one prompt applicable to all images.")
            single_prompt = prompts[0]
            print(f"Using single prompt '{single_prompt[:50]}...' for all {len(images)} images.")
            scores = self.selector.score(images=images, prompt=single_prompt)

        else:
            scores = []
            for i, (image, prompt) in enumerate(zip(images, prompts)):
                print(f"Calculating HPSv2 score for image {i+1}/{len(images)} with prompt: {prompt[:50]}...")
                hps_score = self.selector.score(images=[image], prompt=prompt) 
                scores.extend(hps_score)
            scores = [score * 2 for score in scores]  # Scale scores to match the expected range
        
        return scores

