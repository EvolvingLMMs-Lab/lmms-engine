import io
from collections import defaultdict

import numpy as np
import torch
from PIL import Image


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score():
    from .aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from .clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def image_similarity_score(device):
    from .clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8) / 255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn


def pickscore_score(device):
    from loguru import logger

    from .pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        logger.info(f"pickscore_score: input images type: {type(images)}, prompts type: {type(prompts)}")

        if isinstance(images, torch.Tensor):
            logger.info(f"pickscore_score: images tensor shape: {images.shape}")
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            logger.info(f"pickscore_score: converted to {len(images)} PIL images")

        # Ensure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        logger.info(f"pickscore_score: prompts len: {len(prompts)}, images len: {len(images)}")

        try:
            scores = scorer(prompts, images)
            logger.info(f"pickscore_score: scorer returned type: {type(scores)}, value: {scores}")
        except Exception as e:
            logger.error(f"pickscore_score: scorer failed: {e}", exc_info=True)
            raise

        # Convert tensor to list
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
            logger.info(f"pickscore_score: converted tensor to list, len: {len(scores)}")
        elif not isinstance(scores, list):
            scores = list(scores)
            logger.info(f"pickscore_score: converted to list, len: {len(scores)}")

        logger.info(f"pickscore_score: final scores len: {len(scores)}, scores: {scores}")
        return scores, {}

    return _fn


def imagereward_score(device):
    from .imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        # Ensure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        scores = scorer(prompts, images)
        # Convert tensor to list
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        elif not isinstance(scores, list):
            scores = list(scores)
        return scores, {}

    return _fn


def qwenvl_score(device):
    from .qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        # Ensure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        scores = scorer(prompts, images)
        # Convert tensor to list
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        elif not isinstance(scores, list):
            scores = list(scores)
        return scores, {}

    return _fn


def ocr_score(device):
    from .ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def video_ocr_score(device):
    from .ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward."""
    import pickle
    from io import BytesIO

    import requests
    from requests.adapters import HTTPAdapter, Retry

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()

    # NOTE:禁用代理，直接连接本地服务器 New Added on 2025-11-26
    # This is to avoid the proxy settings in the environment variables affecting the requests library.
    import os

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,0.0.0.0")
    sess.proxies = {"http": None, "https": None}

    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            # 确保不使用代理，直接连接本地服务器
            try:
                response = sess.post(url, data=data_bytes, timeout=120)
                if response.status_code != 200:
                    raise Exception(f"Server returned status code {response.status_code}: {response.content[:500]}")
                response_data = pickle.loads(response.content)
            except Exception as e:
                print(f"Error in deqa_score_remote request: {e}")
                print(f"URL: {url}")
                print(f"Proxy settings: {sess.proxies}")
                print(f"NO_PROXY env: {os.environ.get('NO_PROXY', 'not set')}")
                raise

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn


def geneval_score(device):
    """Submits images to GenEval and computes a reward."""
    import os
    import pickle
    from io import BytesIO

    import requests
    from requests.adapters import HTTPAdapter, Retry

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()

    # NOTE: 禁用代理，直接连接本地服务器
    # This is to avoid the proxy settings in the environment variables affecting the requests library.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,0.0.0.0")
    sess.proxies = {"http": None, "https": None}

    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            # 确保不使用代理，直接连接本地服务器
            try:
                response = sess.post(url, data=data_bytes, timeout=120)
                if response.status_code != 200:
                    raise Exception(f"Server returned status code {response.status_code}: {response.content[:500]}")
                response_data = pickle.loads(response.content)
            except Exception as e:
                print(f"Error in geneval_score request: {e}")
                print(f"URL: {url}")
                print(f"Proxy settings: {sess.proxies}")
                print(f"NO_PROXY env: {os.environ.get('NO_PROXY', 'not set')}")
                raise

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward."""
    import pickle
    from io import BytesIO

    import requests
    from requests.adapters import HTTPAdapter, Retry

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {"images": jpeg_images, "prompts": prompt_batch}
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    import base64
    import re
    from io import BytesIO

    from openai import AsyncOpenAI

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


def multi_score(device, score_dict):
    from types import SimpleNamespace

    from loguru import logger

    # Convert SimpleNamespace to dict if needed
    if isinstance(score_dict, SimpleNamespace):
        score_dict = {k: v for k, v in score_dict.__dict__.items()}
        logger.info(f"multi_score: Converted SimpleNamespace to dict: {score_dict}")

    # Ensure score_dict is a dict
    if not isinstance(score_dict, dict):
        logger.error(f"multi_score: score_dict must be a dict or SimpleNamespace, got {type(score_dict)}")
        raise ValueError(f"score_dict must be a dict, got {type(score_dict)}")

    if len(score_dict) == 0:
        logger.error(f"multi_score: score_dict is empty! This will result in no rewards being computed.")
        raise ValueError("score_dict cannot be empty")

    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        from loguru import logger

        total_scores = []
        score_details = {}

        # Debug: log input shapes (use info level to ensure visibility)
        if isinstance(images, torch.Tensor):
            logger.info(f"multi_score: images shape: {images.shape}, dtype: {images.dtype}")
        else:
            logger.info(
                f"multi_score: images type: {type(images)}, len: {len(images) if hasattr(images, '__len__') else 'N/A'}"
            )
        logger.info(
            f"multi_score: prompts type: {type(prompts)}, len: {len(prompts) if isinstance(prompts, (list, tuple)) else 'N/A'}"
        )
        logger.info(f"multi_score: score_dict: {score_dict}")

        for score_name, weight in score_dict.items():
            try:
                if score_name == "geneval":
                    scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                        images, prompts, metadata, only_strict
                    )
                    score_details["accuracy"] = rewards
                    score_details["strict_accuracy"] = strict_rewards
                    for key, value in group_strict_rewards.items():
                        score_details[f"{key}_strict_accuracy"] = value
                    for key, value in group_rewards.items():
                        score_details[f"{key}_accuracy"] = value
                elif score_name == "image_similarity":
                    scores, rewards = score_fns[score_name](images, ref_images)
                else:
                    scores, rewards = score_fns[score_name](images, prompts, metadata)

                # Convert scores to list if needed
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().tolist()
                elif not isinstance(scores, list):
                    scores = list(scores)

                logger.info(
                    f"multi_score: {score_name} scores type: {type(scores)}, len: {len(scores) if hasattr(scores, '__len__') else 'N/A'}, scores: {scores[:5] if len(scores) > 0 else 'empty'}"
                )

                if len(scores) == 0:
                    logger.warning(
                        f"multi_score: {score_name} returned empty scores! images: {type(images)}, prompts: {type(prompts)}"
                    )
                    continue

                score_details[score_name] = scores
                weighted_scores = [weight * score for score in scores]

                if not total_scores:
                    total_scores = weighted_scores
                else:
                    if len(total_scores) != len(weighted_scores):
                        logger.error(
                            f"multi_score: length mismatch! total_scores: {len(total_scores)}, weighted_scores: {len(weighted_scores)}"
                        )
                    total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
            except Exception as e:
                logger.error(f"multi_score: Error computing {score_name} score: {e}", exc_info=True)
                raise

        logger.info(
            f"multi_score: total_scores len: {len(total_scores)}, total_scores: {total_scores[:5] if len(total_scores) > 0 else 'empty'}"
        )

        if len(total_scores) == 0:
            logger.error(
                f"multi_score: total_scores is empty! score_dict: {score_dict}, images: {type(images)}, prompts: {type(prompts)}"
            )
            # Try to compute at least one score manually for debugging
            if "pickscore" in score_dict:
                logger.error("Attempting to debug pickscore...")
                try:
                    test_scores = score_fns["pickscore"](images, prompts, metadata)
                    logger.error(f"Direct pickscore call result: {test_scores}")
                except Exception as e:
                    logger.error(f"Direct pickscore call failed: {e}", exc_info=True)

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
