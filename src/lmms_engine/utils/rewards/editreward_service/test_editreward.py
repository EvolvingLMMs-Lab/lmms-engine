"""
Test script for EditReward server.

Environment variables:
    EDITREWARD_TEST_IMAGES: Path to test images directory (required)
    
Usage:
    export EDITREWARD_TEST_IMAGES=/path/to/test/images
    python test_editreward.py [server_url]
"""

import os
import pickle
import sys
from io import BytesIO

import requests
from PIL import Image

# Disable proxy for local connections
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


def encode_image(img: Image.Image) -> bytes:
    """Encode PIL Image to PNG bytes."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_server(url: str = "http://127.0.0.1:18087"):
    """Test EditReward server."""
    test_images_dir = os.environ.get("EDITREWARD_TEST_IMAGES")
    if not test_images_dir:
        print("Error: Set EDITREWARD_TEST_IMAGES environment variable to test images directory")
        print("Example: export EDITREWARD_TEST_IMAGES=/path/to/EditReward/assets/examples")
        return

    source_path = os.path.join(test_images_dir, "source_img_1.png")
    good_path = os.path.join(test_images_dir, "target_img_2.png")
    bad_path = os.path.join(test_images_dir, "target_img_1.png")

    for p in [source_path, good_path, bad_path]:
        if not os.path.exists(p):
            print(f"Error: Test image not found: {p}")
            return

    source = Image.open(source_path).convert("RGB")
    good = Image.open(good_path).convert("RGB")
    bad = Image.open(bad_path).convert("RGB")
    prompt = "Add a green bowl on the branch"

    # Test 1: Single image
    print("=" * 50)
    print("Test 1: Single image (good edit)")
    print("=" * 50)

    data = {
        "prompts": [prompt],
        "source_images": [encode_image(source)],
        "edited_images": [encode_image(good)],
    }

    resp = requests.post(url, data=pickle.dumps(data), proxies={"http": None, "https": None}, timeout=120)
    if resp.status_code == 200:
        result = pickle.loads(resp.content)
        print(f"Score: {result['scores']}")
    else:
        print(f"Error: {resp.status_code}\n{resp.content.decode()}")
        return

    # Test 2: Batch comparison
    print("\n" + "=" * 50)
    print("Test 2: Batch (good vs bad)")
    print("=" * 50)

    data = {
        "prompts": [prompt, prompt],
        "source_images": [encode_image(source), encode_image(source)],
        "edited_images": [encode_image(good), encode_image(bad)],
    }

    resp = requests.post(url, data=pickle.dumps(data), proxies={"http": None, "https": None}, timeout=120)
    if resp.status_code == 200:
        result = pickle.loads(resp.content)
        print(f"Good edit: {result['scores'][0]}")
        print(f"Bad edit: {result['scores'][1]}")
    else:
        print(f"Error: {resp.status_code}")

    # Test 3: Health check
    print("\n" + "=" * 50)
    print("Test 3: Health check")
    print("=" * 50)

    resp = requests.get(f"{url}/health", proxies={"http": None, "https": None}, timeout=10)
    print(f"Status: {resp.status_code} - {resp.text}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18087"
    print(f"Testing EditReward server at {url}\n")
    test_server(url)
