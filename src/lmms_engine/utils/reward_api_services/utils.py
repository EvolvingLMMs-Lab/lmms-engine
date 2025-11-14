import pickle
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Union

def serialize_images(images: List[Image.Image]) -> List[bytes]:
    images_bytes = []
    for img in images:
        img_byte_arr = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
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