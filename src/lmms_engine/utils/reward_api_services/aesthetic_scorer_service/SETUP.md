# Aesthetic Scorer Service Setup Guide

## Overview
This service provides aesthetic scoring for images using CLIP and a trained MLP model.

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Model weights file: `sac+logos+ava1-l14-linearMSE.pth`

## Installation

### 1. Install Dependencies
```bash
cd /mnt/raid10/boli/UniRL/rewards_services/api_services/aesthetic_scorer_service
pip install -r requirements.txt
```

### 2. Verify Model Weights
Ensure the model weights file exists:
```bash
ls sac+logos+ava1-l14-linearMSE.pth
```

## Running the Service

### Option 1: Direct Launch (Current Environment)
```bash
bash run.sh
```

### Option 2: With Conda Environment
```bash
bash run.sh --conda-env aes
```

The service will start on `http://0.0.0.0:18080`

## Stopping the Service

### Method 1: If running in foreground
Press `Ctrl+C`

### Method 2: If running in background
```bash
pkill -f 'gunicorn.*aesthetic_scorer_service'
```

## Testing the Service

```python
import requests
import pickle
from PIL import Image

# Prepare test data
images = [Image.open("test.jpg")]
payload = create_payload(images, prompts=["test prompt"])

# Send request
response = requests.post("http://localhost:18080/", data=payload)
result = pickle.loads(response.content)
print(f"Aesthetic scores: {result['scores']}")
```

## Configuration

Edit `gunicorn.conf.py` to adjust:
- `NUM_DEVICES`: Total number of GPUs available (default: 8)
- `USED_DEVICES`: GPUs to use for workers (default: 0-5)
- `port`: Service port (default: 18080)
- `workers`: Number of worker processes (default: NUM_DEVICES)
- `timeout`: Request timeout in seconds (default: 300)

## Troubleshooting

### Error: "gunicorn: command not found"
Install gunicorn:
```bash
pip install gunicorn
```

### Error: "Model weights file not found"
Download or copy the model weights to the service directory:
```bash
cp /path/to/sac+logos+ava1-l14-linearMSE.pth .
```

### Error: "Missing required Python packages"
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Service won't start on port 18080
Check if port is already in use:
```bash
lsof -i :18080
```
Kill the existing process or change the port in `gunicorn.conf.py`

## GPU Assignment

The service automatically distributes workers across GPUs based on `USED_DEVICES` in `gunicorn.conf.py`. Each worker gets assigned a unique GPU via `CUDA_VISIBLE_DEVICES`.

## API Reference

### POST /
**Request:**
- Content-Type: application/octet-stream (pickled payload)
- Payload structure:
  ```python
  {
      "images": List[bytes],  # Serialized JPEG images
      "prompts": List[str],   # Optional text prompts
      "metadata": Dict        # Optional metadata
  }
  ```

**Response:**
- Content-Type: application/octet-stream (pickled response)
- Success (200):
  ```python
  {
      "scores": List[float]  # Aesthetic scores for each image
  }
  ```
- Error (500):
  ```python
  {
      "error": str  # Error message with traceback
  }
  ```
