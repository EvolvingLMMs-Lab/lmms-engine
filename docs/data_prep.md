# Data Preparation

## Recommend Data Format
The recommended way to prepare the data is through yaml file and prepare your sft data in jsonlines. You need to prepare a YAML file to specify the data path and data type. The YAML file should look like this:

```yaml
datasets:
- path: <path to the json/jsonl file>
  data_folder: <path to the data folder>
  data_type: json/jsonl
- path: <path to the json/jsonl file>
  data_folder: <path to the data folder>
  data_type: json/jsonl
...
```

Below is an example script which may help you easier to prepare the YAML file. This script will download the dataset from the Hugging Face Hub and extract the dataset from the `tar.gz` file. You can modify the script to fit your needs.

```python
import tarfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ID = "Evo-LMM/rlaif-v"
DATA_TYPE = "json"
DATA_PATH = "train.json"
OUTPUT_YAML = "data.yaml"

if __name__ == "__main__":
    data_path = snapshot_download(
        repo_id="Evo-LMM/rlaif-v",
        repo_type="dataset",
    )
    data_path = Path(data_path)
    for image_zip in data_path.glob("*.tar.gz"):
        with tarfile.open(image_zip, "r:gz") as tar:
            tar.extractall(path=data_path)
    data_dict = {
        "datasets": [
            {
                "path": str(data_path / DATA_PATH),
                "data_folder": str(data_path),
                "data_type": DATA_TYPE,
            }
        ]
    }
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(data_dict, f)
```

### Cloud Data Access
With the data scaling, it might be very redundant to download and extract all the data to your local storage (and unrealistic). A way to cope this is through object storage. The training framework now supports using `google cloud storage` and `azure blob storage` to access the data file directly. To use it, you should specify in your training config that

```json
{
    "dataset_config": {
                ...
                "object_storage": "azure", # Or gcs
                "bucket_name": "llava",
                ...
    }
}
```

Then the data folder should be the path to the data folder on the cloud storage. You should export the credentials before running the application

```bash
export GOOGLE_APPLICATION_CREDENTIALS="<YOUR CRED>"
export AZURE_STORAGE_SAS_URL="<YOUR SAS URL>"
```

Please contact the adminstrator to get your credential


## HF Format

In our initial code design, we also integrated the huggingface format. But since we believe it is currently relatively hard to scale using this format. This format has mainly been deprecated and not under maintainence.
