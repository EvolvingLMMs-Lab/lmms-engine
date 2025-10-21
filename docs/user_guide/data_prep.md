# Data Preparation

## Recommend Data Format
The recommended way to prepare the data is through yaml file and prepare your sft data in jsonlines/parquet/arrow. You need to prepare a YAML file to specify the data path and data type. The YAML file should look like this:

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

The actual dataset format can refer to the debug dataset we provide on huggingface or refer to the protocol files in `src/lmms_engine/protocol/data_proto.py`

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
