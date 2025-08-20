from huggingface_hub import snapshot_download

if __name__ == "__main__":
    data_path = snapshot_download(repo_id='BLIP3o/BLIP3o-60k', repo_type='dataset')
    print(data_path)
