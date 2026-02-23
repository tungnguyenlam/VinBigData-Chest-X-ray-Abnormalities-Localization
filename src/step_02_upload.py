from huggingface_hub import HfApi

repo_id = "your_username/vinbigdata-preprocessed"  # Replace with user's HF username
print(f"Uploading data/vinbigdata_processed to HF hub: {repo_id}")

api = HfApi()

# create repo if not exists
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Error creating repo: {e}")

api.upload_folder(
    folder_path="data/vinbigdata_processed",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo=".",
)

print("Upload complete!")
