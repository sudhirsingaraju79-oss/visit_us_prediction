from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "sudhirpgcmma02/visit-us-prediction"
repo_type = "dataset"



# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

local_file ="tourism.csv"
folder_path="data/tourism.csv"

api.upload_folder(
    path_or_fileobj=local_file,
    path_in_repo=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
