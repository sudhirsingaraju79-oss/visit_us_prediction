from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

files = ["app.py","data_register.py","Dockerfile","hosting.py","prep.py","requirements.txt","train.py","tourism.csv"]

for f in files:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="sudhirpgcmma02/visit-us-prediction",  # the target repo
        repo_type="space",  # dataset, model, or space
    )

"""
api.upload_folder(
    
    
    
    folder_path="./tourism_project",     # the local folder containing your files
    repo_id="sudhirpgcmma02/visit-us-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
"""