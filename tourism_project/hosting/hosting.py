from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

files = ["app.py","data_register.py","Dockerfile","hosting.py","prep.py","requirements.txt","train.py",
         "tourism.csv","model.pkl","MLmodel","python_env.yaml","best_visit_us_prediction_v1.joblib"]

for f in files:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="sudhirpgcmma02/visit-us-prediction",  # the target repo
        repo_type="space",  # dataset, model, or space
    )
