# week_3_mls/model_building/train.py
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps__visit_us_prediction")

api = HfApi()


Xtrain_path = "hf://datasets/praneeth232/machine-failure-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/praneeth232/machine-failure-prediction/Xtest.csv"
ytrain_path = "hf://datasets/praneeth232/machine-failure-prediction/ytrain.csv"
ytest_path = "hf://datasets/praneeth232/machine-failure-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


num_feat = Xtrain.select_dtypes(include=[np.number]).columns.tolist()
cat_feat=Xtrain.select_dtypes(include=['object']).columns.tolist()

# Encoding the categorical categorical column
label_encoder = LabelEncoder()
for col in cat_feat:
  Xtrain[col] = label_encoder.fit_transform(Xtrain[col].astype(str))


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), num_feat),
    (OneHotEncoder(handle_unknown='ignore'), cat_feat)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model=grid_search.best_estimator_

y_pred=best_model.predict(Xtest)
#y_pred_proba=best_model.predict_proba(Xtest)[:,1]

acc=accuracy_score(ytest, y_pred)
f1=f1_score(ytest, y_pred)
rec=recall_score(ytest, y_pred)
pre=precision_score(ytest, y_pred)



# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    
    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("f1-score",f1)
    mlflow.log_metric("recall",rec)
    mlflow.log_metric("precision",pre)

    mlflow.sklearn.log_model(best_model,"xgb_best_model")


    # Save the model locally
    model_path = "best_visit_us_prediction_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "sudhirpgcmma02/visit-us-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_visit_us_prediction_v1.joblib", # tourism_project
        path_in_repo="best_visit_us_prediction_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
