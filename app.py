import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import uuid
import os
# Download and load the model
model_path = hf_hub_download(repo_id="sudhirpgcmma02/visit-us-prediction", filename="best_visit_us_prediction_v1.joblib")
model = joblib.load(model_path)
data_path = "hf://datasets/sudhirpgcmma02/visit-us-prediction/tourism.csv"


# ---------------------------------------------
# Helper Function: Auto-generate CustomerID
# ---------------------------------------------

def get_next_customer_id(data_path="hf://datasets/sudhirpgcmma02/visit-us-prediction/tourism.csv"):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if "CustomerID" in df.columns and len(df) > 0:
            return len(df) + 1
    return 1

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.title("Visit With Us - Cutomer Purchase Prediction  APP")

st.subheader("Enter Customer Details")

# Auto-generate CustomerID
customer_id = get_next_customer_id()
st.info(f"Auto-generated CustomerID: {customer_id}")

# ---------------------------------------------
# Input Form
# ---------------------------------------------
with st.form("customer_form"):
    #ProdTaken = st.selectbox("Product Taken (Target)", [0, 1])
    Age = st.number_input("Age", min_value=0, max_value=120, step=1)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=20, step=1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number Of Trips Per Year", min_value=0, max_value=50)
    Passport = st.selectbox("Passport", [0, 1])
    OwnCar = st.selectbox("Own Car", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=10)
    Designation = st.text_input("Designation")
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, step=1000)

    st.subheader("Customer Interaction Data")
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
    NumberOfFollowups = st.number_input("Number Of Follow-ups", min_value=0, max_value=20)
    DurationOfPitch = st.number_input("Duration Of Pitch (Minutes)", min_value=0, max_value=120)

    submitted = st.form_submit_button("Save Customer Data")

# ---------------------------------------------
# Save Data
# ---------------------------------------------
    if submitted:
        new_row = {
            "CustomerID": customer_id,
            #"ProdTaken": ProdTaken,
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": Passport,
            "OwnCar": OwnCar,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome,
            "PitchSatisfactionScore": PitchSatisfactionScore,
            "ProductPitched": ProductPitched,
            "NumberOfFollowups": NumberOfFollowups,
            "DurationOfPitch": DurationOfPitch
        }

        data_path="hf://datasets/sudhirpgcmma02/visit-us-prediction/tourism.csv"

        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df = df.append(new_row, ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(data_path, index=False)
        st.success(f"Customer data saved successfully with CustomerID {customer_id}!")

        st.subheader("Predict Purchase (Model Inference)")


        model_path = "model.pkl"

          #  if os.path.exists(model_path):
        import pickle
        model = pickle.load(open(model_path, "rb"))
        st.info("Model Loaded. Enter details for prediction:")


        predict_button = st.form_submit_button("Predict")
        if predict_button:
            pred_df = pd.DataFrame(new_row)
            pred_prob = model.predict_proba(pred_df)[0][1]
            pred_label = model.predict(pred_df)[0]

            st.success(f"Prediction: {'Will Purchase (1)' if pred_label==1 else 'Will Not Purchase (0)'}")
            st.write(f"Probability of Purchase: {pred_prob:.2f}")
        else:
            st.warning("Model file not found. Upload model.pkl to enable prediction.")
    else:
        st.warning("Thank your for our time")

# ---------------------------------------------
# Show Existing Data
# ---------------------------------------------
    st.subheader("Stored Customer Data")
    if os.path.exists(".csv"):
        stored_df = pd.read_csv("hf://datasets/sudhirpgcmma02/visit-us-prediction/tourism.csv")
        st.dataframe(stored_df)
    else:
        st.info("No data stored yet.")

    # If you want to trigger training from the UI (optional)
    train_button = st.sidebar.button("Trigger Training Locally (calls src/train.py)")
    if train_button:
        st.info("Triggering training script — this runs on the machine hosting Streamlit.")
        import subprocess
        # Use a small dataset path; adapt as needed
        res = subprocess.run(["python", "train.py", "--data-csv", "trourism.csv", "--out-dir", "tourism_project/data/"], capture_output=True, text=True)
        st.text(res.stdout)
        if res.returncode != 0:
            st.error(res.stderr)
        else:
            st.success("Training finished")
