from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

#uvicorn app:app --reload
# 743f1fbe1cb54518b2ea372e307f27b7
# Loading the mlflow model
model = mlflow.pyfunc.load_model("models:/credit_scoring/latest")

# IMPORTANT:
# The encoder must have been trained on the following columns:
# ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'age_group', 'loan_amnt_group', 'income_group']
# If this is not the case, retrain it on your training dataset after creating these columns.
encoder = joblib.load("models/encoder.pkl")
expected_columns = joblib.load("models/expected_columns.pkl")

# Defining the input model
class InputData(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float

def preprocess_data(data: pd.DataFrame, encoder: OneHotEncoder, expected_columns: list) -> pd.DataFrame:
    # Creating calculated categorical columns
    age_order = ["18-30", "30-40", "40-50", "50-60", "60+"]
    data['age_group'] = pd.cut(
        data['person_age'],
        bins=[18, 30, 40, 50, 60, float('inf')],
        labels=age_order,
        right=False
    )
    
    loan_amnt_order = ['small', 'medium', 'large', 'very large']
    data['loan_amnt_group'] = pd.cut(
        data['loan_amnt'],
        bins=[0, 5000, 10000, 15000, float('inf')],
        labels=loan_amnt_order,
        right=True
    )
    
    income_order = ['low', 'low-middle', 'middle', 'high-middle', 'high']
    data['income_group'] = pd.cut(
        data['person_income'],
        bins=[0, 20000, 50000, 100000, 200000, float('inf')],
        labels=income_order,
        right=False
    )
    
    # List of categorical columns to encode (the calculated columns MUST be present during the fit)
    categorical_columns = [
        'person_home_ownership',
        'loan_intent',
        'cb_person_default_on_file',
        'age_group',
        'loan_amnt_group',
        'income_group'
    ]
    
    # One-hot encoding with the pre-trained encoder
    encoded_array = encoder.transform(data[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=data.index
    )
    
    # Remove the original categorical columns and concatenate the encoded columns
    data = data.drop(columns=categorical_columns)
    data = pd.concat([data, encoded_df], axis=1)
    
    # Reorder the columns according to the expected order by the model
    data = data.reindex(columns=expected_columns, fill_value=0)
    
    return data

# Creating the FastAPI application
app = FastAPI()

@app.post("/predict/")
def predict(input_data: InputData):
    try:
        # Convert the input data into a DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Calculate derived numerical features
        data['loan_to_income'] = data['loan_amnt'] / data['person_income']
        data['loan_to_emp_length_ratio'] = data['person_emp_length'] / data['loan_amnt']
        data['int_rate_to_loan_amnt_ratio'] = data['loan_int_rate'] / data['loan_amnt']
        
        # Preprocessing: creating calculated columns, encoding, and reordering
        data = preprocess_data(data, encoder, expected_columns)
        
        # Prediction with the model
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
