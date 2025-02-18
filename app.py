from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

#uvicorn app:app --reload
# 743f1fbe1cb54518b2ea372e307f27b7
# Chargement du modèle mlflow
model = mlflow.pyfunc.load_model("models:/credit_scoring/latest")

# IMPORTANT :
# L'encodeur doit avoir été entraîné sur les colonnes suivantes :
# ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'age_group', 'loan_amnt_group', 'income_group']
# Si ce n'est pas le cas, ré-entraînez-le sur votre jeu de données d'entraînement après avoir créé ces colonnes.
encoder = joblib.load("models/encoder.pkl")
expected_columns = joblib.load("models/expected_columns.pkl")

# Définition du modèle d'entrée
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
    # Création des colonnes catégorielles calculées
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
    
    # Liste des colonnes catégorielles à encoder (les colonnes calculées DOIVENT être présentes lors du fit)
    categorical_columns = [
        'person_home_ownership',
        'loan_intent',
        'cb_person_default_on_file',
        'age_group',
        'loan_amnt_group',
        'income_group'
    ]
    
    # Encodage one-hot avec l'encodeur pré-entraîné
    encoded_array = encoder.transform(data[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=data.index
    )
    
    # On retire les colonnes catégorielles d'origine et on concatène les colonnes encodées
    data = data.drop(columns=categorical_columns)
    data = pd.concat([data, encoded_df], axis=1)
    
    # Réordonner les colonnes selon l'ordre attendu par le modèle
    data = data.reindex(columns=expected_columns, fill_value=0)
    
    return data

# Création de l'application FastAPI
app = FastAPI()

@app.post("/predict/")
def predict(input_data: InputData):
    try:
        # Conversion des données d'entrée en DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Calcul des caractéristiques dérivées numériques
        data['loan_to_income'] = data['loan_amnt'] / data['person_income']
        data['loan_to_emp_length_ratio'] = data['person_emp_length'] / data['loan_amnt']
        data['int_rate_to_loan_amnt_ratio'] = data['loan_int_rate'] / data['loan_amnt']
        
        # Prétraitement : création des colonnes calculées, encodage et réordonnancement
        data = preprocess_data(data, encoder, expected_columns)
        
        # Prédiction avec le modèle
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {str(e)}")
