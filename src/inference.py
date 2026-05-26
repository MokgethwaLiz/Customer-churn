import joblib
import pandas as pd

from pathlib import Path

THRESHOLD = 0.25

# Paths

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (BASE_DIR.parent / "artifacts" / "stacking_churn_model.pkl")

FEATURE_PATH = (BASE_DIR.parent/ "artifacts"/ "selected_features.pkl")

# Load trained model

model = joblib.load(MODEL_PATH)

# Load selected features
selected_features = joblib.load(
    FEATURE_PATH
)

# Preprocessing function

def preprocess_new_data(data):


    # Convert dictionary to dataframe
    df = pd.DataFrame([data])

    # Basic preprocessing

    # Convert TotalCharges
    if 'TotalCharges' in df.columns:

        df['TotalCharges'] = (
            pd.to_numeric(
                df['TotalCharges'],
                errors='coerce'
            )
            .fillna(0)
        )

    # Convert SeniorCitizen
    if 'SeniorCitizen' in df.columns:

        df['SeniorCitizen'] = (
            df['SeniorCitizen']
            .fillna(0)
            .astype(int)
        )
    
    for col in df.select_dtypes(include='object').columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )


    # Binary encoding

    binary_mappings = {

        'gender': {
            'Female': 0,
            'Male': 1
        },

        'Partner': {
            'No': 0,
            'Yes': 1
        },

        'Dependents': {
            'No': 0,
            'Yes': 1
        },

        'PhoneService': {
            'No': 0,
            'Yes': 1
        },

        'PaperlessBilling': {
            'No': 0,
            'Yes': 1
        }
    }

    # Apply mappings
    for col, mapping in binary_mappings.items():

        if col in df.columns:

            df[col] = df[col].map(mapping).fillna(0)

    # Multi-class columns

    multi_class_cols = [

        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=multi_class_cols,
        drop_first=True,
        dtype=int
    )

    # Align columns with training data

    # Add missing columns
    for col in selected_features:

        if col not in df.columns:

            df[col] = 0

    # Keep only training columns
    df = df[selected_features]

    # Ensure numeric
    df = df.astype(float)

    return df

# Prediction function

def predict(data):

    # Preprocess incoming data
    processed_data = preprocess_new_data(
        data
    )

    print(processed_data)
    print(processed_data.isna().sum())
    print(processed_data[processed_data.isna().any(axis=1)])

    # Generate probability
    probability = model.predict_proba(
        processed_data
    )[0][1]

    # Apply threshold
    prediction = int(
        probability >= THRESHOLD
    )

    return {

        'prediction': prediction,

        'probability': round(
            float(probability),
            4
        )
    }

