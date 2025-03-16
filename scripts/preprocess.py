import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess Data
def preprocess_data(df):
    df = df.dropna()
    
    print("Dataset Info:", df.info())
    
    df = df.select_dtypes(include=[np.number])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print("Number of Features in Training Data:", X.shape[1])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Save Preprocessor
def save_scaler(scaler, path="models/scaler.pkl"):
    joblib.dump(scaler, path)

# Load Preprocessor
def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path) if os.path.exists(path) else None
