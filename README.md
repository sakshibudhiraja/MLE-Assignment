# Machine Learning Engineer Assignment

## Overview
This project trains a deep learning model to predict DON (Deoxynivalenol) concentration using a dataset of 448 spectral features. It includes data preprocessing, model training, and an API for predictions.

## Repository Structure
```
MLE-Assignment/
│── data/                          # Data-related files
│── models/                        # Saved models
│── scripts/                        # Python scripts
│   ├── preprocess.py               # Handles data preprocessing
│   ├── train.py                    # Model training script
│   ├── predict.py                   # API endpoint
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation
│── .gitignore                       # Ignore files
```

## Dataset
- **File:** `MLE-Assignment.csv`
- **Features:** 448 numerical spectral reflectance values.
- **Target Variable:** `DON_Concentration` (Vomitoxin level).

## Installation
### **Clone the Repository**
```bash
git clone https://github.com/yourusername/MLE-Assignment.git
cd MLE-Assignment
```
### **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Data Pre-processing
```bash
python3 scripts/preprocess.py
```

## Model Training
Train the model using:
```bash
python3 scripts/train.py
```
This will:
- Preprocess the data
- Train a deep learning model
- Save `model.keras` and `scaler.pkl`

## Run the API Server
Start the Flask API:
```bash
python3 scripts/predict.py
```
API will be available at:  
**http://127.0.0.1:2222/predict**
