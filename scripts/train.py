import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_data, preprocess_data, save_scaler

# Build Model
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train Model
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=16):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }, y_pred

# Save Model
def save_model(model, path="models/model.keras"):
    model.save(path)

if __name__ == "__main__":
    df = load_data("data/MLE-Assignment.csv")
    X, y, scaler = preprocess_data(df)
    save_scaler(scaler)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(X_train.shape[1])
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    results, _ = evaluate_model(model, X_test, y_test)
    print("Model Performance:", results)
    
    save_model(model)
    print("Training Completed. Model saved successfully.")
