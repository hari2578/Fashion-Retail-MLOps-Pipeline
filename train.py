import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import argparse
import joblib

def start_model_training():
    # 1. SETUP ARGUMENTS
    parser = argparse.ArgumentParser()
    # SageMaker automatically sets these environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()

    # 2. LOAD DATA
    # Aligned with the filename from preprocess.py
    inventory_path = os.path.join(args.train, "cleaned_fashion_data.csv")
    print(f"Loading data from: {inventory_path}")
    
    fashion_data = pd.read_csv(inventory_path)

    # 3. FEATURE SELECTION
    # Using 'Surge_Flag' for demand surges and 'Engagement_Rank' for personalization
    # Adjust 'Quantity_Sold' if your CSV column has a different name (e.g., 'Sales')
    target_column = 'Quantity_Sold' if 'Quantity_Sold' in fashion_data.columns else fashion_data.columns[-1]
    
    features = fashion_data[['Surge_Flag', 'Engagement_Rank']]
    target = fashion_data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # 4. MODEL TRAINING
    demand_forecaster = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        learning_rate=0.1
    )
    
    print("Beginning training for the Fashion Demand Forecasting model...")
    demand_forecaster.fit(x_train, y_train)

    # 5. EVALUATION
    predictions = demand_forecaster.predict(x_test)
    error_score = mean_absolute_error(y_test, predictions)
    print(f"Training Complete. Validation MAE: {error_score}")

    # 6. SAVE THE MODEL
    # SageMaker looks for 'model.tar.gz' eventually, so we save the joblib file here
    model_output_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(demand_forecaster, model_output_path)
    print(f"Model successfully saved to {model_output_path}")

if __name__ == "__main__":
    start_model_training()