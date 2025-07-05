import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib
import os

def train_pm_models(aligned_csv, output_dir):
    """Train PM2.5 and PM10 prediction models"""
    print("Training PM prediction models...")
    
    data = pd.read_csv(aligned_csv)
    
    # Feature engineering
    data['reflectance_sq'] = data['reflectance'] ** 2
    data['rh_temp_interaction'] = data['rh'] * data['temp']
    
    # Prepare features and targets
    features = data[['reflectance', 'sun_elevation', 'rh', 'temp', 'reflectance_sq', 'rh_temp_interaction']]
    pm25_target = data['pm25']
    pm10_target = data['pm10']
    
    # Train models
    pm25_model = RandomForestRegressor(n_estimators=500, max_depth=8, 
                                      min_samples_split=5, max_features=0.8,
                                      random_state=42, n_jobs=-1)
    pm10_model = RandomForestRegressor(n_estimators=500, max_depth=8, 
                                      min_samples_split=5, max_features=0.8,
                                      random_state=42, n_jobs=-1)
    
    pm25_model.fit(features, pm25_target)
    pm10_model.fit(features, pm10_target)
    
    # Save models
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pm25_model, os.path.join(output_dir, "pm25_model.joblib"))
    joblib.dump(pm10_model, os.path.join(output_dir, "pm10_model.joblib"))
    
    # Evaluate and visualize
    def evaluate_model(model, X, y, pollutant):
        y_pred = model.predict(X)
        mae = np.mean(np.abs(y - y_pred))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, s=150, alpha=0.7)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'g-', lw=2)
        plt.title(f'{pollutant} Prediction - MAE: {mae:.2f} µg/m³')
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{pollutant}_prediction.png"))
        plt.close()
        
        # Feature importance
        if len(X) > 1:
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            sorted_idx = result.importances_mean.argsort()
            
            plt.figure(figsize=(10, 6))
            plt.boxplot(result.importances[sorted_idx].T,
                        vert=False, labels=X.columns[sorted_idx])
            plt.title(f"{pollutant} Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pollutant}_feature_importance.png"))
            plt.close()
        else:
            print(f"Skipping feature importance for {pollutant} - requires multiple samples")
        
        return mae
    
    print("\nPM2.5 Model:")
    mae_pm25 = evaluate_model(pm25_model, features, pm25_target, 'PM2.5')
    
    print("\nPM10 Model:")
    mae_pm10 = evaluate_model(pm10_model, features, pm10_target, 'PM10')
    
    print(f"\nModels saved to {output_dir}")
    return pm25_model, pm10_model

if __name__ == "__main__":
    aligned_csv = "processed/aligned/aligned_data.csv"
    output_dir = "models/"
    train_pm_models(aligned_csv, output_dir)