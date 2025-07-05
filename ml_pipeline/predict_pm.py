import numpy as np
import joblib
import pandas as pd
import os
from .helpers import plot_pm_map_enhanced

def predict_pm_maps(reflectance_path, sun_elevation_path, cloud_mask_path, model_dir, output_dir):
    """Generate PM concentration maps from satellite data"""
    print("Generating PM concentration maps...")
    
    reflectance = np.load(reflectance_path)
    sun_elevation = np.load(sun_elevation_path)
    cloud_mask = np.load(cloud_mask_path)
    
    # Apply cloud mask
    reflectance[cloud_mask == 1] = np.nan
    
    # Load models
    pm25_model = joblib.load(os.path.join(model_dir, "pm25_model.joblib"))
    pm10_model = joblib.load(os.path.join(model_dir, "pm10_model.joblib"))
    
    # Prepare features for prediction
    flat_reflectance = reflectance.flatten()
    flat_sun_elev = sun_elevation.flatten()
    
    features = pd.DataFrame({
        'reflectance': flat_reflectance,
        'sun_elevation': flat_sun_elev,
        'rh': np.full_like(flat_reflectance, 50),  # Placeholder
        'temp': np.full_like(flat_reflectance, 30)  # Placeholder
    })
    
    # Feature engineering
    features['reflectance_sq'] = features['reflectance'] ** 2
    features['rh_temp_interaction'] = features['rh'] * features['temp']
    
    # Predict PM concentrations
    print("Predicting PM2.5...")
    pm25_flat = pm25_model.predict(features)
    print("Predicting PM10...")
    pm10_flat = pm10_model.predict(features)
    
    # Reshape to 2D maps
    pm25_map = pm25_flat.reshape(reflectance.shape)
    pm10_map = pm10_flat.reshape(reflectance.shape)
    
    # Apply cloud mask
    pm25_map[cloud_mask == 1] = np.nan
    pm10_map[cloud_mask == 1] = np.nan
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(reflectance_path).replace('_reflectance.npy', '')
    
    np.save(os.path.join(output_dir, f"{base_name}_pm25.npy"), pm25_map)
    np.save(os.path.join(output_dir, f"{base_name}_pm10.npy"), pm10_map)
    
    # Generate visualizations
    plot_pm_map_enhanced(
        pm25_map, "PM2.5 Concentration", "PM2.5",
        save_path=os.path.join(output_dir, f"{base_name}_pm25.png")
    )
    plot_pm_map_enhanced(
        pm10_map, "PM10 Concentration", "PM10",
        save_path=os.path.join(output_dir, f"{base_name}_pm10.png")
    )
    
    # Verification
    print("\nVerifying PM map values:")
    print(f"PM2.5 - Min: {np.nanmin(pm25_map):.2f}, Max: {np.nanmax(pm25_map):.2f}")
    print(f"PM10 - Min: {np.nanmin(pm10_map):.2f}, Max: {np.nanmax(pm10_map):.2f}")
    
    print(f"\nSaved PM maps to {output_dir}")

if __name__ == "__main__":
    reflectance_path = "processed/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1_reflectance.npy"
    sun_elevation_path = "processed/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1_sun_elevation.npy"
    cloud_mask_path = "processed/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1_cloud_mask.npy"
    model_dir = "models/"
    output_dir = "outputs/"
    predict_pm_maps(reflectance_path, sun_elevation_path, cloud_mask_path, model_dir, output_dir)