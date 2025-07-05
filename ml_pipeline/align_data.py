import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt

def align_ground_satellite(ground_csv, reflectance_path, sun_elevation_path, output_dir):
    """Align ground station measurements with satellite data"""
    print("Aligning ground and satellite data...")
    
    reflectance = np.load(reflectance_path)
    sun_elevation = np.load(sun_elevation_path)
    ground = pd.read_csv(ground_csv)
    
    # Rename columns if needed
    if 'latitude' in ground.columns and 'longitude' in ground.columns:
        ground = ground.rename(columns={
            'station_id': 'StationID',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        })
    
    # Create approximate grid for India
    height, width = reflectance.shape
    lat_grid = np.linspace(37, 8, height)
    lon_grid = np.linspace(68, 97, width)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Prepare satellite data for matching
    valid_mask = ~np.isnan(reflectance)
    valid_indices = np.where(valid_mask)
    valid_lons = lon_mesh[valid_indices]
    valid_lats = lat_mesh[valid_indices]
    valid_reflectance = reflectance[valid_indices]
    valid_sun_elev = sun_elevation[valid_indices]
    
    # Build KDTree for spatial matching
    sat_tree = cKDTree(np.column_stack([valid_lons, valid_lats]))
    
    # Align each ground station
    aligned_data = []
    station_coords = ground[['Longitude', 'Latitude']].values
    
    distances, indices = sat_tree.query(station_coords, k=1)
    
    for i, row in ground.iterrows():
        sat_idx = indices[i]
        
        aligned_data.append({
            'station_id': row.get('StationID', f'station_{i}'),
            'station_lat': row['Latitude'],
            'station_lon': row['Longitude'],
            'sat_lat': valid_lats[sat_idx],
            'sat_lon': valid_lons[sat_idx],
            'distance_km': distances[i] * 111,
            'reflectance': valid_reflectance[sat_idx],
            'sun_elevation': valid_sun_elev[sat_idx],
            'pm25': row['PM2.5 (µg/m³)'],
            'pm10': row['PM10 (µg/m³)'],
            'rh': row['RH (%)'],
            'temp': row['AT (°C)']
        })
    
    # Save aligned data
    aligned_df = pd.DataFrame(aligned_data)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aligned_data.csv')
    aligned_df.to_csv(output_path, index=False)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    plt.scatter(lon_mesh, lat_mesh, c=reflectance, s=1, alpha=0.3, cmap='viridis', vmin=0, vmax=0.1)
    plt.scatter(aligned_df['station_lon'], aligned_df['station_lat'], c='red', s=30, label='Stations')
    plt.colorbar(label='Reflectance')
    plt.title('India: Satellite Data & Air Quality Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(68, 97)
    plt.ylim(8, 37)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'station_alignment.png'))
    plt.close()
    
    print(f"Saved aligned data: {output_path}")
    return aligned_df

if __name__ == "__main__":
    ground_csv = "data/ground/cpcb_data.csv"
    reflectance_path = "processed/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1_reflectance.npy"
    sun_elevation_path = "processed/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1_sun_elevation.npy"
    output_dir = "processed/aligned/"
    align_ground_satellite(ground_csv, reflectance_path, sun_elevation_path, output_dir)