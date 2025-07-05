import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from .helpers import correct_sun_elevation, radiance_to_reflectance, relative_brightness_cloud_detection

def preprocess_satellite(input_path, output_dir, save_visualizations=True):
    """Process INSAT satellite data"""
    print(f"Processing satellite data: {input_path}")
    
    with h5py.File(input_path, 'r') as f:
        radiance = f['VIS_RADIANCE'][0]
        sun_elevation = f['Sun_Elevation'][0]
        time_val = f['time'][0]
    
    acq_time = datetime(1970, 1, 1) + timedelta(seconds=float(time_val))
    print(f"Acquisition Time: {acq_time}")
    
    sun_elevation = correct_sun_elevation(sun_elevation)
    print(f"Sun Elevation: {np.nanmin(sun_elevation):.1f}° to {np.nanmax(sun_elevation):.1f}°")
    
    reflectance = radiance_to_reflectance(radiance, sun_elevation, acq_time)
    print(f"Reflectance: {np.nanmin(reflectance):.4f} to {np.nanmax(reflectance):.4f}")
    
    cloud_mask = relative_brightness_cloud_detection(reflectance)
    cloud_coverage = np.nanmean(cloud_mask) * 100
    print(f"Cloud Coverage: {cloud_coverage:.1f}%")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).replace('.h5', '')
    
    # Save processed data
    np.save(os.path.join(output_dir, f"{base_name}_reflectance.npy"), reflectance)
    np.save(os.path.join(output_dir, f"{base_name}_sun_elevation.npy"), sun_elevation)
    np.save(os.path.join(output_dir, f"{base_name}_cloud_mask.npy"), cloud_mask)
    
    # Save visualizations
    if save_visualizations:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(radiance, cmap='viridis', vmin=0, vmax=30)
        plt.title('Original Radiance')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(sun_elevation, cmap='coolwarm', vmin=-90, vmax=90)
        plt.title('Corrected Sun Elevation')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(reflectance, cmap='viridis', vmin=0, vmax=1)
        plt.title('Top-of-Atmosphere Reflectance')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.hist(reflectance.flatten(), bins=50, range=(0, 1))
        plt.title('Reflectance Histogram')
        plt.xlabel('Reflectance')
        plt.ylabel('Pixel Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_processing.png"))
        plt.close()
        
        # Cloud mask visualization
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(reflectance, cmap='viridis', vmin=0, vmax=0.1)
        plt.title('Reflectance')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(cloud_mask, cmap='gray')
        plt.title('Cloud Mask (Clouds=White)')
        
        plt.subplot(2, 2, 3)
        cloud_free = reflectance.copy()
        cloud_free[cloud_mask == 1] = np.nan
        plt.imshow(cloud_free, cmap='viridis', vmin=0, vmax=0.1)
        plt.title('Cloud-Free Pixels')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.hist(reflectance.flatten(), bins=50, range=(0, 0.1))
        plt.title('Reflectance Histogram')
        plt.xlabel('Reflectance')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_cloud_masking.png"))
        plt.close()
    
    print(f"Saved processed data to {output_dir}")
    return {
        'reflectance': reflectance,
        'sun_elevation': sun_elevation,
        'cloud_mask': cloud_mask,
        'acq_time': acq_time
    }

if __name__ == "__main__":
    input_file = "data/satellite/3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1.h5"
    output_dir = "processed/satellite/"
    preprocess_satellite(input_file, output_dir)