import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter, binary_dilation, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt

# Constants
SOLAR_IRRADIANCE = 1362.0  # W/m²/μm (INSAT-3D VIS band)

def correct_sun_elevation(sun_elev):
    """Fix sun elevation data issues"""
    sun_elev = sun_elev.astype(np.float32)
    invalid_mask = np.abs(sun_elev) > 1000
    print(f"Invalid values found: {np.sum(invalid_mask)}")
    
    if np.nanmax(sun_elev) > 360:  # 3600 tenths = 360°
        sun_elev /= 10.0
    
    sun_elev = np.clip(sun_elev, -90, 90)
    sun_elev[invalid_mask] = np.nan
    return sun_elev

def radiance_to_reflectance(radiance, sun_elev, acq_date):
    """Convert radiance to top-of-atmosphere reflectance"""
    day_of_year = acq_date.timetuple().tm_yday
    d_au = 1 - 0.01672 * np.cos(np.radians(0.9856 * (day_of_year - 4)))
    sza = 90 - sun_elev
    
    with np.errstate(divide='ignore', invalid='ignore'):
        reflectance = (np.pi * radiance * d_au**2) / (SOLAR_IRRADIANCE * np.cos(np.radians(sza)))
    
    reflectance[sun_elev < 5] = np.nan
    reflectance[reflectance < 0] = np.nan
    reflectance[reflectance > 1] = np.nan
    return reflectance

def relative_brightness_cloud_detection(reflectance):
    """Cloud detection based on relative brightness"""
    valid_mask = ~np.isnan(reflectance)
    cloud_mask = np.zeros(reflectance.shape, dtype=np.uint8)
    
    if not np.any(valid_mask):
        return cloud_mask
    
    refl_values = reflectance[valid_mask].flatten()
    p95 = np.percentile(refl_values, 95)
    p75 = np.percentile(refl_values, 75)
    
    bright_mask = reflectance > p75
    smooth_refl = gaussian_filter(reflectance, sigma=5)
    texture = np.abs(reflectance - smooth_refl)
    homogeneous_bright = bright_mask & (texture < 0.01)
    
    cloud_mask[homogeneous_bright & valid_mask] = 1
    cloud_mask = binary_dilation(cloud_mask, structure=disk(3))
    cloud_mask = binary_opening(cloud_mask, structure=disk(3))
    
    return cloud_mask.astype(np.uint8)

def plot_pm_map_enhanced(data, title, pollutant, save_path=None):
    """Visualization for PM maps"""
    plt.figure(figsize=(12, 10))
    img = plt.imshow(data, cmap='RdYlGn_r', 
                    vmin=0, vmax=100 if pollutant == 'PM2.5' else 200,
                    aspect='auto', extent=[68, 97, 8, 37])
    
    cbar = plt.colorbar(img, shrink=0.7)
    cbar.set_label('µg/m³', rotation=270, labelpad=20)
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.plot([68, 97, 97, 68, 68], [8, 8, 37, 37, 8], 'k-', lw=1)
    
    if np.nanmin(data) == np.nanmax(data):
        value = np.nanmean(data)
        plt.text(82.5, 22.5, f"Constant {pollutant}\nConcentration: {value:.1f} µg/m³",
                 fontsize=14, ha='center', va='center', 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()