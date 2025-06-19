import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import (
    get_winds_aloft_table, get_wind_component_interpolators, get_sat_image,
    meters_to_latlon, simulate_freefall_and_canopy
)

# -------------------- INITIAL CONDITIONS & CONSTANTS --------------------
# Location (Dropzone)
IPLat = 39.7065614
IPLong = -75.0352181

# Skydiver/Simulation parameters
EXIT_ALTITUDE_FT = 13000      # Exit altitude (ft)
DEPLOY_ALTITUDE_FT = 3000     # Canopy deployment altitude (ft)
MASS_KG = 90                  # Skydiver mass (kg)
CDA = 0.505                   # Drag area (m^2)
CANOPY_V_VERT_FPS = 8        # Canopy vertical descent rate (ft/s)
CANOPY_V_HORIZ_FPS = 24       # Canopy horizontal speed (ft/s) for glide circle
DT = 0.1                      # Time step (s)
SAT_IMG_ZOOM = 13             # Satellite image zoom level
SAT_IMG_SIZE = 400            # Satellite image size (pixels)
CIRCLE_RESOLUTION = 200       # Number of points for glide circle
# -----------------------------------------------------------------------

def meters_offset_to_latlon(north_offset, east_offset, lat0, lon0):
    dlat = north_offset / 111320
    dlon = east_offset / (111320 * np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon

def main():
    # Pull Winds
    RawWinds = get_winds_aloft_table(IPLat, IPLong)
    print(RawWinds)
    north_interp, east_interp = get_wind_component_interpolators(RawWinds)
    
    # First simulation: exit directly over target
    alts, norths, easts, times, phases = simulate_freefall_and_canopy(
        alt0_ft=EXIT_ALTITUDE_FT,
        mass_kg=MASS_KG,
        CdA=CDA,
        north_interp=north_interp,
        east_interp=east_interp,
        deploy_alt_ft=DEPLOY_ALTITUDE_FT,
        canopy_v_vert_fps=CANOPY_V_VERT_FPS,
        dt=DT
    )
    
    # Calculate required exit offset to land at IPLat/IPLong
    final_north = norths[-1]
    final_east = easts[-1]
    required_north_offset = -final_north
    required_east_offset = -final_east
    
    exit_lat, exit_lon = meters_offset_to_latlon(required_north_offset, required_east_offset, IPLat, IPLong)
    print(f"Exit at: {exit_lat}, {exit_lon}")
    
    # Rerun simulation from the new exit point
    alts, norths, easts, times, phases = simulate_freefall_and_canopy(
        alt0_ft=EXIT_ALTITUDE_FT,
        mass_kg=MASS_KG,
        CdA=CDA,
        north_interp=north_interp,
        east_interp=east_interp,
        deploy_alt_ft=DEPLOY_ALTITUDE_FT,
        canopy_v_vert_fps=CANOPY_V_VERT_FPS,
        dt=DT,
        north0=required_north_offset,
        east0=required_east_offset
    )
    
    # Get satellite image and bounding box
    img, (lat_min, lat_max, lon_min, lon_max) = get_sat_image(IPLat, IPLong, zoom=SAT_IMG_ZOOM, size=SAT_IMG_SIZE)
    
    # Convert trajectory to lat/lon
    traj_lat, traj_lon = meters_to_latlon(norths, easts, IPLat, IPLong)
    
    # Calculate canopy glide distance (in meters)
    canopy_v_vert_mps = CANOPY_V_VERT_FPS * 0.3048
    canopy_v_horiz_mps = CANOPY_V_HORIZ_FPS * 0.3048
    deploy_alt_m = DEPLOY_ALTITUDE_FT * 0.3048
    
    # Time under canopy (seconds)
    canopy_time = deploy_alt_m / canopy_v_vert_mps
    # Glide distance (meters)
    glide_distance = canopy_v_horiz_mps * canopy_time
    
    # Generate circle points around exit location
    theta = np.linspace(0, 2 * np.pi, CIRCLE_RESOLUTION)
    circle_north = glide_distance * np.cos(theta)
    circle_east = glide_distance * np.sin(theta)
    circle_lat, circle_lon = meters_to_latlon(circle_north + required_north_offset, circle_east + required_east_offset, IPLat, IPLong)
    
    # Convert trajectory and circle to miles relative to IP
    norths_miles = norths / 1609.34
    easts_miles = easts / 1609.34
    circle_north_miles = circle_north + required_north_offset
    circle_east_miles = circle_east + required_east_offset
    circle_north_miles /= 1609.34
    circle_east_miles /= 1609.34
    exit_north_miles = required_north_offset / 1609.34
    exit_east_miles = required_east_offset / 1609.34
    
    # For satellite image overlay: calculate image bounds in miles relative to IP
    if img is not None and None not in (lat_min, lat_max, lon_min, lon_max):
        # Calculate north/east offsets (in meters) from IP to image bounds
        def latlon_to_offset(lat, lon, lat0, lon0):
            dlat = (lat - lat0) * 111320
            dlon = (lon - lon0) * (40075000 * np.cos(np.radians(lat0)) / 360)
            return dlat, dlon
    
        north_min, east_min = latlon_to_offset(lat_min, lon_min, IPLat, IPLong)
        north_max, east_max = latlon_to_offset(lat_max, lon_max, IPLat, IPLong)
        # Convert to miles
        north_min /= 1609.34
        north_max /= 1609.34
        east_min /= 1609.34
        east_max /= 1609.34
    
        plt.figure(figsize=(8, 8))
        plt.imshow(img, extent=[east_min, east_max, north_min, north_max], origin='upper')
        freefall_mask = np.array(phases) == 0
        canopy_mask = np.array(phases) == 1
        plt.plot(easts_miles[freefall_mask], norths_miles[freefall_mask], color='red', linewidth=2, label='Freefall')
        plt.plot(easts_miles[canopy_mask], norths_miles[canopy_mask], color='blue', linewidth=2, label='Canopy')
        plt.plot(circle_east_miles, circle_north_miles, color='green', linestyle='--', label='Canopy Glide Circle')
        plt.scatter([exit_east_miles], [exit_north_miles], color='cyan', marker='o', label='Exit Point')
        plt.scatter([0], [0], color='yellow', marker='x', label='Dropzone (IP)')
        plt.xlabel('East Offset (miles)')
        plt.ylabel('North Offset (miles)')
        plt.title('Skydiver Trajectory Relative to Dropzone (Miles)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Satellite image could not be retrieved. Plotting trajectory only.")
        plt.figure(figsize=(8, 8))
        freefall_mask = np.array(phases) == 0
        canopy_mask = np.array(phases) == 1
        plt.plot(easts_miles[freefall_mask], norths_miles[freefall_mask], color='red', linewidth=2, label='Freefall')
        plt.plot(easts_miles[canopy_mask], norths_miles[canopy_mask], color='blue', linewidth=2, label='Canopy')
        plt.plot(circle_east_miles, circle_north_miles, color='green', linestyle='--', label='Canopy Glide Circle')
        plt.scatter([exit_east_miles], [exit_north_miles], color='cyan', marker='o', label='Exit Point')
        plt.scatter([0], [0], color='yellow', marker='x', label='Dropzone (IP)')
        plt.xlabel('East Offset (miles)')
        plt.ylabel('North Offset (miles)')
        plt.title('Skydiver Trajectory Relative to Dropzone (Miles)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    final_lat, final_lon = traj_lat[-1], traj_lon[-1]
    print(f"Landing at: {final_lat}, {final_lon}")

if __name__ == "__main__":
	main()
