import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Functions import get_winds_aloft_table, get_wind_component_interpolators, get_sat_image, meters_to_latlon, simulate_freefall

# Lat Long for Skydive Crosskeys
IPLat = 39.70729978214128
IPLong = -75.03605104306934

# Pull Winds
RawWinds = get_winds_aloft_table(IPLat, IPLong)
north_interp, east_interp = get_wind_component_interpolators(RawWinds)

# First simulation: exit directly over target
alts, norths, easts, times = simulate_freefall(
    alt0_ft=13000,
    mass_kg=90,
    CdA=0.505,
    north_interp=north_interp,
    east_interp=east_interp,
    dt=0.1
)

# Calculate required exit offset to land at IPLat/IPLong
final_north = norths[-1]
final_east = easts[-1]
required_north_offset = -final_north
required_east_offset = -final_east

def meters_offset_to_latlon(north_offset, east_offset, lat0, lon0):
    dlat = north_offset / 111320
    dlon = east_offset / (111320 * np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon

exit_lat, exit_lon = meters_offset_to_latlon(required_north_offset, required_east_offset, IPLat, IPLong)
print(f"Exit at: {exit_lat}, {exit_lon}")

# Rerun simulation from the new exit point
alts, norths, easts, times = simulate_freefall(
    alt0_ft=13000,
    mass_kg=90,
    CdA=0.505,
    north_interp=north_interp,
    east_interp=east_interp,
    dt=0.1,
    north0=required_north_offset,
    east0=required_east_offset
)



# Get satellite image and bounding box
img, (lat_min, lat_max, lon_min, lon_max) = get_sat_image(IPLat, IPLong, zoom=13, size=400)

# Convert trajectory to lat/lon
traj_lat, traj_lon = meters_to_latlon(norths, easts, IPLat, IPLong)

# Plot trajectory over satellite image
if img is not None:
    plt.figure(figsize=(8, 8))
    plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    plt.plot(traj_lon, traj_lat, color='red', linewidth=2, label='Trajectory')
    plt.scatter([exit_lon], [exit_lat], color='cyan', marker='o', label='Exit Point')
    plt.scatter([IPLong], [IPLat], color='yellow', marker='x', label='Dropzone')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Skydiver Trajectory over Yandex Satellite Image (Corrected Exit)')
    plt.legend()
    plt.show()
else:
    print("Satellite image could not be retrieved. Plotting trajectory only.")
    plt.figure(figsize=(8, 8))
    plt.plot(traj_lon, traj_lat, color='red', linewidth=2, label='Trajectory')
    plt.scatter([exit_lon], [exit_lat], color='cyan', marker='o', label='Exit Point')
    plt.scatter([IPLong], [IPLat], color='yellow', marker='x', label='Dropzone')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Skydiver Trajectory (No Satellite Image, Corrected Exit)')
    plt.legend()
    plt.show()

final_lat, final_lon = traj_lat[-1], traj_lon[-1]
print(f"Landing at: {final_lat}, {final_lon}")
print(f"Target:     {IPLat}, {IPLong}")
