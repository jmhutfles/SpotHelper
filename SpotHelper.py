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

# Simulate freefall
alts, norths, easts, times = simulate_freefall(
    alt0_ft=13000,
    mass_kg=90,
    CdA=0.505,
    north_interp=north_interp,
    east_interp=east_interp,
    dt=0.1
)

# Plot North position vs Altitude
plt.figure(figsize=(8, 6))
plt.plot(norths, alts)
plt.xlabel('North Drift (meters)')
plt.ylabel('Altitude (ft)')
plt.title('Skydiver North Drift vs Altitude')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Plot East position vs Altitude
plt.figure(figsize=(8, 6))
plt.plot(easts, alts)
plt.xlabel('East Drift (meters)')
plt.ylabel('Altitude (ft)')
plt.title('Skydiver East Drift vs Altitude')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Plot 2D trajectory (North vs East)
plt.figure(figsize=(8, 6))
plt.plot(easts, norths)
plt.xlabel('East Drift (meters)')
plt.ylabel('North Drift (meters)')
plt.title('Skydiver Horizontal Trajectory')
plt.grid(True)
plt.axis('equal')
plt.show()

# Get satellite image and bounding box
img, (lat_min, lat_max, lon_min, lon_max) = get_sat_image(IPLat, IPLong, zoom=13, size=400)

# Convert trajectory to lat/lon
traj_lat, traj_lon = meters_to_latlon(norths, easts, IPLat, IPLong)

# Plot trajectory over satellite image
if img is not None:
    plt.figure(figsize=(8, 8))
    plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    plt.plot(traj_lon, traj_lat, color='red', linewidth=2, label='Trajectory')
    plt.scatter([IPLong], [IPLat], color='yellow', marker='x', label='Dropzone')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Skydiver Trajectory over Yandex Satellite Image')
    plt.legend()
    plt.show()
else:
    print("Satellite image could not be retrieved. Plotting trajectory only.")
    plt.figure(figsize=(8, 8))
    plt.plot(traj_lon, traj_lat, color='red', linewidth=2, label='Trajectory')
    plt.scatter([IPLong], [IPLat], color='yellow', marker='x', label='Dropzone')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Skydiver Trajectory (No Satellite Image)')
    plt.legend()
    plt.show()

# Plot Altitude vs Time
plt.figure(figsize=(8, 6))
plt.plot(times, alts)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (ft)')
plt.title('Skydiver Altitude vs Time')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Plot North Drift vs Time
plt.figure(figsize=(8, 6))
plt.plot(times, norths)
plt.xlabel('Time (s)')
plt.ylabel('North Drift (meters)')
plt.title('Skydiver North Drift vs Time')
plt.grid(True)
plt.show()

# Plot East Drift vs Time
plt.figure(figsize=(8, 6))
plt.plot(times, easts)
plt.xlabel('Time (s)')
plt.ylabel('East Drift (meters)')
plt.title('Skydiver East Drift vs Time')
plt.grid(True)
plt.show()

