import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Functions import get_winds_aloft_table, get_wind_component_interpolators

# Lat Long for Skydive Crosskeys
IPLat = 39.70729978214128
IPLong = -75.03605104306934

# Pull Winds
RawWinds = get_winds_aloft_table(IPLat, IPLong)
north_interp, east_interp = get_wind_component_interpolators(RawWinds)

# Generate altitude range from 0 to 13,000 ft
alt_grid = np.linspace(0, 13000, 200)

# Interpolated wind components
north_wind_vals = north_interp(alt_grid)
east_wind_vals = east_interp(alt_grid)

# Constants
CdA = 0.505  # m^2
g = 9.81     # m/s^2
m = 90       # kg
rho = 1.225  # kg/m^3 (air density at sea level)
dt = 0.1     # time step in seconds

# Initial conditions
alt = 13000 * 0.3048  # convert ft to meters
north = 0.0           # meters
east = 0.0            # meters
v_vert = 0.0          # vertical velocity (down, m/s)

alts = []
norths = []
easts = []
times = []

t = 0.0
while alt > 0:
    # Get wind at current altitude (convert meters to feet for interpolation)
    alt_ft = alt / 0.3048
    wind_north = north_interp(alt_ft)  # m/s
    wind_east = east_interp(alt_ft)    # m/s

    # Drag force
    drag = 0.5 * rho * v_vert**2 * CdA * np.sign(v_vert)
    # Net force (down is positive)
    F_net = m * g - drag
    # Acceleration
    a = F_net / m
    # Update vertical velocity and altitude
    v_vert += a * dt
    alt -= v_vert * dt

    # Update horizontal position (wind drift)
    north += wind_north * dt
    east += wind_east * dt

    # Store for plotting
    alts.append(alt / 0.3048)  # store in feet
    norths.append(north)
    easts.append(east)
    times.append(t)
    t += dt

# Convert lists to arrays for plotting
alts = np.array(alts)
norths = np.array(norths)
easts = np.array(easts)
times = np.array(times)

# Plot North position vs Altitude
plt.figure(figsize=(8, 6))
plt.plot(norths, alts)
plt.xlabel('North Drift (meters)')
plt.ylabel('Altitude (ft)')
plt.title('Skydiver North Drift vs Altitude')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Optional: Plot East position vs Altitude
plt.figure(figsize=(8, 6))
plt.plot(easts, alts)
plt.xlabel('East Drift (meters)')
plt.ylabel('Altitude (ft)')
plt.title('Skydiver East Drift vs Altitude')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Optional: Plot 2D trajectory (North vs East)
plt.figure(figsize=(8, 6))
plt.plot(easts, norths)
plt.xlabel('East Drift (meters)')
plt.ylabel('North Drift (meters)')
plt.title('Skydiver Horizontal Trajectory')
plt.grid(True)
plt.axis('equal')
plt.show()

