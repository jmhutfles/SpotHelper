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

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alt_grid, north_wind_vals, label='North Wind (m/s)')
plt.plot(alt_grid, east_wind_vals, label='East Wind (m/s)')
plt.xlabel('Altitude (ft)')
plt.ylabel('Wind Component (m/s)')
plt.title('Interpolated North and East Winds vs Altitude')
plt.legend()
plt.grid(True)
plt.show()

