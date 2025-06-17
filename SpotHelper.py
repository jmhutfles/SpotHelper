import numpy as np
import pandas as pd
from Functions import get_winds_aloft_table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#Lat Long for Skydive Crosskeys
IPLat = 39.70729978214128
IPLong = -75.03605104306934


#Pull Winds
RawWinds = get_winds_aloft_table(IPLat, IPLong)


# Calculate wind components (in knots)
ConponentWinds = pd.DataFrame()
ConponentWinds["Altitude (ft)"] = RawWinds["Altitude (ft)"]
ConponentWinds["North Winds (m/s)"] = RawWinds["Wind Speed (m/s)"] * np.cos(np.radians(RawWinds["Wind Direction (deg)"]))
ConponentWinds["East Winds (m/s)"] = RawWinds["Wind Speed (m/s)"] * np.sin(np.radians(RawWinds["Wind Direction (deg)"]))

print(ConponentWinds)

# Example: Interpolate North and East wind components
altitudes = ConponentWinds["Altitude (ft)"]
north_winds = ConponentWinds["North Winds (m/s)"]
east_winds = ConponentWinds["East Winds (m/s)"]

# Create interpolation functions (linear by default)
north_interp = interp1d(altitudes, north_winds, kind='linear', fill_value="extrapolate")
east_interp = interp1d(altitudes, east_winds, kind='linear', fill_value="extrapolate")

# Example: Estimate wind at 5000 ft
# alt_query = 5000
# north_at_5000 = north_interp(alt_query)
# east_at_5000 = east_interp(alt_query)
