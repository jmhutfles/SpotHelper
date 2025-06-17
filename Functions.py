import requests
import pandas as pd
from datetime import datetime, timezone

def get_winds_aloft_table(latitude, longitude):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&hourly=wind_speed_10m,wind_direction_10m,"
        "wind_speed_80m,wind_direction_80m,"
        "wind_speed_100m,wind_direction_100m,"
        "wind_speed_1000hPa,wind_direction_1000hPa,"
        "wind_speed_925hPa,wind_direction_925hPa,"
        "wind_speed_850hPa,wind_direction_850hPa,"
        "wind_speed_700hPa,wind_direction_700hPa,"
        "wind_speed_500hPa,wind_direction_500hPa,"
        "wind_speed_400hPa,wind_direction_400hPa,"
        "wind_speed_300hPa,wind_direction_300hPa"
    )

    response = requests.get(url)
    data = response.json()

    level_to_altitude = {
        "10m": 33,
        "80m": 262,
        "100m": 328,
        "1000hPa": 364,
        "925hPa": 2500,
        "850hPa": 4800,
        "700hPa": 9900,
        "500hPa": 18000,
        "400hPa": 23000,
        "300hPa": 30000,
    }

    levels = [
        "10m", "80m", "100m",
        "1000hPa", "925hPa", "850hPa", "700hPa", "500hPa", "400hPa", "300hPa"
    ]

    times = data['hourly']['time']
    now = datetime.now(timezone.utc)
    current_index = min(
        range(len(times)),
        key=lambda i: abs(
            datetime.fromisoformat(times[i].replace('Z', '+00:00')).astimezone(timezone.utc) - now
        )
    )

    winds = []
    for level in levels:
        speed = data['hourly'][f'wind_speed_{level}'][current_index]
        direction = data['hourly'][f'wind_direction_{level}'][current_index]
        winds.append({
            'Altitude (ft)': level_to_altitude[level],
            'Wind Speed (m/s)': speed,
            'Wind Direction (deg)': direction,
            'Level': level
        })

    df = pd.DataFrame(winds)
    return df

# Example usage:
# df = get_winds_aloft_table(40.7128, -74.0060)
# print(df)


import numpy as np
from scipy.interpolate import interp1d

def get_wind_component_interpolators(wind_df):
    """
    Given a DataFrame with 'Altitude (ft)', 'Wind Speed (m/s)', and 'Wind Direction (deg)',
    returns two interpolation functions: north_wind(altitude_ft), east_wind(altitude_ft).
    """
    altitudes = wind_df["Altitude (ft)"].values
    wind_speeds = wind_df["Wind Speed (m/s)"].values
    wind_dirs = wind_df["Wind Direction (deg)"].values

    # Calculate components
    north_winds = wind_speeds * np.cos(np.radians(wind_dirs))
    east_winds = wind_speeds * np.sin(np.radians(wind_dirs))

    # Create interpolation functions
    north_interp = interp1d(altitudes, north_winds, kind='linear', fill_value="extrapolate")
    east_interp = interp1d(altitudes, east_winds, kind='linear', fill_value="extrapolate")

    return north_interp, east_interp

# Example usage:
# RawWinds = get_winds_aloft_table(IPLat, IPLong)
# north_fn, east_fn = get_wind_component_interpolators(RawWinds)
# print(north_fn(5000), east_fn(5000))