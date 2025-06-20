import requests
from requests.exceptions import RequestException
import pandas as pd
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
from io import BytesIO

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
        "&windspeed_unit=ms"
    )

    response = requests.get(url)
    data = response.json()
    #print("Raw Open-Meteo response:", data)

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

def get_wind_component_interpolators(wind_df):
    """
    Given a DataFrame with 'Altitude (ft)', 'Wind Speed (m/s)', and 'Wind Direction (deg)',
    returns two interpolation functions: north_wind(altitude_ft), east_wind(altitude_ft).
    """
    altitudes = wind_df["Altitude (ft)"].values
    wind_speeds = wind_df["Wind Speed (m/s)"].values
    wind_dirs = wind_df["Wind Direction (deg)"].values

    # Calculate components
    north_winds = wind_speeds * np.cos(np.radians(wind_dirs)) * -1
    east_winds = wind_speeds * np.sin(np.radians(wind_dirs)) * -1

    # Create interpolation functions
    north_interp = interp1d(altitudes, north_winds, kind='linear', fill_value="extrapolate")
    east_interp = interp1d(altitudes, east_winds, kind='linear', fill_value="extrapolate")

    return north_interp, east_interp

def get_sat_image(latitude, longitude, zoom=13, size=400):
    """
    Downloads a satellite image centered at (latitude, longitude) using Yandex Static Maps.
    Returns a PIL Image and the bounding box (lat_min, lat_max, lon_min, lon_max).
    """
    url = (
        "https://static-maps.yandex.ru/1.x/"
        f"?ll={longitude},{latitude}"
        f"&z={zoom}"
        f"&l=sat"
        f"&size={size},{size}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if 'image' not in resp.headers.get('Content-Type', ''):
            print("Yandex did not return an image. Response headers:", resp.headers)
            print("Response content (truncated):", resp.content[:200])
            return None, (None, None, None, None)
        img = Image.open(BytesIO(resp.content))
    except RequestException as e:
        print(f"Could not retrieve satellite image: {e}")
        return None, (None, None, None, None)

    meters_per_pixel = 156543.03392 * np.cos(np.radians(latitude)) / (2 ** zoom)
    half_side_m = (size / 2) * meters_per_pixel
    dlat = (half_side_m / 111320)
    dlon = half_side_m / (40075000 * np.cos(np.radians(latitude)) / 360)
    lat_min = latitude - dlat
    lat_max = latitude + dlat
    lon_min = longitude - dlon
    lon_max = longitude + dlon

    return img, (lat_min, lat_max, lon_min, lon_max)


def meters_to_latlon(north, east, lat0, lon0):
    """
    Converts north/east meters to latitude/longitude offsets from (lat0, lon0).
    Returns arrays of latitudes and longitudes.
    """
    dlat = north / 111320  # meters per degree latitude
    dlon = east / (40075000 * np.cos(np.radians(lat0)) / 360)
    return lat0 + dlat, lon0 + dlon

def air_pressure(alt_m):
    """
    Returns air pressure in Pascals at altitude alt_m (meters) using the barometric formula.
    """
    P0 = 101325      # Sea level standard atmospheric pressure, Pa
    L = 0.0065       # Temperature lapse rate, K/m
    T0 = 288.15      # Sea level standard temperature, K
    g = 9.80665      # Gravity, m/s^2
    M = 0.0289644    # Molar mass of dry air, kg/mol
    R = 8.3144598    # Universal gas constant, J/(mol·K)
    return P0 * (1 - L * alt_m / T0) ** (g * M / (R * L))

def simulate_freefall(
    alt0_ft,
    mass_kg,
    CdA,
    north_interp,
    east_interp,
    dt=0.1,
    v_vert0=0.0,
    north0=0.0,
    east0=0.0
):
    """
    Simulate a skydiver's freefall with altitude-dependent air pressure/density and wind drift.
    Returns arrays: alts_ft, norths_m, easts_m, times_s
    """
    alt = alt0_ft * 0.3048  # initial altitude in meters
    v_vert = v_vert0         # initial vertical velocity (down, m/s)
    north = north0           # initial north position (meters)
    east = east0             # initial east position (meters)
    g = 9.81                 # gravity (m/s^2)

    alts = []
    norths = []
    easts = []
    times = []

    t = 0.0
    while alt > 0:
        alt_ft = alt / 0.3048
        wind_north = north_interp(alt_ft)
        wind_east = east_interp(alt_ft)

        # Air pressure and temperature at this altitude
        pressure = air_pressure(alt)
        temp = 288.15 - 0.0065 * alt

        # Air density
        R_specific = 287.058
        rho = pressure / (R_specific * temp)

        # Drag force
        drag = 0.5 * rho * v_vert**2 * CdA * np.sign(v_vert)
        F_net = mass_kg * g - drag
        a = F_net / mass_kg

        v_vert += a * dt
        alt -= v_vert * dt

        north += wind_north * dt
        east += wind_east * dt

        alts.append(alt / 0.3048)
        norths.append(north)
        easts.append(east)
        times.append(t)
        t += dt

    return np.array(alts), np.array(norths), np.array(easts), np.array(times)

def simulate_freefall_and_canopy(
    alt0_ft,
    mass_kg,
    CdA,
    north_interp,
    east_interp,
    deploy_alt_ft=3000,
    canopy_v_vert_fps=14,
    dt=0.1,
    v_vert0=0.0,
    north0=0.0,
    east0=0.0
):
    """
    Simulate freefall to deploy_alt_ft, then non-gliding canopy descent at canopy_v_vert_fps.
    Returns arrays: alts_ft, norths_m, easts_m, times_s, phases (0=freefall, 1=canopy)
    """
    alt = alt0_ft * 0.3048
    deploy_alt_m = deploy_alt_ft * 0.3048
    v_vert = v_vert0
    north = north0
    east = east0
    g = 9.81

    canopy_v_vert = canopy_v_vert_fps * 0.3048

    alts = []
    norths = []
    easts = []
    times = []
    phases = []

    t = 0.0
    phase = 0  # 0 = freefall, 1 = canopy

    while alt > 0:
        alt_ft = alt / 0.3048
        wind_north = north_interp(alt_ft)
        wind_east = east_interp(alt_ft)

        if alt > deploy_alt_m:
            # Freefall phase
            pressure = air_pressure(alt)
            temp = 288.15 - 0.0065 * alt
            R_specific = 287.058
            rho = pressure / (R_specific * temp)
            drag = 0.5 * rho * v_vert**2 * CdA * np.sign(v_vert)
            F_net = mass_kg * g - drag
            a = F_net / mass_kg
            v_vert += a * dt
            alt -= v_vert * dt
            north += wind_north * dt
            east += wind_east * dt
            phase = 0
        else:
            # Non-gliding canopy: only wind drift, constant vertical descent
            v_vert = canopy_v_vert
            north += wind_north * dt
            east += wind_east * dt
            alt -= v_vert * dt
            phase = 1

        alts.append(alt / 0.3048)
        norths.append(north)
        easts.append(east)
        times.append(t)
        phases.append(phase)
        t += dt

    return np.array(alts), np.array(norths), np.array(easts), np.array(times), np.array(phases)

def prompt_manual_winds():
    """
    Prompt the user to manually enter wind data if online retrieval fails.
    Returns a DataFrame with columns: 'Altitude (ft)', 'Wind Speed (m/s)', 'Wind Direction (deg)'
    """
    print("Could not retrieve winds aloft from the internet.")
    print("Please enter wind data manually for each level (altitude_ft, wind_speed_m/s, wind_direction_deg).")
    print("Example: 3000,5,270")
    print("Enter one per line. Type 'done' when finished.")
    rows = []
    while True:
        line = input("Enter (altitude_ft,wind_speed_m/s,wind_direction_deg): ")
        if line.strip().lower() == 'done':
            break
        try:
            alt, speed, direction = map(float, line.strip().split(','))
            rows.append({'Altitude (ft)': alt, 'Wind Speed (m/s)': speed, 'Wind Direction (deg)': direction})
        except Exception:
            print("Invalid input, please try again.")
    if not rows:
        raise RuntimeError("No wind data entered.")
    return pd.DataFrame(rows)