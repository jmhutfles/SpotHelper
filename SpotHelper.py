import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from Functions import (
    get_winds_aloft_table, get_wind_component_interpolators, get_sat_image,
    meters_to_latlon, simulate_freefall_and_canopy, prompt_manual_winds
)

# -------------------- HARD CONSTANTS --------------------
DT = 0.1
SAT_IMG_SIZE = 400
CIRCLE_RESOLUTION = 200
# --------------------------------------------------------

# -------------------- DEFAULT SIMULATION PARAMETERS --------------------
DEFAULTS = {
    "IPLat": 39.7065614,
    "IPLong": -75.0352181,
    "EXIT_ALTITUDE_FT": 13000,
    "DEPLOY_ALTITUDE_FT": 3000,
    "MASS_KG": 90,
    "CDA": 0.505,
    "CANOPY_V_VERT_FPS": 8,
    "CANOPY_V_HORIZ_FPS": 24,
    "SAT_IMG_ZOOM": 13
}
# -----------------------------------------------------------------------

def meters_offset_to_latlon(north_offset, east_offset, lat0, lon0):
    dlat = north_offset / 111320
    dlon = east_offset / (111320 * np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon

def run_simulation(params):
    # Get parameters from UI
    try:
        IPLat = float(params["IPLat"].get())
        IPLong = float(params["IPLong"].get())
        EXIT_ALTITUDE_FT = float(params["EXIT_ALTITUDE_FT"].get())
        DEPLOY_ALTITUDE_FT = float(params["DEPLOY_ALTITUDE_FT"].get())
        MASS_KG = float(params["MASS_KG"].get())
        CDA = float(params["CDA"].get())
        CANOPY_V_VERT_FPS = float(params["CANOPY_V_VERT_FPS"].get())
        CANOPY_V_HORIZ_FPS = float(params["CANOPY_V_HORIZ_FPS"].get())
        SAT_IMG_ZOOM = int(params["SAT_IMG_ZOOM"].get())
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
        return

    # Pull Winds with redundancy
    try:
        RawWinds = get_winds_aloft_table(IPLat, IPLong)
    except Exception as e:
        messagebox.showwarning("Winds Aloft", f"Error retrieving winds aloft: {e}\nYou will be prompted to enter them manually in the terminal.")
        RawWinds = prompt_manual_winds()

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

    # --- Calculate bounding box for all points ---
    all_lats = np.concatenate([traj_lat, circle_lat])
    all_lons = np.concatenate([traj_lon, circle_lon])
    lat_min, lat_max = np.min(all_lats), np.max(all_lats)
    lon_min, lon_max = np.min(all_lons), np.max(all_lons)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Helper for haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = phi2 - phi1
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    width_m = haversine_distance(center_lat, lon_min, center_lat, lon_max)
    height_m = haversine_distance(lat_min, center_lon, lat_max, center_lon)
    width_m *= 1.2   # Add margin
    height_m *= 1.2  # Add margin

    # --- Get high-res satellite image ---
    from Functions import get_highres_sat_image
    img, (lat_min_img, lat_max_img, lon_min_img, lon_max_img) = get_highres_sat_image(
        center_lat, center_lon, zoom=SAT_IMG_ZOOM, size=SAT_IMG_SIZE, width_m=width_m, height_m=height_m
    )

    # Convert trajectory and circle to miles relative to IP
    norths_miles = norths / 1609.34
    easts_miles = easts / 1609.34
    circle_north_miles = (circle_north + required_north_offset) / 1609.34
    circle_east_miles = (circle_east + required_east_offset) / 1609.34
    exit_north_miles = required_north_offset / 1609.34
    exit_east_miles = required_east_offset / 1609.34

    # For satellite image overlay: calculate image bounds in miles relative to IP
    if img is not None and None not in (lat_min_img, lat_max_img, lon_min_img, lon_max_img):
        def latlon_to_offset(lat, lon, lat0, lon0):
            dlat = (lat - lat0) * 111320
            dlon = (lon - lon0) * (40075000 * np.cos(np.radians(lat0)) / 360)
            return dlat, dlon

        north_min, east_min = latlon_to_offset(lat_min_img, lon_min_img, IPLat, IPLong)
        north_max, east_max = latlon_to_offset(lat_max_img, lon_max_img, IPLat, IPLong)
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
    #messagebox.showinfo("Simulation Complete", f"Exit at: {exit_lat}, {exit_lon}\nLanding at: {final_lat}, {final_lon}")

def main():
    root = tk.Tk()
    root.title("SpotHelper Simulation Parameters")

    params = {}
    row = 0
    for key, val in DEFAULTS.items():
        tk.Label(root, text=key).grid(row=row, column=0, sticky="e")
        entry = tk.Entry(root)
        entry.insert(0, str(val))
        entry.grid(row=row, column=1)
        params[key] = entry
        row += 1

    run_btn = tk.Button(root, text="Run Simulation", command=lambda: run_simulation(params))
    run_btn.grid(row=row, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()