import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import os
import xarray as xr
from scipy.ndimage import zoom


class ClimateSystem:
    # =========================
    # SETTINGS
    # =========================

    NX, NY = 360, 180
    MAX_EVENTS = 2000
    DAYS_FORECAST = 7

    TEMP_GRAD_WEAK = 0.00001
    TEMP_GRAD_MED  = 0.0001
    TEMP_GRAD_STR  = 0.001

    LAND_DAMPING = 0.6
    MOUNTAIN_BLOCK = 1500.0

    OUTPUT_DIR = "data"
    TEMP_FILE = f"{OUTPUT_DIR}/temperature.npy"
    LAT_FILE  = f"{OUTPUT_DIR}/lat.npy"
    LON_FILE  = f"{OUTPUT_DIR}/lon.npy"

    GEBCO_FILE = "GEBCO_2025_sub_ice.nc"
    TEMP_NC    = "air.4Xday.ltm.1991-2020.nc"
    GEBCO_DOWNSCALE = 20

    # =========================
    # UTILS
    # =========================

    def ensure_dirs(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    # =========================
    # ========= MODE 1 ========
    # ===== GENERATE .NPY =====
    # =========================

    def generate_fields(self):
        print("\n🌍 Generating climate fields...")
        self.ensure_dirs()

        # ---- GEBCO ----
        print("📏 Loading GEBCO...")
        ds = xr.open_dataset(self.GEBCO_FILE)
        if "elevation" in ds:
            B = ds["elevation"].values.astype(np.float32)
        elif "z" in ds:
            B = ds["z"].values.astype(np.float32)
        else:
            raise RuntimeError("GEBCO elevation not found")

        lat_b = ds["lat"].values
        lon_b = ds["lon"].values
        ds.close()

        B = zoom(B, 1 / self.GEBCO_DOWNSCALE, order=1)
        lat_b = lat_b[::self.GEBCO_DOWNSCALE]
        lon_b = lon_b[::self.GEBCO_DOWNSCALE]

        # ---- TEMPERATURE ----
        print("🌡 Loading temperature...")
        ds = xr.open_dataset(self.TEMP_NC, decode_times=False)
        air = ds["air"]
        T = air.mean(dim=air.dims[:-2]).values.astype(np.float32)
        lat_t = ds["lat"].values
        lon_t = ds["lon"].values
        ds.close()

        # ---- INTERPOLATION ----
        print("🔄 Interpolating temperature to GEBCO grid...")
        da = xr.DataArray(
            T,
            coords={"lat": lat_t, "lon": lon_t},
            dims=("lat", "lon")
        )

        T_i_raw = da.interp(lat=lat_b, lon=lon_b, method="linear")

        # КРИТИЧЕСКАЯ ПРАВКА: Заполняем NaN (пустоты на краях) ближайшими значениями
        T_i = T_i_raw.ffill("lat").bfill("lat").ffill("lon").bfill("lon").values.astype(np.float32)

        # Если после заполнения всё еще остались NaN (например, весь файл пустой), заменим на 0
        T_i = np.nan_to_num(T_i, nan=0.0)

        np.save(self.TEMP_FILE, T_i)
        np.save(self.LAT_FILE, lat_b.astype(np.float32))
        np.save(self.LON_FILE, lon_b.astype(np.float32))
        print(f"✅ Saved! Data range: {T_i.min():.1f} to {T_i.max():.1f}")

    # =========================
    # ========= MODE 2 ========
    # ========= ANALYSIS ======
    # =========================

    def load_npy(self, path):
        return np.load(path)

    def load_gebco(self):
        ds = xr.open_dataset(self.GEBCO_FILE, decode_times=False)
        elev = ds["elevation"].astype(np.float32)
        elev = elev.where(elev != elev.attrs.get("_FillValue", -9999))
        return elev.values, ds["lat"].values, ds["lon"].values

    def classify_event(self, strength, lat):
        if strength < self.TEMP_GRAD_WEAK:
            return "cloud", "green"
        elif strength < self.TEMP_GRAD_MED:
            return "front", "yellow"
        elif abs(lat) > 55:
            return "polar", "blue"
        else:
            return "cyclone", "red"

    def detect_events(self, T, lat, lon, B):
        events = []
        T_safe = np.nan_to_num(T, nan=np.nanmean(T))
        dTy, dTx = np.gradient(T_safe)
        grad = np.sqrt(dTx ** 2 + dTy ** 2)

        rows, cols = T.shape
        step = 6  # Увеличим шаг для чистоты карты

        for i in range(2, rows - 2, step):
            # Пропускаем самые края (полюса), где часто бывают ошибки данных (те самые "линии")
            if abs(lat[i]) > 85:
                continue

            for j in range(2, cols - 2, step):
                strength = grad[i, j]

                # Игнорируем NaN и абсолютный штиль
                if np.isnan(strength) or strength < 1e-7:
                    continue

                dx, dy = dTx[i, j], dTy[i, j]

                # КРИТИЧЕСКАЯ ПРАВКА: Ограничиваем влияние аномалий
                # Если градиент слишком огромный (ошибка данных), приравниваем его к среднему
                if strength > 2.0:
                    strength = 0.5

                etype, color = self.classify_event(strength, lat[i])

                events.append({
                    "lat0": float(lat[i]),
                    "lon0": float(lon[j]),
                    "dx": float(dx),
                    "dy": float(dy),
                    "strength": float(strength),
                    "type": etype,
                    "color": color
                })

        # Вместо простой сортировки перемешаем события,
        # чтобы они не рисовались только в одном регионе
        import random
        random.shuffle(events)

        return events[:self.MAX_EVENTS]

    def forecast_path(self, event):
        # Умножаем на большой коэффициент (например, 50-100),
        # чтобы даже слабый ветерок стал длинной стрелкой
        boost = 15.0

        lat_f = event["lat0"] + (event["dy"] * self.DAYS_FORECAST * boost)
        lon_f = event["lon0"] + (event["dx"] * self.DAYS_FORECAST * boost)

        # Ограничения, чтобы не улететь за край
        lat_f = np.clip(lat_f, -90, 90)
        lon_f = ((lon_f + 180) % 360) - 180
        return lat_f, lon_f

    def analyze_and_plot(self):
        print("\n📡 Loading fields...")

        T = self.load_npy(self.TEMP_FILE)
        lat = self.load_npy(self.LAT_FILE)
        lon = self.load_npy(self.LON_FILE)

        # =========================
        # OPTIONAL: BATHYMETRY
        # =========================
        B = None
        has_bathymetry = False

        try:
            B_full, lat_b, lon_b = self.load_gebco()
            B = zoom(
                B_full,
                (len(lat) / len(lat_b), len(lon) / len(lon_b)),
                order=0
            )
            has_bathymetry = True
            print("🗺 Bathymetry loaded (optional)")
        except Exception as e:
            print("⚠️ Bathymetry not found — plotting without continents")

        # =========================
        # COORDINATE FIX
        # =========================
        lon = ((lon + 180) % 360) - 180
        idx = np.argsort(lon)
        lon, T = lon[idx], T[:, idx]

        if lat[0] > lat[-1]:
            lat, T = lat[::-1], T[::-1, :]

        # =========================
        # EVENT DETECTION
        # =========================
        events = self.detect_events(T, lat, lon, B)
        print(f"Detected events: {len(events)}")

        # =========================
        # PLOTTING
        # =========================
        plt.figure(figsize=(15, 7))

        # --- CONTINENTS (OPTIONAL) ---
        if has_bathymetry:
            LON_B, LAT_B = np.meshgrid(lon, lat)
            plt.contour(
                LON_B, LAT_B, B,
                levels=[0],
                colors="black",
                linewidths=0.7,
                zorder=2
            )

        # --- TEMPERATURE FIELD ---
        plt.imshow(
            np.nan_to_num(T, nan=np.nanmean(T)),
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            origin="lower",
            cmap="coolwarm",
            alpha=0.4,
            aspect="auto",
            zorder=1
        )

        # --- EVENTS ---
        for e in events:
            lat_f, lon_f = self.forecast_path(e)

            plt.arrow(
                e["lon0"], e["lat0"],
                lon_f - e["lon0"], lat_f - e["lat0"],
                color=e["color"],
                alpha=0.6,
                head_width=1.2,
                length_includes_head=True,
                zorder=4
            )

            plt.scatter(
                e["lon0"], e["lat0"],
                color=e["color"],
                s=20 + 30 * e["strength"],
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                zorder=5
            )

        title = "7-Day Forecast"
        if has_bathymetry:
            title += " with Continental Outlines"

        plt.title(title)
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


# =========================
# MAIN
# =========================

def main():
    system = ClimateSystem()

    print("\n1 — Generate .npy climate fields")
    print("2 — Analyze & show 7-day map")

    mode = input("Select mode (1/2): ").strip()

    if mode == "1":
        system.generate_fields()
    elif mode == "2":
        system.analyze_and_plot()
    else:
        print("Invalid mode")


if __name__ == "__main__":
    main()




