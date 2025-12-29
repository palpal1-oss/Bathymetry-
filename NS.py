import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================
# 1. УТИЛИТЫ
# ============================================================

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def corrcoef(a, b):
    a = a.flatten()
    b = b.flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


# ============================================================
# 2. ГЕНЕРАЦИЯ БАТИМЕТРИИ
# ============================================================

def generate_bathymetry(n):
    b = np.random.randn(n, n)
    b = gaussian_filter(b, sigma=6)
    b -= b.max()
    b = -3000 * (b / np.min(b))
    return b


# ============================================================
# 3. ДИНАМИКА ТЕЧЕНИЙ
# ============================================================

def evolve_currents(u, v, bathy, dt, chi=20.0):
    """
    Простая хаотическая динамика,
    чувствительная к градиентам дна
    """

    bx = np.gradient(bathy, axis=1)
    by = np.gradient(bathy, axis=0)

    # хаос + реакция на рельеф
    u_new = u + dt * (-by + chi * np.sin(v))
    v_new = v + dt * ( bx + chi * np.sin(u))

    # мягкое затухание
    u_new *= 0.995
    v_new *= 0.995

    return u_new, v_new


# ============================================================
# 4. ИНВАРИАНТ (ВИХРЬ)
# ============================================================

def invariant(u, v, dx=1.0, dy=1.0):
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    vort = dv_dx - du_dy
    return vort


# ============================================================
# 5. ПАМЯТЬ И ИНКРЕМЕНТАЛЬНАЯ РЕКОНСТРУКЦИЯ
# ============================================================

def update_bathymetry(B_mem, W_mem, signal, alpha=0.05, perc=80):
    """
    Кусочное накопление информации
    """

    thr = np.percentile(np.abs(signal), perc)
    mask = np.abs(signal) > thr

    B_mem[mask] += alpha * signal[mask]
    W_mem[mask] += alpha

    return B_mem, W_mem


def reconstruct(B_mem, W_mem):
    B = np.zeros_like(B_mem)
    valid = W_mem > 0
    B[valid] = B_mem[valid] / W_mem[valid]
    return B


# ============================================================
# 6. ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ============================================================

def run_simulation(
    N=128,
    T=4.0,
    dt=0.5,
    chi=20.0,
    alpha=0.05,
    perc=80,
    smooth_sigma=1.0
):

    # истинное дно
    bathy_true = generate_bathymetry(N)

    # течения
    u = np.random.randn(N, N)
    v = np.random.randn(N, N)

    # память
    B_mem = np.zeros_like(bathy_true)
    W_mem = np.zeros_like(bathy_true)

    steps = int(T / dt)

    print("=" * 50)
    for k in range(steps):

        # динамика
        u, v = evolve_currents(u, v, bathy_true, dt, chi)

        # инвариант
        sig = invariant(u, v)

        # накопление
        B_mem, W_mem = update_bathymetry(
            B_mem, W_mem, sig,
            alpha=alpha,
            perc=perc
        )

        # реконструкция
        B_rec = reconstruct(B_mem, W_mem)

        # сглаживание (важно!)
        if smooth_sigma > 0:
            B_rec = gaussian_filter(B_rec, smooth_sigma)

        # метрики
        e = rmse(B_rec, bathy_true)
        c = corrcoef(B_rec, bathy_true)

        print(
            f"[t = {k*dt:4.1f} s] "
            f"RMSE = {e:8.2f} m | "
            f"Corr = {c:6.3f} | "
            f"Reconstruction = {100*c:6.2f} %"
        )

    print("=" * 50)

    return bathy_true, B_rec


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_results(bathy_true, bathy_rec):

    res = bathy_rec - bathy_true

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title("True bathymetry")
    plt.imshow(bathy_true, cmap="terrain")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.imshow(bathy_rec, cmap="terrain")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Residual")
    plt.imshow(res, cmap="seismic")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    bathy_true, bathy_rec = run_simulation(
        N=528,
        T=10.0,
        dt=0.5,
        chi=21.0,
        alpha=0.06,
        perc=75,
        smooth_sigma=1.2
    )

    plot_results(bathy_true, bathy_rec)
