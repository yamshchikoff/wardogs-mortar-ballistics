"""Быстрая подгонка траектории миномёта L81 (Wardogs) по range-mil.json.

Модель:  dv/dt = -g*y_hat - gamma*v - beta*|v|*v,
    gamma = k1/m (вязкое, 1/с),  beta = k2/m (квадратичное, 1/м).

Шкала прицела зафиксирована: theta = TH0 + C*mil,  150 mil -> 37°, 950 mil -> 86.5°.
Интегрирование векторизовано по всем углам, якобиан dR/dp — через уравнения
чувствительности (S = ds/dp), поэтому оптимизатор — настоящий Гаусс-Ньютон.
Стартовые точки гоняются параллельно (ProcessPoolExecutor).

Запуск:  python3 drag-fit.py
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

G = 9.81
C = (86.5 - 37.0) / 800.0          # градусов на mil
TH0 = 37.0 - C * 150.0             # угол при mil = 0
MASSES = (3.2, 4.0, 4.5)           # 81-мм мина, кг


def load(path="range-mil.json"):
    d = json.load(open(path, encoding="utf-8"))
    mil = np.array([p["mil"] for p in d["points"]], float)
    rng = np.array([p["range_m"] for p in d["points"]], float)
    return np.radians(TH0 + C * mil), rng


def range_jac(th, v0, gamma, beta, dt=0.006, tmax=70.0):
    """Дальности для набора углов + якобиан dR/d(v0,gamma,beta)."""
    th = np.atleast_1d(np.asarray(th, float))
    N = th.size
    Y = np.zeros((N, 4))
    Y[:, 2] = v0 * np.cos(th)
    Y[:, 3] = v0 * np.sin(th)
    S = np.zeros((N, 4, 3))
    S[:, 2, 0] = np.cos(th)
    S[:, 3, 0] = np.sin(th)
    landed = np.zeros(N, bool)
    R = np.full(N, np.nan)
    JR = np.full((N, 3), np.nan)

    def deriv(Y, S):
        vx, vy = Y[:, 2], Y[:, 3]
        r = np.hypot(vx, vy)
        r = np.where(r < 1e-9, 1e-9, r)
        fY = np.stack([vx, vy,
                       -gamma * vx - beta * r * vx,
                       -G - gamma * vy - beta * r * vy], axis=1)
        Js = np.zeros((Y.shape[0], 4, 4))
        Js[:, 0, 2] = 1.0
        Js[:, 1, 3] = 1.0
        Js[:, 2, 2] = -gamma - beta * (r + vx * vx / r)
        Js[:, 2, 3] = -beta * vx * vy / r
        Js[:, 3, 2] = -beta * vy * vx / r
        Js[:, 3, 3] = -gamma - beta * (r + vy * vy / r)
        Jp = np.zeros((Y.shape[0], 4, 3))
        Jp[:, 2, 1] = -vx
        Jp[:, 3, 1] = -vy
        Jp[:, 2, 2] = -r * vx
        Jp[:, 3, 2] = -r * vy
        return fY, np.einsum("nij,njk->nik", Js, S) + Jp

    t = 0.0
    while t < tmax and not landed.all():
        Y0, S0 = Y.copy(), S.copy()
        f1, d1 = deriv(Y, S)
        f2, d2 = deriv(Y + .5 * dt * f1, S + .5 * dt * d1)
        f3, d3 = deriv(Y + .5 * dt * f2, S + .5 * dt * d2)
        f4, d4 = deriv(Y + dt * f3, S + dt * d3)
        Y = Y0 + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        S = S0 + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        new = (Y[:, 1] < 0) & ~landed
        if new.any():
            fr = (Y0[new, 1] / (Y0[new, 1] - Y[new, 1]))[:, None]
            xc = Y0[new, 0] + (Y[new, 0] - Y0[new, 0]) * fr[:, 0]
            vxc = Y0[new, 2] + (Y[new, 2] - Y0[new, 2]) * fr[:, 0]
            vyc = Y0[new, 3] + (Y[new, 3] - Y0[new, 3]) * fr[:, 0]
            Sc = S0[new] + (S[new] - S0[new]) * fr[:, :, None]
            R[new] = xc
            JR[new] = Sc[:, 0, :] - vxc[:, None] * Sc[:, 1, :] / vyc[:, None]
            landed |= new
        t += dt
    return R, JR


def fit(th, target, keys, starts):
    """keys — свободные параметры из ('v0','gamma','beta')."""
    from scipy.optimize import least_squares
    full = {"v0": 166.0, "gamma": 0.0, "beta": 0.0}
    idx = {"v0": 0, "gamma": 1, "beta": 2}
    cols = [idx[k] for k in keys]
    bounds = {"v0": (50, 400), "gamma": (0, 20), "beta": (0, 1)}

    def unpack(p):
        f = dict(full)
        f.update(dict(zip(keys, p)))
        return f["v0"], f["gamma"], f["beta"]

    def fun(p):
        R, _ = range_jac(th, *unpack(p))
        return np.where(np.isfinite(R), R, 1e6) - target

    def jac(p):
        _, J = range_jac(th, *unpack(p))
        return J[:, cols]

    best = None
    for st in starts:
        s = least_squares(fun, st, jac=jac, bounds=([bounds[k][0] for k in keys],
                                                    [bounds[k][1] for k in keys]),
                          xtol=1e-11, ftol=1e-11)
        rms = float(np.sqrt(np.mean(s.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, s.x.copy())
    return best


TH, TARGET = load()
MODELS = [
    ("линейное", ("v0", "gamma"), [(166, 0.18), (120, 0.10), (250, 0.30), (140, 0.14)]),
    ("квадратичное", ("v0", "beta"), [(166, 0.001), (250, 0.003), (120, 0.0005)]),
    ("оба", ("v0", "gamma", "beta"),
     [(166, 0.18, 0.0), (200, 0.05, 0.001), (150, 0.0, 0.001), (250, 0.1, 0.002)]),
]


def work(arg):
    name, keys, start = arg
    rms, x = fit(TH, TARGET, keys, [start])
    return name, keys, tuple(start), rms, tuple(x)


if __name__ == "__main__":
    jobs = [m for name, keys, starts in MODELS for m in
            [(name, keys, s) for s in starts]]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=min(10, len(jobs))) as ex:
        for name, keys, start, rms, x in ex.map(work, jobs):
            rows.append((name, rms, keys, x, start))
    print("шкала: theta = %.2f° + %.5f°/mil  (%.0f mil на круг)" % (TH0, C, 360 / C))
    for name, keys, starts in MODELS:
        best = min((r for r in rows if r[0] == name), key=lambda r: r[1])
        print("%-14s rms=%6.2f  %s" % (name, best[1],
              "  ".join("%s=%.5f" % (k, v) for k, v in zip(best[2], best[3]))))
    print("время: %.1f с" % (time.time() - t0))
