"""Подгонка модели полёта к таблице range-mil.json (Wardogs, миномёт L81).

Модель:  m dv/dt = -m g - k1 v          (вязкое сопротивление, линейное по v)

Масса входит только отношением k1/m, поэтому свободные параметры — начальная
скорость v0 и gamma = k1/m. Абсолютное k1 = gamma * m требует знать массу мины.

Угол возвышения берётся из показаний прицела:
    theta(mil) = TH0 + C * mil,
где шкала зафиксирована двумя подтверждёнными якорями:
    150 mil -> 37.0°,   950 mil -> 86.5°.

Для линейного сопротивления дальность считается в замкнутой форме, без
численного интегрирования (см. rng_lin).

Запуск:  python3 trajectory-fit.py
"""

import json
import numpy as np
from scipy.optimize import brentq, least_squares

G = 9.81
MIL_LOW, TH_LOW = 150.0, 37.0
MIL_HIGH, TH_HIGH = 950.0, 86.5
C = (TH_HIGH - TH_LOW) / (MIL_HIGH - MIL_LOW)     # градусов на mil
TH0 = TH_LOW - C * MIL_LOW                        # угол при mil = 0
MASSES = (3.2, 4.0, 4.5)                          # 81-мм мина, кг


def load(path="range-mil.json"):
    d = json.load(open(path, encoding="utf-8"))
    mil = np.array([p["mil"] for p in d["points"]], float)
    rng = np.array([p["range_m"] for p in d["points"]], float)
    return mil, rng


def theta(mil):
    return np.radians(TH0 + C * mil)


def rng_lin(th, v0, gamma):
    """Дальность полёта при линейном сопротивлении. th — угол, рад."""
    vx0, vy0 = v0 * np.cos(th), v0 * np.sin(th)
    if gamma < 1e-9:
        return vx0 * 2 * vy0 / G
    def y(t):
        return (vy0 + G / gamma) * (1 - np.exp(-gamma * t)) / gamma - (G / gamma) * t
    t = brentq(y, 1e-6, 1e5, xtol=1e-12, rtol=1e-14)
    return vx0 * (1 - np.exp(-gamma * t)) / gamma


def residuals(p, th, target):
    v0, gamma = p
    model = np.array([rng_lin(t, v0, gamma) for t in th])
    return model - target


def fit(th, target):
    best = None
    for start in [(100, 0.05), (166, 0.18), (300, 0.4), (600, 0.9), (1200, 2.0)]:
        s = least_squares(residuals, start, args=(th, target),
                          bounds=([5, 0], [20000, 80]), xtol=1e-14, ftol=1e-14)
        rms = float(np.sqrt(np.mean(s.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, s.x.copy())
    return best


def profile_gamma(th, target, gammas):
    """Для каждого gamma — лучший v0. Показывает вырожденность v0 <-> gamma."""
    out = []
    for gm in gammas:
        f = lambda v0: residuals((v0[0], gm), th, target)
        s = least_squares(f, [200.0], bounds=([5], [20000]), xtol=1e-12, ftol=1e-12)
        out.append((gm, float(s.x[0]), float(np.sqrt(np.mean(s.fun ** 2)))))
    return out


def main():
    mil, target = load()
    th = theta(mil)
    rms, (v0, gamma) = fit(th, target)

    print(f"шкала: theta = {TH0:.2f}° + {C:.5f}°/mil  "
          f"({360 / C:.0f} mil на круг)")
    print(f"углы theta(150)={TH0 + C * 150:.1f}°, theta(950)={TH0 + C * 950:.1f}°")
    print(f"лучший фит: v0 = {v0:.1f} м/с,  gamma = k1/m = {gamma:.4f} 1/с,  rms = {rms:.2f} м")
    for m in MASSES:
        print(f"    при m = {m} кг:  k1 = gamma*m = {gamma * m:.3f} кг/с")

    model = np.array([rng_lin(t, v0, gamma) for t in th])
    print("\n mil  угол,°   данные  модель   ошибка")
    for i in range(len(mil)):
        print(f"{mil[i]:4.0f}  {np.degrees(th[i]):6.2f}  {target[i]:6.0f}  "
              f"{model[i]:6.1f}  {model[i] - target[i]:+7.1f}")

    print("\nвырожденность (профиль по gamma, v0 оптимизируется):")
    print("  gamma    v0(опт)    rms")
    for gm, v, r in profile_gamma(th, target, [0.10, 0.14, 0.164, 0.18, 0.20, 0.22, 0.26]):
        print(f"  {gm:5.3f}  {v:8.1f}  {r:6.2f}")


if __name__ == "__main__":
    main()
