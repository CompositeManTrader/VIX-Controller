"""
svi.py — SVI Gatheral (2004) smile fit por slice.

FIX vs app.py original:
  - Recibe `T` real (en años) para cada slice, en lugar de T_proxy=1.0.
  - Constrains butterfly (Durrleman 2005): b(1+|ρ|) ≤ 2.
  - Rechaza fits con |ρ| >= SVI_RHO_MAX (0.99 default).

Retorna dict {a, b, rho, m, sigma, iv_fit, log_mon, r2, T} o None.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from ..config import SVI_MAX_ITER, SVI_MIN_POINTS, SVI_RHO_MAX


def svi_raw_w(k: np.ndarray, a: float, b: float, rho: float,
              m: float, sigma: float) -> np.ndarray:
    """Varianza total w(k) = a + b·[ρ·(k-m) + √((k-m)² + σ²)]."""
    disc = np.maximum((k - m) ** 2 + sigma ** 2, 1e-12)
    return a + b * (rho * (k - m) + np.sqrt(disc))


def fit_svi_slice(strikes: np.ndarray, ivs: np.ndarray, F: float, T: float,
                  max_iter: int = SVI_MAX_ITER) -> dict | None:
    """
    Fit Raw-SVI a un slice (un vencimiento).

    Args:
        strikes: strikes K del slice
        ivs:     IVs observadas (decimal, e.g. 0.20 = 20%)
        F:       forward al vencimiento, F = S·exp((r-q)·T)
        T:       time-to-expiry EN AÑOS (fix vs original que usaba 1.0)

    Returns:
        dict con a, b, rho, m, sigma, iv_fit, log_mon, r2, T
        o None si no converge.
    """
    if len(strikes) < SVI_MIN_POINTS or len(ivs) < SVI_MIN_POINTS:
        return None
    if T <= 0 or F <= 0:
        return None

    k = np.log(np.asarray(strikes, dtype=float) / F)
    iv = np.asarray(ivs, dtype=float)
    w_obs = iv ** 2 * T   # varianza TOTAL (fix clave)

    def loss(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if b * (1 + abs(rho)) > 2:          # Durrleman butterfly
            return 1e10
        w_fit = svi_raw_w(k, a, b, rho, m, sigma)
        if np.any(w_fit < 0):
            return 1e10
        return float(np.sum((w_obs - w_fit) ** 2))

    a0 = float(np.median(w_obs)) * 0.8
    x0 = np.array([a0, 0.1, -0.3, 0.0, 0.1])

    try:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": max_iter,
                                "xatol": 1e-6, "fatol": 1e-8})
        if not res.success and res.fun > 0.1:
            return None
        a, b, rho, m, sigma = res.x
        if b <= 0 or sigma <= 0 or abs(rho) >= SVI_RHO_MAX:
            return None
        w_fit = svi_raw_w(k, a, b, rho, m, sigma)
        iv_fit = np.sqrt(np.maximum(w_fit / T, 0.0))
        ss_tot = float(np.sum((w_obs - w_obs.mean()) ** 2))
        r2 = float(1 - np.sum((w_obs - w_fit) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        return {
            "a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma),
            "iv_fit": iv_fit, "log_mon": k, "r2": r2, "T": float(T),
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
