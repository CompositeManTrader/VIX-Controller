"""
options_desk.py — Medidas de la página Opciones. Funciones puras, sin Streamlit.

SPY (Black-Scholes con dividendo continuo, forward F = S·e^{(r−q)T}):
  · Sonrisa con el lado fuera del dinero: puts con K < F, calls con K ≥ F.
  · ATM = IV interpolada en el forward. 25Δ = IV interpolada donde la delta
    BS vale −0,25 (put) o +0,25 (call). RR25 = call25 − put25 (negativo = se
    paga más por protegerse de caídas). BF25 = media de las dos − ATM.
  · A 30 días: varianza total interpolada entre los dos vencimientos que
    rodean los 30 días (la misma idea que el cálculo del VIX).
  · Movimiento descontado: straddle en el strike más cercano al spot / spot.

GEX (convención estándar del mercado: los dealers compran a los clientes las
calls que estos venden y venden las puts que estos compran):
  gamma de calls × interés abierto con signo +, de puts con signo −;
  en dólares por cada 1 % de movimiento: Γ·OI·100·S²·0,01. Positivo = los
  dealers compran caídas y venden subidas (amortiguan). Negativo = amplifican.
  El nivel de giro es el precio del SPY en el que la GEX total cambia de signo,
  recalculando la gamma de cada opción a ese precio con su IV de hoy.

VIX (Black-76 sobre el futuro de cada vencimiento): el forward sale de la
paridad put-call en el strike donde call y put valen más parecido. Las
opciones del VIX se liquidan contra el futuro, no contra el VIX de contado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DIAS_ANIO = 365.0
MULT = 100
IV_MIN, IV_MAX = 0.01, 3.00        # fuera de este rango la IV es un error de cotización


# ──────────────────────────────────────────────────────────────────────
# SPY: IV, delta y sonrisa
# ──────────────────────────────────────────────────────────────────────
def _t(dte: float) -> float:
    return max(float(dte), 0.5) / DIAS_ANIO


def tiempo_habil(vencimiento: str, hoy=None) -> float:
    """Años en sesiones hábiles hasta el vencimiento (/252). Con días naturales,
    un vencimiento con fin de semana de por medio sale con la IV hundida."""
    hoy = np.datetime64(pd.Timestamp(hoy or pd.Timestamp.now(tz="America/New_York")).date(), "D")
    n = np.busday_count(hoy, np.datetime64(pd.Timestamp(vencimiento).date(), "D"))
    return max(float(n), 0.5) / 252.0


def _T(dato: dict) -> float:
    return dato.get("T") or _t(dato["dte"])


def forward(spot: float, r: float, q: float, T: float) -> float:
    return spot * np.exp((r - q) * T)


def _precio_bs(S, K, T, r, q, v, call):
    d1 = (np.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    c = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(call, c, c - S * np.exp(-q * T) + K * np.exp(-r * T))


def _precio_b76(F, K, T, r, v, call):
    d1 = (np.log(F / K) + 0.5 * v * v * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    c = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.where(call, c, c - np.exp(-r * T) * (F - K))


def _biseccion(precio_fn, objetivo: np.ndarray, lo: float, hi: float, n: int = 60) -> np.ndarray:
    """IV por bisección vectorizada: el precio es creciente en la volatilidad.
    NaN si el precio no cabe entre el de `lo` y el de `hi` (sin solución)."""
    a = np.full(objetivo.shape, lo)
    b = np.full(objetivo.shape, hi)
    ok = (precio_fn(a) <= objetivo) & (precio_fn(b) >= objetivo)
    for _ in range(n):
        m = 0.5 * (a + b)
        arriba = precio_fn(m) > objetivo
        b = np.where(arriba, m, b)
        a = np.where(arriba, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def iv_bs(spot: float, K, T: float, r: float, q: float, precio, call) -> np.ndarray:
    K, precio = np.asarray(K, float), np.asarray(precio, float)
    call = np.broadcast_to(np.asarray(call, bool), K.shape)
    return _biseccion(lambda v: _precio_bs(spot, K, T, r, q, v, call), precio, IV_MIN, IV_MAX)


def iv_b76(F: float, K, T: float, r: float, precio, call, lo: float = 0.05,
           hi: float = 6.0) -> np.ndarray:
    K, precio = np.asarray(K, float), np.asarray(precio, float)
    call = np.broadcast_to(np.asarray(call, bool), K.shape)
    return _biseccion(lambda v: _precio_b76(F, K, T, r, v, call), precio, lo, hi)


def preparar(cadenas: dict, spot: float, r: float, q: float, hoy=None) -> dict:
    """Copia de las cadenas con el tiempo en sesiones hábiles (`T`) y la IV de
    cada opción (columna `iv`): se calcula una vez."""
    out = {}
    for venc, dato in cadenas.items():
        T = tiempo_habil(venc, hoy)
        nuevo = {"dte": dato["dte"], "T": T}
        for lado in ("calls", "puts"):
            d = dato[lado].copy()
            d["iv"] = iv_bs(spot, d["strike"].to_numpy(), T, r, q, d["mid"].to_numpy(), lado == "calls")
            nuevo[lado] = d
        out[venc] = nuevo
    return out


def _iv_de(d: pd.DataFrame, spot: float, T: float, r: float, q: float, call: bool) -> np.ndarray:
    if "iv" in d.columns:
        return d["iv"].to_numpy()
    return iv_bs(spot, d["strike"].to_numpy(), T, r, q, d["mid"].to_numpy(), call)


MEZCLA = 0.015    # ±1,5 % alrededor del forward: se promedian put y call


def sonrisa(dato: dict, spot: float, r: float, q: float) -> pd.DataFrame:
    """
    Sonrisa con IV y delta BS. Columnas: strike, tipo, mid, oi, iv, delta, m.
    Lejos del forward, el lado fuera del dinero. A ±1,5 % del forward, media
    ponderada de put y call del mismo strike: las opciones de SPY son
    americanas y la paridad no cierra exacta, lo que dejaría un salto en el ATM.
    """
    T = _T(dato)
    F = forward(spot, r, q, T)
    c, p = dato["calls"].copy(), dato["puts"].copy()
    c["iv_c"] = _iv_de(c, spot, T, r, q, True)
    p["iv_p"] = _iv_de(p, spot, T, r, q, False)
    s = pd.merge(p[["strike", "mid", "openInterest", "iv_p"]].rename(columns={"mid": "mid_p", "openInterest": "oi_p"}),
                 c[["strike", "mid", "openInterest", "iv_c"]].rename(columns={"mid": "mid_c", "openInterest": "oi_c"}),
                 on="strike", how="outer").sort_values("strike")
    w = np.clip((s["strike"] / F - (1 - MEZCLA)) / (2 * MEZCLA), 0.0, 1.0)     # peso de la call
    iv = np.where(s["iv_c"].isna(), np.where(w < 1, s["iv_p"], np.nan),
                  np.where(s["iv_p"].isna(), np.where(w > 0, s["iv_c"], np.nan),
                           w * s["iv_c"] + (1 - w) * s["iv_p"]))
    out = pd.DataFrame({"strike": s["strike"].to_numpy(),
                        "tipo": np.where(s["strike"] >= F, "C", "P"),
                        "mid": np.where(s["strike"] >= F, s["mid_c"], s["mid_p"]),
                        "oi": np.where(s["strike"] >= F, s["oi_c"], s["oi_p"]),
                        "iv": iv})
    out = out.dropna(subset=["iv"])
    out = out[(out["iv"] > 0)]
    if out.empty:
        return pd.DataFrame(columns=["strike", "tipo", "mid", "oi", "iv", "delta", "m"])
    d1 = (np.log(spot / out["strike"]) + (r - q + 0.5 * out["iv"] ** 2) * T) / (out["iv"] * np.sqrt(T))
    out["delta"] = np.where(out["tipo"] == "C", np.exp(-q * T) * norm.cdf(d1),
                            -np.exp(-q * T) * norm.cdf(-d1))
    out["m"] = out["strike"] / spot - 1.0
    out.attrs["F"] = F
    return out.reset_index(drop=True)


def _interp(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Interpolación lineal sin extrapolar: NaN si x0 queda fuera de los datos."""
    if len(x) < 2:
        return np.nan
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    if x0 < x[0] or x0 > x[-1]:
        return np.nan
    return float(np.interp(x0, x, y))


def iv_atm(s: pd.DataFrame) -> float:
    if s.empty:
        return np.nan
    return _interp(s["strike"].to_numpy(), s["iv"].to_numpy(), s.attrs.get("F", np.nan))


def iv_en_delta(s: pd.DataFrame, delta: float) -> float:
    """IV donde la delta BS vale `delta` (negativa = puts, positiva = calls)."""
    lado = s[s["tipo"] == ("P" if delta < 0 else "C")]
    return _interp(lado["delta"].to_numpy(), lado["iv"].to_numpy(), delta)


def metricas_vencimiento(dato: dict, spot: float, r: float, q: float) -> dict:
    s = sonrisa(dato, spot, r, q)
    atm, p25, c25 = iv_atm(s), iv_en_delta(s, -0.25), iv_en_delta(s, 0.25)
    return {"dte": dato["dte"], "atm": atm, "put25": p25, "call25": c25,
            "put10": iv_en_delta(s, -0.10),
            "rr25": c25 - p25, "bf25": (c25 + p25) / 2 - atm, "n": len(s)}


def estructura(cadenas: dict, spot: float, r: float, q: float) -> pd.DataFrame:
    filas = []
    for venc, dato in cadenas.items():
        m = metricas_vencimiento(dato, spot, r, q)
        m["vencimiento"] = venc
        filas.append(m)
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values("dte").reset_index(drop=True)


def a_30_dias(est: pd.DataFrame, col: str = "atm", dias: float = 30.0) -> float:
    """
    Valor a `dias` días. Para la IV ATM se interpola la varianza total
    (σ²·T); para el resto (RR, BF) linealmente en días. Sin extrapolar.
    """
    e = est.dropna(subset=[col])
    if len(e) < 2 or dias < e["dte"].min() or dias > e["dte"].max():
        return np.nan
    if col == "atm":
        w = e[col] ** 2 * e["dte"]
        return float(np.sqrt(np.interp(dias, e["dte"], w) / dias))
    return float(np.interp(dias, e["dte"], e[col]))


def movimiento_descontado(dato: dict, spot: float) -> float:
    """Straddle en el strike más cercano al spot, en % del spot."""
    c, p = dato["calls"], dato["puts"]
    comunes = sorted(set(c["strike"]) & set(p["strike"]), key=lambda k: abs(k - spot))
    if not comunes:
        return np.nan
    k = comunes[0]
    straddle = float(c.loc[c["strike"] == k, "mid"].iloc[0] + p.loc[p["strike"] == k, "mid"].iloc[0])
    return straddle / spot * 100.0


# ──────────────────────────────────────────────────────────────────────
# GEX
# ──────────────────────────────────────────────────────────────────────
def _filas_gex(cadenas: dict, spot: float, r: float, q: float) -> pd.DataFrame:
    """strike, signo (+1 call / −1 put), oi, iv, T de todas las cadenas."""
    partes = []
    for dato in cadenas.values():
        T = _T(dato)
        for lado, tipo, signo in (("calls", "C", 1.0), ("puts", "P", -1.0)):
            d = dato[lado]
            if d.empty:
                continue
            iv = _iv_de(d, spot, T, r, q, tipo == "C")
            partes.append(pd.DataFrame({"strike": d["strike"].to_numpy(), "signo": signo,
                                        "oi": d["openInterest"].to_numpy(dtype=float),
                                        "iv": iv, "T": T}))
    if not partes:
        return pd.DataFrame(columns=["strike", "signo", "oi", "iv", "T"])
    return pd.concat(partes, ignore_index=True).dropna(subset=["iv"])


def _gamma(S: float, K: np.ndarray, r: float, q: float, iv: np.ndarray, T: np.ndarray) -> np.ndarray:
    d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T))


def gex_por_strike(cadenas: dict, spot: float, r: float, q: float) -> pd.DataFrame:
    """GEX en miles de millones de $ por 1 % de movimiento: calls, puts y neto por strike."""
    f = _filas_gex(cadenas, spot, r, q)
    if f.empty:
        return pd.DataFrame(columns=["strike", "calls", "puts", "neto"])
    g = _gamma(spot, f["strike"].to_numpy(), r, q, f["iv"].to_numpy(), f["T"].to_numpy())
    f["gex"] = f["signo"] * f["oi"] * g * MULT * spot ** 2 * 0.01 / 1e9
    t = f.pivot_table(index="strike", columns="signo", values="gex", aggfunc="sum").fillna(0.0)
    out = pd.DataFrame({"strike": t.index,
                        "calls": t[1.0].to_numpy() if 1.0 in t.columns else 0.0,
                        "puts": t[-1.0].to_numpy() if -1.0 in t.columns else 0.0})
    out["neto"] = out["calls"] + out["puts"]
    return out.reset_index(drop=True)


def gex_en_precios(cadenas: dict, spot: float, r: float, q: float,
                   precios: np.ndarray) -> pd.Series:
    """GEX total si el SPY estuviera en cada precio (IV de cada opción fija)."""
    f = _filas_gex(cadenas, spot, r, q)
    if f.empty:
        return pd.Series(dtype=float)
    K, iv, T = f["strike"].to_numpy(), f["iv"].to_numpy(), f["T"].to_numpy()
    w = (f["signo"] * f["oi"]).to_numpy()
    tot = [float(np.sum(w * _gamma(S, K, r, q, iv, T)) * MULT * S ** 2 * 0.01 / 1e9) for S in precios]
    return pd.Series(tot, index=np.asarray(precios, dtype=float))


def nivel_giro(curva: pd.Series, spot: float) -> float | None:
    """Precio donde la GEX total cambia de signo; el cruce más cercano al spot."""
    if curva.empty:
        return None
    x, y = curva.index.to_numpy(), curva.to_numpy()
    cruces = []
    for i in range(len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]) and y[i] != y[i + 1]:
            cruces.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float(min(cruces, key=lambda c: abs(c - spot))) if cruces else None


def muros(gex: pd.DataFrame, spot: float, rango: float = 0.08) -> dict:
    """Strike con más GEX de calls (techo) y de puts (suelo) a ±rango del spot."""
    g = gex[gex["strike"].between(spot * (1 - rango), spot * (1 + rango))]
    if g.empty:
        return {"call": None, "put": None}
    return {"call": float(g.loc[g["calls"].idxmax(), "strike"]) if g["calls"].max() > 0 else None,
            "put": float(g.loc[g["puts"].idxmin(), "strike"]) if g["puts"].min() < 0 else None}


# ──────────────────────────────────────────────────────────────────────
# VIX: Black-76 sobre el futuro
# ──────────────────────────────────────────────────────────────────────
def vix_forward(dato: dict, r: float) -> float:
    """Forward por paridad put-call en el strike con |C − P| mínimo."""
    T = _T(dato)
    c = dato["calls"].set_index("strike")["mid"]
    p = dato["puts"].set_index("strike")["mid"]
    k = c.index.intersection(p.index)
    if k.empty:
        return np.nan
    dif = (c[k] - p[k]).abs()
    k0 = dif.idxmin()
    return float(k0 + np.exp(r * T) * (c[k0] - p[k0]))


def sonrisa_vix(dato: dict, r: float) -> pd.DataFrame:
    """Sonrisa del VIX (fuera del dinero frente al forward) con IV y delta Black-76."""
    T = _T(dato)
    F = vix_forward(dato, r)
    if not np.isfinite(F) or F <= 0:
        return pd.DataFrame()
    partes = []
    for lado, tipo in (("puts", "P"), ("calls", "C")):
        d = dato[lado]
        d = d[d["strike"] < F] if tipo == "P" else d[d["strike"] >= F]
        iv = iv_b76(F, d["strike"].to_numpy(), T, r, d["mid"].to_numpy(), tipo == "C")
        partes.append(pd.DataFrame({"strike": d["strike"].to_numpy(), "tipo": tipo,
                                    "mid": d["mid"].to_numpy(), "oi": d["openInterest"].to_numpy(),
                                    "iv": iv}))
    s = pd.concat(partes, ignore_index=True).dropna(subset=["iv"]).sort_values("strike")
    if s.empty:
        return s
    d1 = (np.log(F / s["strike"]) + 0.5 * s["iv"] ** 2 * T) / (s["iv"] * np.sqrt(T))
    s["delta"] = np.where(s["tipo"] == "C", np.exp(-r * T) * norm.cdf(d1),
                          -np.exp(-r * T) * norm.cdf(-d1))
    s["m"] = s["strike"] / F - 1.0
    s.attrs["F"] = F
    return s.reset_index(drop=True)


def metricas_vix(dato: dict, r: float) -> dict:
    s = sonrisa_vix(dato, r)
    if s.empty:
        return {}
    F = s.attrs["F"]
    atm = _interp(s["strike"].to_numpy(), s["iv"].to_numpy(), F)
    c25 = iv_en_delta(s, 0.25)
    calls = s[s["tipo"] == "C"]
    k50 = calls.iloc[(calls["strike"] - 1.5 * F).abs().argsort()[:1]] if not calls.empty else calls
    return {"dte": dato["dte"], "forward": F, "atm": atm, "call25": c25, "sesgo": c25 - atm,
            "k50": float(k50["strike"].iloc[0]) if not k50.empty else np.nan,
            "precio50": float(k50["mid"].iloc[0]) if not k50.empty else np.nan,
            "iv50": float(k50["iv"].iloc[0]) if not k50.empty else np.nan,
            "oi_calls": float(dato["calls"]["openInterest"].sum()),
            "oi_puts": float(dato["puts"]["openInterest"].sum())}


def vix_principal(vix: dict, dte_min: int = 7, dte_max: int = 60) -> str | None:
    """El vencimiento del VIX con más interés abierto entre `dte_min` y `dte_max`
    días: el mensual, que es el que liquida contra el futuro que mueve al VXX.
    Los semanales tienen una fracción del interés abierto y sonrisas con huecos."""
    cand = {f: float(d["calls"]["openInterest"].sum() + d["puts"]["openInterest"].sum())
            for f, d in vix.items() if dte_min <= d["dte"] <= dte_max}
    return max(cand, key=cand.get) if cand else None


# ──────────────────────────────────────────────────────────────────────
# Foto diaria (lo que guarda scripts/update_options_snapshot.py)
# ──────────────────────────────────────────────────────────────────────
def foto(spy: dict, spot: float, r: float, q: float, vix: dict | None = None,
         rango_giro: float = 0.10) -> dict:
    spy = preparar(spy, spot, r, q)
    est = estructura(spy, spot, r, q)
    gex = gex_por_strike(spy, spot, r, q)
    precios = np.linspace(spot * (1 - rango_giro), spot * (1 + rango_giro), 81)
    curva = gex_en_precios(spy, spot, r, q, precios)
    mu = muros(gex, spot)
    out = {"spot": spot,
           "iv30": a_30_dias(est, "atm") * 100, "rr25_30": a_30_dias(est, "rr25") * 100,
           "bf25_30": a_30_dias(est, "bf25") * 100,
           "gex": float(gex["neto"].sum()) if not gex.empty else np.nan,
           "giro": nivel_giro(curva, spot), "muro_call": mu["call"], "muro_put": mu["put"]}
    principal = vix_principal(vix) if vix else None
    if principal:
        m = metricas_vix(vix[principal], r)
        if m:
            out.update({"vix_fwd": m["forward"], "vix_atm": m["atm"] * 100,
                        "vix_call25": m["call25"] * 100, "vix_sesgo": m["sesgo"] * 100})
    return out
