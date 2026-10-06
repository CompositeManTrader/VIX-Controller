"""
VIX Controller · Spread Trading Club

Todo sobre la volatilidad del VIX y el seguimiento de la estrategia VIX Inverse.
Datos: API pública de CBOE (curva y liquidación), yfinance (precios), Yahoo (opciones).
Navegación por páginas: cada sección solo calcula cuando se abre.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, date
from io import StringIO
import re, sys, time, warnings, logging, os
from zoneinfo import ZoneInfo
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata

# ── Módulos quant propios (refactor) ──────────────────────────
from vix_controller import config as cfg
from vix_controller.rates import get_risk_free_rate, get_dividend_yield
from vix_controller.quant.bs import (
    bs_call as _bs_call_mod, bs_put as _bs_put_mod,
    bs_gamma as _bs_gamma_mod, bs_iv as _bs_iv_mod,
    bs_gamma_vec as _bs_gamma_vec,
)
from vix_controller.quant.svi import fit_svi_slice as _fit_svi_slice_mod
from vix_controller.quant.vrp import compute_vrp_tracker
from vix_controller.quant.regime import fit_volatility_regime
from vix_controller.quant.cross_asset import compute_stress_signals
from vix_controller.quant import vix_inverse as vinv
from vix_controller.quant import term_structure as tsig
from vix_controller.quant.percentile import rolling_percentile as _rolling_percentile_mod
from vix_controller.quant.signals import bb_position_state
from vix_controller.quant import vix_inverse_live as _vl
from vix_controller.data import cboe as _cboe
from vix_controller import alerts as _alerts
from vix_controller.ui import theme as T
from vix_controller.ui import strategy_page, methodology_page

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

CDMX_TZ = cfg.CDMX_TZ
def now_cdmx():
    return datetime.now(CDMX_TZ)

st.set_page_config(page_title="VIX Controller · Spread Trading Club",
                   page_icon="assets/brand/favicon-64.png", layout="wide",
                   initial_sidebar_state="collapsed")


T.register_plotly_template()
st.markdown(T.CSS, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CBOE_URL = 'https://www.cboe.com/delayed_quotes/futures/future_quotes'
MONTHLY_RE = re.compile(r'^VX/[A-Z]\d+$')
MN = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LAYER — PLAYWRIGHT (browser persistente, sin relanzar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LAYER — PLAYWRIGHT (browser abre y cierra en el mismo thread)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=cfg.CACHE_TTL["cboe_scrape"], show_spinner=False)
def fetch_cboe_curve() -> pd.DataFrame:
    """
    Curva de futuros VX en vivo desde el API JSON de CBOE (el mismo que usa su
    web). Sustituye al scraping con Playwright + Chromium: sin navegador, sin
    dos minutos de arranque en frío, sin timeouts de 45 s.
    """
    log = logging.getLogger("vix_controller")
    try:
        df = _cboe.fetch_futures_quotes(today=pd.Timestamp(now_cdmx().date()))
    except _cboe.CboeError as e:
        log.warning("CBOE futuros: %s", e)
        st.session_state["curve_error"] = str(e)
        return pd.DataFrame()
    st.session_state.pop("curve_error", None)
    df["Scraped_At"] = now_cdmx().strftime("%Y-%m-%d %H:%M:%S")
    return df


@st.cache_data(ttl=cfg.CACHE_TTL["yahoo_spot"], show_spinner=False)
def fetch_index_live(symbol: str) -> dict | None:
    """Cotización retrasada de un índice CBOE (VIX, VIX3M, VIX9D...)."""
    try:
        return _cboe.fetch_index_quote(symbol)
    except _cboe.CboeError as e:
        logging.getLogger("vix_controller").warning("CBOE %s: %s", symbol, e)
        return None


# ── Circuit breaker para yfinance rate-limit ────────────────────────────────
# Streamlit Cloud comparte IPs con miles de apps → Yahoo aplica rate limits
# globales (HTTP 429 → YFRateLimitError). Cuando lo detectamos, marcamos un
# cooldown global para que TODAS las llamadas a yfinance retornen rápidamente
# sin volver a golpear el endpoint durante N minutos.
_YF_RATE_LIMIT_COOLDOWN_SEC = 15 * 60   # 15 min — alineado con duración típica del bloqueo


def _yf_is_rate_limited() -> bool:
    """True si estamos dentro de la ventana de cooldown post-rate-limit."""
    try:
        deadline = st.session_state.get("_yf_rl_until", 0)
        return time.time() < deadline
    except Exception:
        return False


def _yf_mark_rate_limited(reason: str = "") -> None:
    """Marca cooldown global ante un rate-limit detectado."""
    try:
        st.session_state["_yf_rl_until"] = time.time() + _YF_RATE_LIMIT_COOLDOWN_SEC
    except Exception:
        pass
    logging.getLogger("vix_controller").warning(
        f"yfinance rate-limit detectado{f': {reason}' if reason else ''} — "
        f"cooldown {_YF_RATE_LIMIT_COOLDOWN_SEC // 60} min")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detecta rate-limit por tipo o por mensaje (compat con varias versiones de yfinance)."""
    cls = type(exc).__name__
    if cls in ("YFRateLimitError", "RateLimitError"):
        return True
    s = str(exc).lower()
    return any(k in s for k in ("rate limit", "too many requests", "429", "throttle"))


def _yf_history_safe(symbol: str, **kwargs) -> pd.DataFrame:
    """
    Wrapper seguro alrededor de yf.Ticker(symbol).history(**kwargs).

    - Respeta el circuit breaker global: si estamos en cooldown, retorna df vacío
      sin golpear yfinance.
    - Captura YFRateLimitError y cualquier otra excepción → df vacío + log warning.
    - Marca el cooldown global si detecta rate-limit.

    El caller debe manejar df.empty como "no data available" y caer a fallback
    (parquet, cached, NaN). Nunca propaga excepción.
    """
    log = logging.getLogger("vix_controller")
    if _yf_is_rate_limited():
        log.debug(f"yf({symbol}): rate-limit cooldown activo — skip")
        return pd.DataFrame()
    try:
        return yf.Ticker(symbol).history(**kwargs)
    except Exception as e:
        if _is_rate_limit_error(e):
            _yf_mark_rate_limited(f"{symbol}.history")
        else:
            log.warning(f"yf({symbol}).history failed: {e}")
        return pd.DataFrame()


def _yf_download_safe(symbol: str, **kwargs) -> pd.DataFrame:
    """Mismo patrón para yf.download (usado con `auto_adjust`, `threads`, etc)."""
    log = logging.getLogger("vix_controller")
    if _yf_is_rate_limited():
        return pd.DataFrame()
    try:
        h = yf.download(symbol, progress=False, **kwargs)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        return h
    except Exception as e:
        if _is_rate_limit_error(e):
            _yf_mark_rate_limited(f"download({symbol})")
        else:
            log.warning(f"yf.download({symbol}) failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=cfg.CACHE_TTL["yahoo_spot"])
def fetch_vix_spot():
    """VIX en vivo: CBOE primero (sin límites de Yahoo), yfinance de respaldo."""
    q = fetch_index_live("VIX")
    if q and q.get("price"):
        prev = q.get("prev_close") or q["price"]
        return dict(price=round(q["price"], 2), prev=round(prev, 2),
                    chg=round(q["price"] - prev, 2))
    h = _yf_history_safe("^VIX", period="5d")
    if h is None or h.empty or "Close" not in h.columns:
        return None
    c = round(float(h['Close'].iloc[-1]), 2)
    p = round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else c
    return dict(price=c, prev=p, chg=round(c - p, 2))


@st.cache_data(show_spinner=False, ttl=cfg.CACHE_TTL["yahoo_spot"])
def fetch_etps():
    out = {}
    for name, sym in [("VXX","VXX"),("SVXY","SVXY"),("SVIX","SVIX"),("SPY","SPY")]:
        h = _yf_history_safe(sym, period="5d")
        if h is None or h.empty or "Close" not in h.columns:
            continue
        out[name] = dict(
            close=round(float(h['Close'].iloc[-1]), 2),
            open=round(float(h['Open'].iloc[-1]), 2) if "Open" in h.columns else None,
            prev=round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else None,
        )
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE ANALYTICS — DATA LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["edge_extra"])
def fetch_edge_extra():
    """
    Fetches VVIX, SKEW, HYG, IEF, y toda la familia VIX para barómetro.

    Estrategia híbrida:
      1. Si existe data/baro_history.parquet (actualizado por GitHub Action diario),
         lo usamos como fuente primaria — instantáneo, sin rate limits.
      2. Solo descargamos de yfinance los últimos 3 días para top-up
         (por si el parquet está desactualizado 1-2 días).
      3. Si el parquet NO existe, fallback a yfinance con period="max".

    Esto reduce ~95% de llamadas a yfinance y elimina el problema de rate-limit
    en Streamlit Cloud.
    """
    log = logging.getLogger("vix_controller")
    out = {}
    tickers = [
        ("VVIX",   "^VVIX"),
        ("SKEW",   "^SKEW"),
        ("HYG",    "HYG"),
        ("IEF",    "IEF"),
        ("VIX9D",  "^VIX9D"),
        ("VIX3M",  "^VIX3M"),
        ("VIX6M",  "^VIX6M"),
        ("VIX1Y",  "^VIX1Y"),
        ("VIX",    "^VIX"),
        # Cross-asset early warning
        ("MOVE",   "^MOVE"),
        ("LQD",    "LQD"),
        ("DXY",    "DX-Y.NYB"),
    ]

    # ── Paso 1: intentar cargar del parquet ──────────────────
    baro_df = load_baro_parquet()
    parquet_tickers = set()

    if not baro_df.empty:
        log.info(f"fetch_edge_extra: usando parquet baro ({len(baro_df):,} filas)")
        for name, _ in tickers:
            if name in baro_df.columns and baro_df[name].notna().sum() > 100:
                df_ticker = pd.DataFrame({'Close': baro_df[name].dropna()})
                out[name] = df_ticker
                parquet_tickers.add(name)

    # ── Paso 2: top-up con últimos 3 días de yfinance ────────
    # Solo si el parquet existe; si falta, hacemos download completo abajo
    if parquet_tickers:
        for name, sym in tickers:
            if name not in parquet_tickers:
                continue
            h = _yf_download_safe(sym, period="5d", auto_adjust=True, threads=False)
            if h is None or h.empty or 'Close' not in h.columns:
                continue
            try:
                if hasattr(h.index, "tz") and h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                h.index = pd.DatetimeIndex(h.index).normalize()

                # Merge: últimos días sobrescriben al parquet
                existing = out[name]
                fresh = pd.DataFrame({'Close': h['Close']})
                merged = pd.concat([existing, fresh])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                out[name] = merged
            except Exception as ex:
                log.debug(f"top-up {sym} merge failed: {ex}")
                continue

    # ── Paso 3: fallback total a yfinance si el parquet no existe ──
    missing = [name for name, _ in tickers if name not in out]
    if missing:
        if not parquet_tickers:
            log.info(f"fetch_edge_extra: parquet ausente — descargando {len(missing)} tickers de yfinance")
        else:
            log.info(f"fetch_edge_extra: parquet incompleto — completando {missing} de yfinance")

        for name, sym in tickers:
            if name in out:
                continue
            h = _yf_download_safe(sym, period="max", auto_adjust=True, threads=False)
            if h is None or h.empty:
                continue
            try:
                if hasattr(h.index, "tz") and h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                h.index = pd.DatetimeIndex(h.index).normalize()
                out[name] = h
            except Exception as ex:
                log.warning(f"fetch_edge_extra {sym} merge: {ex}")
                continue

    log.info(f"fetch_edge_extra: {len(out)} tickers cargados")
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOL SKEW & IV SURFACE — BLACK-SCHOLES IV ENGINE
# IV calculada desde cero via Brent's method (igual que functions.py
# del proyecto Volatility Surface de Georgios Drosogiannis).
# No dependemos de la IV de yfinance (ruidosa/incorrecta).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# BS y IV → delegan a vix_controller.quant.bs (testable, sin magic numbers).
_bs_call = _bs_call_mod
_bs_put  = _bs_put_mod

def _bs_iv(S, X, r, T, price, option_type, q, tol=cfg.IV_TOL):
    """IV via Brent con clamp [IV_LOWER, IV_UPPER] de config."""
    return _bs_iv_mod(S, X, r, T, price, option_type, q,
                      tol=tol, lo=cfg.IV_LOWER, hi=cfg.IV_UPPER)

# ─── Fetch raw options (sin IV de yfinance) ─────────────────────────────────

# ── Yahoo Finance API endpoints (rotación anti-rate-limit) ───────────────
_YF_ENDPOINTS = [
    "https://query1.finance.yahoo.com/v8/finance/options/{ticker}",
    "https://query2.finance.yahoo.com/v8/finance/options/{ticker}",
]

@st.cache_resource
def _get_cffi_session():
    """Sesión curl_cffi compartida (cache_resource = una sola instancia por deployment)."""
    try:
        from curl_cffi import requests as cffi_req
        s = cffi_req.Session(impersonate="chrome124")
        return s
    except ImportError:
        return None


@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["yahoo_options"])   # 1h — rate-limits de Yahoo suelen durar 15-60 min
def fetch_options_chains(ticker: str = "SPY", n_exp: int = 4) -> tuple:
    """
    Descarga opciones con triple estrategia anti-rate-limit:
    1. curl_cffi con impersonación Chrome124 (TLS fingerprint real)
       - Rota entre query1 y query2 en cada chain
       - Delay adaptativo: duplica si obtiene 429
    2. Fallback yfinance con backoff exponencial
    3. TTL=1h para no re-golpear tras rate-limit
    4. Cache negativo (session_state) para 30 min si falla completamente

    Streamlit Cloud comparte IP con miles de apps → Yahoo aplica rate limits
    agresivos. Si detectamos rate limit persistente, marcamos en session_state
    para no volver a intentar en 30 min, evitando el error visible en cada rerun.
    """
    log = logging.getLogger("vix_controller")

    # ── Cache negativo: evita re-golpear Yahoo si ya falló recientemente ──
    fail_key = f"_opts_fail_{ticker}"
    if fail_key in st.session_state:
        last_fail = st.session_state[fail_key]
        if (time.time() - last_fail) < 1800:  # 30 min
            log.info(f"Options {ticker}: cache negativo activo (falló hace "
                     f"{int((time.time() - last_fail)/60)}min) — skip")
            return {}, None
        else:
            del st.session_state[fail_key]

    def _clean(df_raw, spot_px):
        """
        Filtros de calidad estrictos para cadenas de opciones.

        REGLA FUNDAMENTAL: solo aceptar opciones con bid > 0 AND ask > 0.
        Usar lastPrice como fallback es INCORRECTO — puede ser una
        transacción de hace semanas/meses en un strike sin mercado activo.
        Un bid=0/ask=0 significa que el market-maker NO está dispuesto a
        cotizar ese strike → precio no confiable → IV no confiable.

        Filtros aplicados:
        1. bid > 0 AND ask > 0    — mercado activo
        2. moneyness ∈ [0.70, 1.30] — ±30% del spot
        3. spread < 50% del mid    — liquidez mínima
        4. openInterest ≥ 10       — algo de profundidad
        5. midPrice > 0.05         — precio mínimo válido (evita peniques)
        """
        df_c = df_raw.copy()
        for col in ["bid","ask","lastPrice","openInterest","volume","strike"]:
            df_c[col] = pd.to_numeric(df_c.get(col, 0), errors="coerce").fillna(0)

        # 1. Requiere bid Y ask positivos — sin esto, el precio no es real
        df_c = df_c[(df_c["bid"] > 0) & (df_c["ask"] > 0)]
        df_c = df_c[df_c["strike"] > 0]

        # 2. midPrice únicamente con bid/ask (no lastPrice)
        df_c["midPrice"] = 0.5 * (df_c["bid"] + df_c["ask"])

        # 3. Moneyness ±30% — strikes fuera de este rango son illíquidos
        df_c["moneyness"] = df_c["strike"] / spot_px
        df_c = df_c[df_c["moneyness"].between(0.70, 1.30)]

        # 4. Spread < 50% del mid — spread mayor indica precio basura
        spread_pct = (df_c["ask"] - df_c["bid"]) / df_c["midPrice"]
        df_c = df_c[spread_pct < 0.50]

        # 5. OI mínimo y precio mínimo
        df_c = df_c[df_c["openInterest"] >= 10]
        df_c = df_c[df_c["midPrice"] >= 0.05]

        df_c = df_c.dropna(subset=["strike","midPrice"])
        return df_c.sort_values("strike").reset_index(drop=True)

    sess = _get_cffi_session()
    if sess is not None:
        hdrs = {
            "Accept": "application/json,text/html,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://finance.yahoo.com/",
            "Origin":  "https://finance.yahoo.com",
        }
        delay = 0.8   # delay inicial entre chains; se dobla si hay 429

        for ep_idx, ep_tmpl in enumerate(_YF_ENDPOINTS):
            base = ep_tmpl.format(ticker=ticker)
            try:
                log.info(f"Options {ticker}: curl_cffi → {['query1','query2'][ep_idx]}")
                r0 = sess.get(base, headers=hdrs, timeout=20)
                if r0.status_code == 429:
                    log.warning(f"429 en {base} — probando siguiente endpoint")
                    time.sleep(3)
                    continue
                r0.raise_for_status()
                root  = r0.json()["optionChain"]["result"][0]
                spot  = float(root["quote"].get("regularMarketPrice", 0))
                if not spot: raise ValueError("spot=0")
                timestamps = root.get("expirationDates", [])
                today  = date.today()
                sel = sorted(
                    [(ts, datetime.fromtimestamp(ts).date().strftime("%Y-%m-%d"),
                      (datetime.fromtimestamp(ts).date() - today).days)
                     for ts in timestamps
                     if (datetime.fromtimestamp(ts).date() - today).days >= 7],
                    key=lambda x: x[2])[:n_exp]
                chains = {}
                for i, (ts, exp_str, dte) in enumerate(sel):
                    time.sleep(delay)
                    # Rotar endpoints por chain
                    chain_ep = _YF_ENDPOINTS[i % len(_YF_ENDPOINTS)].format(ticker=ticker)
                    try:
                        rx = sess.get(f"{chain_ep}?date={ts}", headers=hdrs, timeout=20)
                        if rx.status_code == 429:
                            delay = min(delay * 2, 5.0)
                            log.warning(f"429 en chain {exp_str} — delay→{delay:.1f}s")
                            time.sleep(delay)
                            rx = sess.get(f"{chain_ep}?date={ts}", headers=hdrs, timeout=20)
                        rx.raise_for_status()
                        opts = rx.json()["optionChain"]["result"][0]["options"][0]
                        c_df = _clean(pd.DataFrame(opts.get("calls",[])), spot)
                        p_df = _clean(pd.DataFrame(opts.get("puts", [])), spot)
                        if len(c_df) < 3 or len(p_df) < 3: continue
                        chains[exp_str] = {"calls":c_df,"puts":p_df,"dte":dte}
                        delay = max(delay * 0.85, 0.8)  # reduce delay si va bien
                    except Exception as ex:
                        log.warning(f"curl_cffi chain {ticker} {exp_str}: {ex}")
                if chains:
                    log.info(f"curl_cffi OK {ticker}: {len(chains)} chains · spot={spot:.2f}")
                    return chains, spot
            except Exception as e:
                log.warning(f"curl_cffi endpoint {ep_idx} failed: {e}")
                time.sleep(2)
                continue

    # ── Fallback: yfinance con backoff ─────────────────────────────────────
    if _yf_is_rate_limited():
        log.warning(f"Options {ticker}: cooldown global activo — skip yfinance")
        return {}, None
    log.info(f"Options {ticker}: yfinance fallback")
    try:
        t = yf.Ticker(ticker)
        def _bo(fn, label, n=5):
            for i in range(n):
                try: return fn()
                except Exception as ex:
                    if _is_rate_limit_error(ex) and i < n-1:
                        w = 2**(i+1)
                        log.warning(f"{label} RL→wait {w}s")
                        time.sleep(w)
                    elif _is_rate_limit_error(ex):
                        _yf_mark_rate_limited(label)
                        raise
                    else: raise
            return None
        exps = _bo(lambda: t.options, f"{ticker}.options")
        if not exps: return {}, None
        time.sleep(1.0)
        hist = _bo(lambda: t.history(period="2d"), f"{ticker}.hist")
        spot = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        if not spot: return {}, None
        today  = date.today()
        valid  = sorted(
            [(e,(datetime.strptime(e,"%Y-%m-%d").date()-today).days)
             for e in exps
             if (datetime.strptime(e,"%Y-%m-%d").date()-today).days >= 7],
            key=lambda x: x[1])[:n_exp]
        time.sleep(1.2)
        chains = {}; streak = 0
        for exp_str, dte in valid:
            if streak >= 2: time.sleep(15); streak = 0
            try:
                ch = _bo(lambda e=exp_str: t.option_chain(e), f"{ticker}.chain")
                if ch is None: continue
                streak = 0
                chains[exp_str] = {
                    "calls": _clean(ch.calls, spot),
                    "puts":  _clean(ch.puts,  spot),
                    "dte":   dte,
                }
                time.sleep(1.8)
            except Exception as ex:
                if any(k in str(ex).lower() for k in ["rate limit","too many","429"]):
                    streak += 1
                else:
                    log.warning(f"yfinance chain {ticker} {exp_str}: {ex}")
        log.info(f"yfinance {ticker}: {len(chains)} chains · spot={spot:.2f}")
        # Si no logramos ni una cadena, activar cache negativo
        if not chains:
            st.session_state[fail_key] = time.time()
            log.warning(f"Options {ticker}: cero chains → cache negativo 30 min")
        return chains, spot
    except Exception as e:
        log.error(f"fetch_options_chains {ticker}: {e}")
        # Cache negativo ante rate limit persistente
        if any(k in str(e).lower() for k in ["rate limit","too many","429","throttle"]):
            st.session_state[fail_key] = time.time()
            log.warning(f"Options {ticker}: rate limit → cache negativo 30 min")
        return {}, None


def compute_bs_iv_for_chains(chains: dict, spot: float, r: float, q: float) -> dict:
    """
    Calcula IV Black-Scholes (Brent) para cada opción de cada chain.
    Aplica filtros adicionales de moneyness y IV razonable.

    Rango IV aceptado: 1% - 300%
    Strikes fuera de ±30% del spot se descartan (refuerza _clean).
    """
    result = {}
    for exp_str, data in chains.items():
        dte = data["dte"]
        T   = dte / 365.0
        if T <= 0:
            continue
        for side, opt_type in [("calls","C"), ("puts","P")]:
            df = data[side].copy()
            # Redundant safety: moneyness guard in case data came through yfinance fallback
            df = df[df["moneyness"].between(0.70, 1.30)]
            if df.empty:
                continue
            df["iv"] = df.apply(
                lambda row: _bs_iv(spot, row["strike"], r, T,
                                   row["midPrice"], opt_type, q),
                axis=1
            )
            # IV range: 1% to 300% — anything outside is a data artifact
            df = df[df["iv"].notna() & (df["iv"] >= 0.01) & (df["iv"] <= 3.0)]
            data[side] = df.reset_index(drop=True)
        if len(data["calls"]) >= 3 and len(data["puts"]) >= 3:
            result[exp_str] = data
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEX — GAMMA EXPOSURE ENGINE
# GEX = OI × Gamma_BS × SpotPrice² × ContractMultiplier × ΔSpot(1%)
# Convención: Dealers son contraparte de retail → short calls, long puts
#   → dealer_gamma_calls = -OI × Gamma × S² × 100 × 0.01
#   → dealer_gamma_puts  = +OI × Gamma × S² × 100 × 0.01
# Net Dealer GEX = sum de puts - sum de calls
# Niveles positivos = dealer compra cuando S baja (soporte)
# Niveles negativos = dealer vende cuando S baja (acelera caída)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Gamma → delega al módulo (misma fórmula, factorizada)
_bs_gamma = _bs_gamma_mod


def compute_gex_profile(chains: dict, spot: float,
                        r: float | None = None, q: float | None = None,
                        contract_mult: int = cfg.GEX_CONTRACT_MULT) -> pd.DataFrame:
    """
    Calcula el GEX neto de dealers por strike, sumando todos los vencimientos.
    Usa la IV calculada por BS (columna 'iv') si está disponible,
    sino reintenta con la IV de yfinance como fallback.

    Vectorizado: gamma se computa con numpy sobre el array completo de
    strikes (antes: .apply fila a fila + .iterrows → ~10× más lento con
    chains grandes).

    Retorna DataFrame con:
      strike, calls_gex, puts_gex, net_gex (todo en USD millones)
    """
    # Si r/q no se pasan, cargar dinámicamente (fallback a cfg si red falla)
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")

    frames = []
    for exp_str, data in chains.items():
        dte = data["dte"]
        T   = dte / 365.0
        if T <= 0:
            continue

        for side, sign in [("calls", -1), ("puts", +1)]:
            df = data[side]
            if df.empty:
                continue
            # Usar IV BS si disponible, si no yfinance impliedVolatility
            if "iv" in df.columns and df["iv"].notna().any():
                iv_col = "iv"
            elif "impliedVolatility" in df.columns:
                iv_col = "impliedVolatility"
            else:
                continue

            df = df[df[iv_col].notna() & (df[iv_col] > 0.005)]
            if df.empty:
                continue

            gamma = _bs_gamma_vec(spot, df["strike"].values, r, T,
                                  df[iv_col].values, q)
            # GEX en dólares: OI × Gamma × S² × multiplier × 1% move
            gex_usd = (sign * df["openInterest"].values * gamma
                       * (spot ** 2) * contract_mult * 0.01)
            frames.append(pd.DataFrame({
                "strike":  df["strike"].values,
                "side":    side,
                "dte":     dte,
                "gex_usd": gex_usd,
                "oi":      df["openInterest"].values,
            }))

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    # Agrupar por strike
    gex_by_strike = (
        df_all.groupby(["strike", "side"])["gex_usd"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Garantizar columnas
    for col in ["calls", "puts"]:
        if col not in gex_by_strike.columns:
            gex_by_strike[col] = 0.0

    gex_by_strike["calls_gex"] = gex_by_strike["calls"] / 1e6   # → millones USD
    gex_by_strike["puts_gex"]  = gex_by_strike["puts"]  / 1e6
    gex_by_strike["net_gex"]   = gex_by_strike["puts_gex"] + gex_by_strike["calls_gex"]

    return gex_by_strike.sort_values("strike").reset_index(drop=True)


def compute_gex_summary(gex_df: pd.DataFrame, spot: float) -> dict:
    """
    Métricas clave del GEX profile:
    - gamma_flip: strike donde GEX neto cambia de positivo a negativo
    - total_gex:  GEX neto total (positivo = mercado anclado, negativo = amplificador)
    - biggest_call_wall: strike con mayor GEX de calls (resistencia)
    - biggest_put_wall:  strike con mayor GEX de puts (soporte)
    - gex_percentile:    % de OTM strikes con GEX positivo
    """
    if gex_df.empty or spot <= 0:
        return {}

    total_gex = float(gex_df["net_gex"].sum())

    # Gamma flip: strike donde la suma acumulada cambia de signo
    df_sorted = gex_df.sort_values("strike")
    cumsum    = df_sorted["net_gex"].cumsum().values
    flip_idx  = np.where(np.diff(np.sign(cumsum)))[0]
    if len(flip_idx) > 0:
        flip_strike = float(df_sorted["strike"].iloc[flip_idx[0]])
    else:
        flip_strike = None

    # Paredes
    call_wall = float(gex_df.loc[gex_df["calls_gex"].abs().idxmax(), "strike"]) \
                if not gex_df.empty else None
    put_wall  = float(gex_df.loc[gex_df["puts_gex"].abs().idxmax(), "strike"]) \
                if not gex_df.empty else None

    # Zona OTM: strikes en ±15% del spot
    otm_zone = gex_df[gex_df["strike"].between(spot * 0.85, spot * 1.15)]
    pct_pos  = (otm_zone["net_gex"] > 0).mean() * 100 if not otm_zone.empty else None

    return {
        "total_gex":    round(total_gex, 2),
        "flip_strike":  round(flip_strike, 2) if flip_strike else None,
        "call_wall":    round(call_wall, 2)   if call_wall   else None,
        "put_wall":     round(put_wall, 2)    if put_wall    else None,
        "pct_pos_otm":  round(pct_pos, 1)     if pct_pos is not None else None,
        "regime":       "POSITIVE" if total_gex > 0 else "NEGATIVE",
    }


def build_gex_profile_chart(gex_df: pd.DataFrame, spot: float,
                             summary: dict, ticker: str = "SPY",
                             strike_range_pct: float = 0.12) -> go.Figure:
    """
    Gráfico de barras GEX por strike:
    - Barras verdes: GEX positivo (zona pin, dealers compran dips)
    - Barras rojas:  GEX negativo (zona acelerador, dealers venden dips)
    - Línea spot, gamma flip, call wall, put wall
    """
    fig = go.Figure()
    if gex_df.empty or spot <= 0:
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    df = gex_df[gex_df["strike"].between(lo, hi)].copy()
    if df.empty:
        return fig

    colors = ["#16C784" if v >= 0 else "#EA3943" for v in df["net_gex"]]

    fig.add_trace(go.Bar(
        x=df["strike"], y=df["net_gex"],
        name="Net GEX (dealers)",
        marker_color=colors,
        opacity=0.8,
        hovertemplate="Strike: $%{x:.0f}<br>Net GEX: $%{y:.2f}M<extra></extra>",
    ))

    # Spot
    fig.add_vline(x=spot, line_dash="solid", line_color="#F4F5F6", line_width=2,
                  annotation_text=f"  Spot ${spot:.1f}",
                  annotation_font=dict(size=10, color="#F4F5F6", family="JetBrains Mono"))

    # Gamma flip
    gf = summary.get("flip_strike")
    if gf:
        fig.add_vline(x=gf, line_dash="dash", line_color="#F5A623", line_width=1.5,
                      annotation_text=f"  Flip ${gf:.0f}",
                      annotation_font=dict(size=9, color="#F5A623", family="JetBrains Mono"),
                      annotation_position="top right")

    # Call wall
    cw = summary.get("call_wall")
    if cw and lo <= cw <= hi:
        fig.add_vline(x=cw, line_dash="dot", line_color="#C9821A", line_width=1.5,
                      annotation_text=f"  Call Wall ${cw:.0f}",
                      annotation_font=dict(size=9, color="#C9821A", family="JetBrains Mono"),
                      annotation_position="bottom right")

    # Put wall
    pw = summary.get("put_wall")
    if pw and lo <= pw <= hi:
        fig.add_vline(x=pw, line_dash="dot", line_color="#9AA1A9", line_width=1.5,
                      annotation_text=f"  Put Wall ${pw:.0f}",
                      annotation_font=dict(size=9, color="#9AA1A9", family="JetBrains Mono"),
                      annotation_position="bottom left")

    regime    = summary.get("regime", "?")
    total_gex = summary.get("total_gex", 0)
    regime_clr = "#16C784" if regime == "POSITIVE" else "#EA3943"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>GEX Profile — {ticker}</b>"
                f"<sup>  Net GEX: <span style='color:{regime_clr}'>${total_gex:+.1f}M · {regime}</span>"
                f"  |  Gamma Flip: ${gf:.0f}" if gf else
                f"<b>GEX Profile — {ticker}</b>"
                f"<sup>  Net GEX: ${total_gex:+.1f}M · {regime}</sup>"
            ),
            font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5,
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=55, r=30, t=65, b=50),
        xaxis=dict(
            title=dict(text="Strike Price ($)", font=dict(size=10, color="#9AA1A9")),
            gridcolor="#1C1F24",
            tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono"),
            tickprefix="$",
        ),
        yaxis=dict(
            title=dict(text="Net GEX ($M / 1% move)", font=dict(size=10, color="#9AA1A9")),
            gridcolor="#1C1F24",
            tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono"),
            ticksuffix="M",
            zeroline=True, zerolinecolor="#3A3F47", zerolinewidth=2,
        ),
        showlegend=False,
        hovermode="x unified",
        bargap=0.1,
    )
    return fig


def build_gex_by_expiry_chart(chains: dict, spot: float,
                               r: float | None = None, q: float | None = None) -> go.Figure:
    """
    GEX total por vencimiento — muestra qué expiración concentra más gamma.
    """
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")
    fig = go.Figure()
    if not chains or not spot:
        return fig

    rows = []
    for exp_str, data in sorted(chains.items(), key=lambda x: x[1]["dte"]):
        dte = data["dte"]; T = dte / 365.0
        if T <= 0:
            continue
        net = 0.0
        for side, sign in [("calls", -1), ("puts", +1)]:
            df = data[side].copy()
            if df.empty:
                continue
            iv_col = "iv" if "iv" in df.columns and df["iv"].notna().any() \
                     else "impliedVolatility"
            if iv_col not in df.columns:
                continue
            df = df[df[iv_col].notna() & (df[iv_col] > 0.005)]
            for _, row in df.iterrows():
                g = _bs_gamma(spot, row["strike"], r, T, row[iv_col], q)
                net += sign * row["openInterest"] * g * spot**2 * 100 * 0.01
        rows.append({"exp": exp_str, "dte": dte,
                     "net_gex_m": net / 1e6})

    if not rows:
        return fig

    df_e = pd.DataFrame(rows)
    colors = ["#16C784" if v >= 0 else "#EA3943" for v in df_e["net_gex_m"]]

    fig.add_trace(go.Bar(
        x=df_e["exp"], y=df_e["net_gex_m"],
        marker_color=colors, opacity=0.8,
        hovertemplate="%{x}<br>GEX: $%{y:.2f}M<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="#3A3F47", line_width=1.5)
    fig.update_layout(
        title=dict(
            text="<b>GEX por Vencimiento</b><sup>  Qué expiración concentra más gamma</sup>",
            font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5,
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"),
                   title=dict(text="Vencimiento", font=dict(size=10, color="#9AA1A9"))),
        yaxis=dict(title=dict(text="Net GEX ($M)", font=dict(size=10, color="#9AA1A9")),
                   gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"),
                   ticksuffix="M", zeroline=True, zerolinecolor="#3A3F47"),
        showlegend=False, bargap=0.2,
    )
    return fig


# ─── Métricas de skew (usa columna 'iv' BS) ────────────────────────────────
def build_gex_delta_exposure_chart(chains: dict, spot: float,
                                    r: float | None = None, q: float | None = None,
                                    strike_range_pct: float = 0.15) -> go.Figure:
    """
    DEX (Delta Exposure): muestra el delta neto de dealers por strike.
    Donde DEX cruza cero = nivel de máximo dolor (max pain).
    Complementa GEX para entender hacia dónde se mueve el precio.
    """
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")
    fig = go.Figure()
    if not chains or not spot:
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    rows = []

    for exp_str, data in chains.items():
        dte = data["dte"]; T = dte / 365.0
        if T <= 0: continue
        for side, sign, opt_type in [("calls", -1, "C"), ("puts", +1, "P")]:
            df = data[side].copy()
            if df.empty: continue
            iv_col = "iv" if "iv" in df.columns else "impliedVolatility"
            df = df[df.get(iv_col, pd.Series([np.nan]*len(df))).notna()]
            df = df[df["strike"].between(lo, hi)]
            if df.empty: continue

            for _, row in df.iterrows():
                iv = float(row.get(iv_col, 0) or 0)
                if iv <= 0: continue
                K = row["strike"]; oi = row["openInterest"]
                d1 = (np.log(spot/K) + (r - q + 0.5*iv**2)*T) / (iv*np.sqrt(T)) if T > 0 and iv > 0 else 0
                from scipy.stats import norm as _norm
                delta = np.exp(-q*T) * _norm.cdf(d1) if opt_type == "C" else -np.exp(-q*T)*_norm.cdf(-d1)
                # Dealer delta: short calls = -delta, long puts = +delta (approx)
                rows.append({"strike": K, "delta_usd": sign * oi * delta * spot * 100 / 1e6})

    if not rows:
        return fig

    df_d = pd.DataFrame(rows).groupby("strike")["delta_usd"].sum().reset_index()
    colors_d = ["#16C784" if v >= 0 else "#EA3943" for v in df_d["delta_usd"]]

    fig.add_trace(go.Bar(
        x=df_d["strike"], y=df_d["delta_usd"],
        marker_color=colors_d, opacity=0.75, name="Delta Exposure",
        hovertemplate="Strike: $%{x:.0f}<br>DEX: $%{y:.2f}M<extra></extra>"))

    fig.add_vline(x=spot, line_dash="solid", line_color="#F4F5F6", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F4F5F6"))
    fig.add_hline(y=0, line_dash="dash", line_color="#3A3F47", line_width=1.5)

    # Max pain: strike con mayor dolor para holders de opciones
    combined_all = pd.concat([
        data["calls"].assign(type="C") for data in chains.values()
    ] + [data["puts"].assign(type="P") for data in chains.values()])
    if not combined_all.empty:
        strikes_all = sorted(combined_all["strike"].unique())
        pain = []
        for s in strikes_all:
            calls_loss = combined_all[(combined_all["type"]=="C") & (combined_all["strike"] <= s)].apply(
                lambda r: (s - r["strike"]) * r["openInterest"] * 100, axis=1).sum()
            puts_loss  = combined_all[(combined_all["type"]=="P") & (combined_all["strike"] >= s)].apply(
                lambda r: (r["strike"] - s) * r["openInterest"] * 100, axis=1).sum()
            pain.append({"strike": s, "pain": calls_loss + puts_loss})
        if pain:
            df_pain = pd.DataFrame(pain)
            mp = float(df_pain.loc[df_pain["pain"].idxmin(), "strike"])
            fig.add_vline(x=mp, line_dash="dot", line_color="#F5A623", line_width=2,
                          annotation_text=f"  Max Pain ${mp:.0f}",
                          annotation_font=dict(size=9, color="#F5A623", family="JetBrains Mono"))

    fig.update_layout(
        title=dict(text="<b>Delta Exposure (DEX)</b><sup>  Presión de hedging por strike · Max Pain marcado</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="DEX ($M)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9"), ticksuffix="M",
                   zeroline=True, zerolinecolor="#3A3F47", zerolinewidth=2),
        showlegend=False, hovermode="x unified", bargap=0.1)
    return fig


def build_gex_vanna_charm_chart(chains: dict, spot: float,
                                 r: float | None = None, q: float | None = None,
                                 strike_range_pct: float = 0.15) -> go.Figure:
    """
    Vanna + Charm por strike.
    Vanna = ∂Delta/∂Vol = ∂Gamma/∂Spot (efecto de cambio de IV sobre delta)
    Charm = ∂Delta/∂t  (decaimiento del delta con el tiempo, importante en expiry weeks)
    Ambos generan flujos de hedging autónomos — críticos para predecir pinning en expiración.
    """
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")
    from scipy.stats import norm as _norm
    fig = go.Figure()
    if not chains or not spot:
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    vanna_rows, charm_rows = [], []

    for data in chains.values():
        dte = data["dte"]; T = dte / 365.0
        if T <= 0: continue
        for side, sign in [("calls", -1), ("puts", +1)]:
            df = data[side].copy()
            if df.empty: continue
            iv_col = "iv" if "iv" in df.columns else "impliedVolatility"
            df = df[df["strike"].between(lo, hi)].copy()
            if df.empty: continue
            for _, row in df.iterrows():
                iv = float(row.get(iv_col, 0) or 0)
                K = row["strike"]; oi = row["openInterest"]
                if iv <= 0 or T <= 0: continue
                d1 = (np.log(spot/K) + (r - q + 0.5*iv**2)*T) / (iv*np.sqrt(T))
                d2 = d1 - iv*np.sqrt(T)
                pdf_d1 = _norm.pdf(d1)
                # Vanna: dDelta/dVol per $1M notional
                vanna = np.exp(-q*T) * pdf_d1 * d2 / iv if iv > 0 else 0
                # Charm: dDelta/dt
                charm = np.exp(-q*T) * pdf_d1 * (
                    2*(r-q)*T - d2*iv*np.sqrt(T)) / (2*T*iv*np.sqrt(T)) if T > 0 else 0
                vanna_rows.append({"strike": K, "val": sign * oi * vanna * spot * 100 / 1e6})
                charm_rows.append({"strike": K, "val": sign * oi * charm * 100 / 1e3})

    if not vanna_rows:
        return fig

    df_v = pd.DataFrame(vanna_rows).groupby("strike")["val"].sum().reset_index()
    df_c = pd.DataFrame(charm_rows).groupby("strike")["val"].sum().reset_index()

    fig.add_trace(go.Bar(
        x=df_v["strike"], y=df_v["val"],
        name="Vanna ($M/vol pt)",
        marker_color="#F4F5F6", opacity=0.7,
        hovertemplate="Strike: $%{x:.0f}<br>Vanna: %{y:.3f}M<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=df_c["strike"], y=df_c["val"],
        name="Charm (×1000)", yaxis="y2",
        line=dict(color="#C9821A", width=2),
        hovertemplate="Strike: $%{x:.0f}<br>Charm: %{y:.3f}<extra></extra>"))

    fig.add_vline(x=spot, line_dash="dash", line_color="#F4F5F6", line_width=1.5,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F4F5F6"))
    fig.add_hline(y=0, line_dash="dot", line_color="#3A3F47", line_width=1)

    fig.update_layout(
        title=dict(text="<b>Vanna & Charm</b><sup>  Flujos de hedging por cambio de vol (Vanna) y tiempo (Charm)</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=55, r=60, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="Vanna ($M)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#F4F5F6")),
        yaxis2=dict(title="Charm (×1000)", overlaying="y", side="right",
                    tickfont=dict(size=9, color="#C9821A"), showgrid=False),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#F4F5F6", family="JetBrains Mono")),
        hovermode="x unified", bargap=0.1)
    return fig


def build_gex_cumulative_chart(gex_df: pd.DataFrame, spot: float,
                                strike_range_pct: float = 0.15) -> go.Figure:
    """
    GEX acumulado: suma corrida de GEX desde strikes bajos a altos.
    El cruce por cero = Gamma Flip confirmado visualmente.
    La pendiente muestra la "fuerza" del régimen.
    """
    fig = go.Figure()
    if gex_df.empty or spot <= 0:
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    df = gex_df[gex_df["strike"].between(lo, hi)].sort_values("strike").copy()
    if df.empty:
        return fig

    df["gex_cumsum"] = df["net_gex"].cumsum()
    zero_crossings = df[df["gex_cumsum"] * df["gex_cumsum"].shift(1) < 0]
    colors_area = ["rgba(22,199,132,0.15)" if v >= 0 else "rgba(234,57,67,0.15)"
                   for v in df["gex_cumsum"]]

    fig.add_trace(go.Scatter(
        x=df["strike"], y=df["gex_cumsum"],
        name="GEX Acumulado",
        line=dict(color="#9AA1A9", width=2.5),
        fill="tozeroy", fillcolor="rgba(154,161,169,0.1)",
        hovertemplate="Strike: $%{x:.0f}<br>GEX Acum: $%{y:.2f}M<extra></extra>"))

    fig.add_vline(x=spot, line_dash="solid", line_color="#F4F5F6", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F4F5F6"))
    fig.add_hline(y=0, line_dash="dash", line_color="#F5A623", line_width=2,
                  annotation_text="  Gamma Flip Zone",
                  annotation_font=dict(size=9, color="#F5A623"))

    for _, row in zero_crossings.iterrows():
        fig.add_vline(x=row["strike"], line_dash="dot", line_color="#F5A623",
                      line_width=1.5)

    flip_pct = (zero_crossings["strike"].iloc[0] / spot - 1)*100 if not zero_crossings.empty else None
    subtitle = f"Flip en ${zero_crossings['strike'].iloc[0]:.0f} ({flip_pct:+.1f}% del spot)" if flip_pct else "Sin flip visible en rango"

    fig.update_layout(
        title=dict(text=f"<b>GEX Acumulado</b><sup>  {subtitle}</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="GEX Acum ($M)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9"), ticksuffix="M",
                   zeroline=True, zerolinecolor="#F5A623", zerolinewidth=2),
        showlegend=False, hovermode="x unified")
    return fig


def build_gex_expected_move_chart(gex_df: pd.DataFrame, chains: dict, spot: float,
                                   strike_range_pct: float = 0.15) -> go.Figure:
    """
    Expected Move implícito: usa la IV ATM del vencimiento más próximo
    para calcular el rango esperado ±1σ y ±2σ.
    Superpone niveles GEX clave para ver si los muros actúan como límites del rango.
    """
    from scipy.stats import norm as _norm
    fig = go.Figure()
    if gex_df.empty or not chains or spot <= 0:
        return fig

    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    df = gex_df[gex_df["strike"].between(lo, hi)].sort_values("strike").copy()
    if df.empty:
        return fig

    # GEX profile
    colors_gex = ["#16C784" if v >= 0 else "#EA3943" for v in df["net_gex"]]
    fig.add_trace(go.Bar(
        x=df["strike"], y=df["net_gex"].abs(),
        marker_color=colors_gex, opacity=0.4, name="|GEX|",
        hovertemplate="Strike: $%{x:.0f}<br>|GEX|: $%{y:.2f}M<extra></extra>"))

    # ATM IV del front month
    front_exp = sorted(chains.keys(), key=lambda x: chains[x]["dte"])[0]
    front_data = chains[front_exp]
    dte_f = front_data["dte"]
    T_f   = dte_f / 365.0
    iv_col = "iv" if "iv" in front_data["calls"].columns else "impliedVolatility"
    atm_c = front_data["calls"][front_data["calls"]["moneyness"].between(0.97, 1.03)]
    atm_p = front_data["puts"][front_data["puts"]["moneyness"].between(0.97, 1.03)]
    atm_all = pd.concat([atm_c, atm_p])

    if not atm_all.empty and iv_col in atm_all.columns and T_f > 0:
        atm_iv = float(np.average(atm_all[iv_col].values,
                                   weights=atm_all["openInterest"].values + 1))
        # Expected move = spot × IV × √T
        em1 = spot * atm_iv * np.sqrt(T_f)
        em2 = em1 * 2

        for em, lbl, clr, dash in [
            (em1, f"±1σ ({dte_f}d exp)", "#F5A623", "dash"),
            (em2, f"±2σ ({dte_f}d exp)", "#EA3943", "dot"),
        ]:
            for sign in [1, -1]:
                fig.add_vline(
                    x=spot + sign*em,
                    line_dash=dash, line_color=clr, line_width=1.5,
                    annotation_text=f"  {lbl}" if sign > 0 else None,
                    annotation_font=dict(size=8, color=clr, family="JetBrains Mono"))

        # Agrega texto del ATM IV
        fig.add_annotation(
            x=hi*0.99, y=df["net_gex"].abs().max()*0.9,
            text=f"ATM IV: {atm_iv*100:.1f}%<br>±1σ: ±${em1:.1f}",
            showarrow=False,
            font=dict(size=9, color="#F5A623", family="JetBrains Mono"),
            align="right")

    fig.add_vline(x=spot, line_dash="solid", line_color="#F4F5F6", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F4F5F6"))

    fig.update_layout(
        title=dict(text="<b>Expected Move + GEX Levels</b>"
                        "<sup>  Rango ±1σ/±2σ vs muros de gamma</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="|GEX| ($M)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9"), ticksuffix="M"),
        showlegend=False, hovermode="x unified", bargap=0.1)
    return fig


def fit_svi_slice(strikes: np.ndarray, ivs: np.ndarray, F: float,
                  T: float,
                  max_iter: int = cfg.SVI_MAX_ITER) -> dict | None:
    """
    Wrapper de vix_controller.quant.svi.fit_svi_slice.

    T (time-to-expiry EN AÑOS) es parámetro obligatorio para normalizar
    varianza total w = IV² · T correctamente por slice. El fallback legacy
    T=1.0 se eliminó: sesgaba la calibración de expirations cortos.

    Ver vix_controller/quant/svi.py para la docstring completa (Gatheral 2004,
    Durrleman 2005, constraints butterfly).
    """
    return _fit_svi_slice_mod(strikes, ivs, F, T, max_iter=max_iter)


def forecast_vol_surface(chains: dict, spot: float,
                          r: float | None = None, q: float | None = None,
                          iv_change_pct: float = -0.15,
                          n_grid: int = 50) -> dict:
    """
    Forecasta la superficie de vol para mañana usando SVI fitted per slice.
    Metodología:
      1. Fittear SVI al smile actual por cada vencimiento
      2. Modelar el cambio esperado de vol: IV_t+1 ≈ IV_t × (1 + Δ)
         donde Δ es el escenario de cambio (default: -15%, mean-reversion).
      3. Usar HAR-RV forecast para ajustar el nivel ATM esperado.
      4. Identificar opciones con mayor theta positiva esperada
         (opciones donde IV_actual >> IV_forecasted = prima vendible).

    RETORNA:
      - svi_fits: {exp_str: SVI params + fitted smile}
      - forecast_chains: IV forecasted surface
      - sell_candidates: top opciones con P&L esperado positivo
    """
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")
    log = logging.getLogger("vix_controller")

    svi_fits = {}
    for exp_str, data in chains.items():
        dte = data["dte"]; T = dte / cfg.CAL_DAYS_YEAR
        if T <= 0: continue
        puts_f  = data["puts"][data["puts"]["moneyness"].between(0.80, 1.02)]
        calls_f = data["calls"][data["calls"]["moneyness"].between(0.98, 1.20)]
        combo   = pd.concat([puts_f, calls_f]).drop_duplicates("strike").sort_values("strike")
        iv_col  = "iv" if "iv" in combo.columns else "impliedVolatility"
        combo   = combo[combo[iv_col].notna() & (combo[iv_col] > cfg.IV_LOWER)]
        if len(combo) < cfg.SVI_MIN_POINTS: continue

        # Forward al vencimiento real (no 30d fijo) → consistente con T del slice
        F_T = spot * np.exp((r - q) * T)
        fit = fit_svi_slice(combo["strike"].values, combo[iv_col].values, F_T, T=T)
        if fit:
            # F_T se guarda por slice: cada vencimiento tiene su propio forward.
            svi_fits[exp_str] = {**fit, "dte": dte, "combo": combo, "F_T": F_T}
            log.info(f"SVI {exp_str} (T={T:.3f}y): R²={fit['r2']:.3f} "
                     f"ρ={fit['rho']:.3f} b={fit['b']:.3f}")

    if not svi_fits:
        return {}

    # Grid de moneyness para superficie
    k_grid = np.linspace(-0.25, 0.20, n_grid)

    # Forecast: aplica cambio de vol + mean-reversion ATM
    forecast_data = {}
    for exp_str, fit in svi_fits.items():
        a, b, rho, m, sigma = fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"]
        dte = fit["dte"]; T = dte / 365.0
        # IV forecasted smile en el grid
        disc    = np.maximum((k_grid - m)**2 + sigma**2, 1e-12)
        w_fc    = a*(1+iv_change_pct) + b * (rho*(k_grid-m) + np.sqrt(disc))
        iv_fc   = np.sqrt(np.maximum(w_fc, 0))
        # IV actual en el mismo grid
        w_cur   = a + b * (rho*(k_grid-m) + np.sqrt(disc))
        iv_cur  = np.sqrt(np.maximum(w_cur, 0))

        forecast_data[exp_str] = {
            "k_grid":  k_grid,
            "iv_cur":  iv_cur,
            "iv_fc":   iv_fc,
            "iv_drop": iv_cur - iv_fc,   # IV que "se derrite"
            "dte":     dte,
        }

    # Sell candidates: buscar strikes donde IV actual >> forecasted
    sell_candidates = []
    for exp_str, data in chains.items():
        if exp_str not in svi_fits: continue
        fit = svi_fits[exp_str]
        dte = data["dte"]; T = dte / 365.0
        for side in ["puts", "calls"]:
            df = data[side].copy()
            if df.empty: continue
            iv_col = "iv" if "iv" in df.columns else "impliedVolatility"
            df = df[df[iv_col].notna()
                    & (df["strike"].between(spot * cfg.SELL_MONEYNESS_LO,
                                            spot * cfg.SELL_MONEYNESS_HI))]
            if df.empty: continue

            F_T = spot * np.exp((r - q) * T)   # forward al vencimiento del slice
            for _, row in df.iterrows():
                K   = row["strike"]; iv_c = float(row.get(iv_col, 0) or 0)
                oi  = row["openInterest"]; mid = row.get("midPrice", 0)
                if K <= 0 or iv_c <= 0 or mid <= 0: continue
                if not (cfg.SELL_MONEYNESS_LO <= K/spot <= cfg.SELL_MONEYNESS_HI):
                    continue  # ±18% max — fuera de ahí la IV es ruido

                k_  = np.log(K / F_T)
                disc_ = np.maximum((k_ - fit["m"])**2 + fit["sigma"]**2, 1e-12)
                w_fc  = fit["a"]*(1+iv_change_pct) + fit["b"]*(fit["rho"]*(k_-fit["m"]) + np.sqrt(disc_))
                iv_fc = float(np.sqrt(max(w_fc, 0)))

                iv_drop_abs = iv_c - iv_fc     # cuántos puntos de vol se derriten
                if iv_drop_abs <= 0: continue

                # Vega (sensitivity to IV): dV/dIV ≈ S*√T*N'(d1)
                from scipy.stats import norm as _norm
                d1 = (np.log(spot/K) + (r-q+0.5*iv_c**2)*T) / (iv_c*np.sqrt(T)) if T > 0 else 0
                vega_per_contract = spot * np.sqrt(T) * _norm.pdf(d1) * 100

                # P&L esperado por vender 1 contrato si IV cae iv_drop_abs
                pnl_expected = vega_per_contract * iv_drop_abs
                # Theta diaria
                theta_daily = mid * 0.015 if T > 0 else 0   # aprox

                moneyness_pct = (K/spot - 1)*100
                sell_candidates.append({
                    "Tipo":          side[:-1].upper(),
                    "Exp":           exp_str,
                    "DTE":           dte,
                    "Strike":        round(K, 0),
                    "Dist Spot %":   round(moneyness_pct, 1),
                    "IV Actual %":   round(iv_c*100, 1),
                    "IV Forecast %": round(iv_fc*100, 1),
                    "IV Drop pts":   round(iv_drop_abs*100, 2),
                    "Vega/ct ($)":   round(vega_per_contract, 1),
                    "P&L Esp. ($)":  round(pnl_expected, 1),
                    "Mid $":         round(mid, 2),
                    "OI":            int(oi),
                })

    # Ordenar por P&L esperado descendente
    sell_df = pd.DataFrame(sell_candidates)
    if not sell_df.empty:
        sell_df = sell_df.sort_values("P&L Esp. ($)", ascending=False).reset_index(drop=True)

    return {
        "svi_fits":      svi_fits,
        "forecast_data": forecast_data,
        "sell_df":       sell_df,
        "iv_change_pct": iv_change_pct,
        "spot":          spot,
    }


def build_svi_smile_chart(forecast_result: dict, exp_str: str,
                           spot: float) -> go.Figure:
    """Smile actual vs forecasted para un vencimiento específico."""
    fig = go.Figure()
    if not forecast_result or "forecast_data" not in forecast_result:
        return fig

    fc = forecast_result["forecast_data"].get(exp_str)
    fit = forecast_result["svi_fits"].get(exp_str)
    if fc is None or fit is None:
        return fig

    k_pct = fc["k_grid"] * 100
    iv_change_pct = forecast_result.get("iv_change_pct", -0.15)
    dte = fc["dte"]

    # Puntos observados — log-moneyness vs forward del slice (F_T), no spot.
    combo = fit.get("combo", pd.DataFrame())
    iv_col = "iv" if "iv" in combo.columns else "impliedVolatility"
    F_T = fit.get("F_T", spot)   # fallback a spot si fit antiguo no lo guardó
    if not combo.empty and iv_col in combo.columns:
        obs_k   = np.log(combo["strike"].values / F_T) * 100
        obs_iv  = combo[iv_col].values * 100
        fig.add_trace(go.Scatter(
            x=obs_k, y=obs_iv, mode="markers",
            name="Observado",
            marker=dict(color="#9AA1A9", size=6, opacity=0.7),
            hovertemplate="k: %{x:.1f}%<br>IV obs: %{y:.1f}%<extra></extra>"))

    # Fitted SVI actual
    fig.add_trace(go.Scatter(
        x=k_pct, y=fc["iv_cur"]*100, mode="lines",
        name="SVI Actual",
        line=dict(color="#F4F5F6", width=2.5),
        hovertemplate="k: %{x:.1f}%<br>IV SVI: %{y:.1f}%<extra></extra>"))

    # SVI Forecasted
    fig.add_trace(go.Scatter(
        x=k_pct, y=fc["iv_fc"]*100, mode="lines",
        name=f"SVI Forecast ({iv_change_pct:+.0%})",
        line=dict(color="#16C784", width=2.5, dash="dash"),
        hovertemplate="k: %{x:.1f}%<br>IV forecast: %{y:.1f}%<extra></extra>"))

    # Área de oportunidad (IV drop)
    fig.add_trace(go.Scatter(
        x=list(k_pct)+list(k_pct[::-1]),
        y=list(fc["iv_cur"]*100)+list(fc["iv_fc"]*100)[::-1],
        fill="toself", fillcolor="rgba(22,199,132,0.12)",
        line=dict(width=0), name="IV Drop (oportunidad)", hoverinfo="skip"))

    fig.add_vline(x=0, line_dash="dash", line_color="#9AA1A9", line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=9, color="#9AA1A9"))

    fig.update_layout(
        title=dict(
            text=f"<b>SVI Smile — {exp_str} ({dte}d)</b>"
                 f"<sup>  Azul=actual · Verde=forecast · zona=oportunidad de venta</sup>",
            font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Log-moneyness k (%)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9", family="JetBrains Mono"), ticksuffix="%",
                   zeroline=True, zerolinecolor="#262A30"),
        yaxis=dict(title="IV (%)", gridcolor="#1C1F24",
                   tickfont=dict(size=9, color="#9AA1A9"), ticksuffix="%"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#F4F5F6", family="JetBrains Mono")),
        hovermode="x unified")
    return fig


def build_forecast_surface_chart(forecast_result: dict, spot: float,
                                   view: str = "drop") -> go.Figure:
    """
    Superficie 3D del cambio de IV (drop) esperado.
    view='drop': Z = IV_actual - IV_forecast (cuánto se derrite)
    view='forecast': Z = IV_forecast (nivel esperado mañana)
    """
    fig = go.Figure()
    if not forecast_result or "forecast_data" not in forecast_result:
        return fig

    fc_data = forecast_result["forecast_data"]
    if not fc_data:
        return fig

    # Construir arrays para Surface
    dtes = sorted([v["dte"] for v in fc_data.values()])
    k_grid = list(fc_data.values())[0]["k_grid"] * 100

    Z_rows = []
    for exp_str in sorted(fc_data.keys(), key=lambda x: fc_data[x]["dte"]):
        fd = fc_data[exp_str]
        if view == "drop":
            Z_rows.append(fd["iv_drop"] * 100)
        else:
            Z_rows.append(fd["iv_fc"] * 100)

    if not Z_rows:
        return fig

    Z = np.array(Z_rows)

    colorscale = ([[0.0,"#5E656E"],[0.3,"#16C784"],[0.6,"#F5A623"],[1.0,"#EA3943"]]
                  if view == "drop" else
                  [[0.0,"#3A3F47"],[0.3,"#C7CCD1"],[0.6,"#16C784"],[0.8,"#F5A623"],[1.0,"#EA3943"]])

    fig.add_trace(go.Surface(
        x=k_grid, y=dtes, z=Z,
        colorscale=colorscale,
        colorbar=dict(title=dict(text="∆IV pts" if view=="drop" else "IV%",
                                  font=dict(color="#9AA1A9", size=10)),
                      tickfont=dict(color="#9AA1A9", size=9), len=0.6, thickness=12),
        hovertemplate="k: %{x:.1f}%<br>DTE: %{y}d<br>" +
                      ("IV Drop: %{z:.1f} pts<extra></extra>" if view=="drop"
                       else "IV Forecast: %{z:.1f}%<extra></extra>"),
        opacity=0.9))

    title_map = {
        "drop":     "Superficie de IV Drop Esperado (oportunidad de venta)",
        "forecast": "Superficie IV Forecasted (mañana)",
    }

    fig.update_layout(
        title=dict(text=f"<b>{title_map.get(view,'')}</b>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        scene=dict(
            xaxis=dict(title="Log-moneyness (%)", gridcolor="#262A30",
                       backgroundcolor="#0A0B0D", tickfont=dict(size=9, color="#9AA1A9")),
            yaxis=dict(title="DTE (días)", gridcolor="#262A30",
                       backgroundcolor="#0A0B0D", tickfont=dict(size=9, color="#9AA1A9")),
            zaxis=dict(title="∆IV pts" if view=="drop" else "IV %",
                       gridcolor="#262A30", backgroundcolor="#0A0B0D",
                       tickfont=dict(size=9, color="#9AA1A9")),
            bgcolor="#0A0B0D",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.9))),
        paper_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def compute_skew_metrics(chains: dict, spot: float) -> dict:
    """
    Métricas de skew del primer vencimiento válido usando IV Black-Scholes.
    """
    metrics = {}
    if not chains or not spot: return metrics

    for exp_str, data in sorted(chains.items(), key=lambda x: x[1]["dte"]):
        puts  = data["puts"]
        calls = data["calls"]
        dte   = data["dte"]
        if "iv" not in puts.columns or "iv" not in calls.columns: continue
        if len(puts) < 5 or len(calls) < 5: continue

        def get_iv_at_m(df, target, tol=0.05):
            sub = df[df["moneyness"].between(target-tol, target+tol)]
            if sub.empty: return np.nan
            w = sub["openInterest"].values + 1
            return float(np.average(sub["iv"].values, weights=w))

        atm_iv   = get_iv_at_m(puts, 1.00, 0.03)
        if np.isnan(atm_iv): atm_iv = get_iv_at_m(calls, 1.00, 0.03)
        put_25d  = get_iv_at_m(puts,  0.90, 0.04)
        call_25d = get_iv_at_m(calls, 1.10, 0.04)

        rr25 = (call_25d - put_25d)*100 if not (np.isnan(put_25d) or np.isnan(call_25d)) else np.nan
        bf25 = ((call_25d+put_25d)/2 - atm_iv)*100 if not np.isnan(atm_iv) else np.nan

        sp = puts[puts["moneyness"].between(0.80, 1.00)]
        if len(sp) >= 3:
            coef = np.polyfit(sp["moneyness"].values, sp["iv"].values, 1)[0]
            skew_slope = coef * 0.10 * 100
        else:
            skew_slope = np.nan

        pc_ratio = (puts["volume"].sum() / calls["volume"].sum()
                    if calls["volume"].sum() > 0 else np.nan)

        metrics = {
            "exp": exp_str, "dte": dte,
            "atm_iv":     round(atm_iv*100, 2)  if not np.isnan(atm_iv)    else None,
            "put_25d_iv": round(put_25d*100, 2)  if not np.isnan(put_25d)   else None,
            "call_25d_iv":round(call_25d*100,2)  if not np.isnan(call_25d)  else None,
            "rr25":       round(rr25, 2)          if not np.isnan(rr25)      else None,
            "bf25":       round(bf25, 2)          if not np.isnan(bf25)      else None,
            "skew_slope": round(skew_slope, 2)    if not np.isnan(skew_slope)else None,
            "pc_ratio":   round(pc_ratio, 3)      if not np.isnan(pc_ratio)  else None,
        }
        break
    return metrics

# ─── Chart: Skew Curves ────────────────────────────────────────────────────
SKEW_PALETTE = [
    "#F4F5F6","#C9821A","#16C784","#C9821A",
    "#9AA1A9","#F5A623","#EA3943","#C7CCD1",
]

def build_skew_curves(chains: dict, spot: float,
                      moneyness_range=(0.75, 1.25),
                      y_mode: str = "moneyness",
                      r: float | None = None,
                      q: float | None = None) -> go.Figure:
    """
    Curvas IV (BS) vs Moneyness o log-moneyness por vencimiento.
    y_mode: 'moneyness' → % vs spot | 'log' → ln(K/F)
    Usa columna 'iv' (BS calculado), no yfinance.

    r, q: para el forward F = S·exp((r-q)·T) del eje log-moneyness.
    Si None, se obtienen de rates (cached).
    """
    if r is None: r = get_risk_free_rate()
    if q is None: q = get_dividend_yield("SPY")
    fig = go.Figure()
    if not chains or not spot: return fig
    lo, hi = moneyness_range

    for idx, (exp_str, data) in enumerate(sorted(chains.items(), key=lambda x: x[1]["dte"])):
        clr   = SKEW_PALETTE[idx % len(SKEW_PALETTE)]
        dte   = data["dte"]; T = dte / 365.0
        puts  = data["puts"][data["puts"]["moneyness"].between(lo, 1.02)].copy()
        calls = data["calls"][data["calls"]["moneyness"].between(0.98, hi)].copy()
        combined = pd.concat([puts, calls]).drop_duplicates("strike").sort_values("moneyness")
        if len(combined) < 3: continue

        iv_smooth = combined["iv"].rolling(3, min_periods=1, center=True).mean()

        if y_mode == "log":
            # Forward real del slice: F = S·exp((r-q)·T). Antes se usaba S directo
            # (y una F calculada con exp(0·T) que ni se usaba) → log-moneyness
            # sesgado ~(r-q)·T en strikes OTM, peor en vencimientos largos.
            F = spot * np.exp((r - q) * T)
            x_vals = np.log(combined["strike"].values / F)
            x_label = "Log-moneyness  ln(K/F)"
            x_suffix = ""
        else:
            x_vals  = combined["moneyness"].values * 100 - 100
            x_label = "% vs Spot  (neg=OTM puts | pos=OTM calls)"
            x_suffix = "%"

        fig.add_trace(go.Scatter(
            x=x_vals, y=iv_smooth * 100,
            mode="lines+markers", name=f"{exp_str} ({dte}d)",
            line=dict(color=clr, width=2.5, shape="spline"),
            marker=dict(size=5, color=clr, opacity=0.7),
            hovertemplate=f"<b>{exp_str}</b><br>x: %{{x:.2f}}{x_suffix}<br>IV(BS): %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#9AA1A9", line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=10, color="#9AA1A9"))
    fig.update_layout(
        title=dict(text="<b>Volatility Skew</b><sup>  IV Black-Scholes por vencimiento</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title=dict(text=x_label, font=dict(size=10, color="#9AA1A9")),
                   gridcolor="#1C1F24", zeroline=True, zerolinecolor="#262A30",
                   tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono")),
        yaxis=dict(title=dict(text="Implied Volatility BS (%)", font=dict(size=10, color="#9AA1A9")),
                   gridcolor="#1C1F24",
                   tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono"),
                   ticksuffix="%"),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(21,23,26,0.9)", bordercolor="#262A30", borderwidth=1,
                    font=dict(size=9, color="#F4F5F6", family="JetBrains Mono")),
        hovermode="x unified",
    )
    return fig


# ─── Chart: ATM Term Structure ─────────────────────────────────────────────
def build_atm_term_structure(chains: dict, spot: float) -> go.Figure:
    fig = go.Figure()
    if not chains or not spot: return fig
    rows = []
    for exp_str, data in sorted(chains.items(), key=lambda x: x[1]["dte"]):
        dte = data["dte"]
        atm = pd.concat([
            data["puts"][data["puts"]["moneyness"].between(0.97, 1.03)],
            data["calls"][data["calls"]["moneyness"].between(0.97, 1.03)],
        ])
        if atm.empty or "iv" not in atm.columns: continue
        atm_iv = float(np.average(atm["iv"].values,
                                  weights=atm["openInterest"].values + 1)) * 100
        rows.append({"dte":dte,"atm_iv":atm_iv,"exp":exp_str})
    if not rows: return fig
    df_atm = pd.DataFrame(rows).sort_values("dte")
    fig.add_trace(go.Scatter(
        x=df_atm["dte"], y=df_atm["atm_iv"],
        mode="lines+markers+text", name="ATM IV (BS)",
        line=dict(color="#9AA1A9", width=3, shape="spline"),
        marker=dict(size=10, color="#9AA1A9", line=dict(width=2, color="#0A0B0D")),
        text=[f"{v:.1f}%" for v in df_atm["atm_iv"]],
        textposition="top center",
        textfont=dict(size=9, color="#F4F5F6", family="JetBrains Mono"),
        hovertemplate="DTE: %{x}d<br>ATM IV: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>ATM IV Term Structure</b><sup>  IV en el dinero por vencimiento</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title=dict(text="Días al Vencimiento (DTE)", font=dict(size=10, color="#9AA1A9")),
                   gridcolor="#1C1F24",
                   tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono")),
        yaxis=dict(title=dict(text="ATM IV (%)", font=dict(size=10, color="#9AA1A9")),
                   gridcolor="#1C1F24",
                   tickfont=dict(size=10, color="#9AA1A9", family="JetBrains Mono"),
                   ticksuffix="%"),
        hovermode="x unified", showlegend=False,
    )
    return fig


# ─── Chart: IV Surface 3D (griddata — método del otro proyecto) ────────────
def build_iv_surface(chains: dict, spot: float,
                     moneyness_range=(0.80, 1.20), n_grid=40,
                     y_mode="moneyness") -> go.Figure:
    """
    Superficie 3D con scipy.griddata (lineal + nearest fallback).
    X = DTE, Y = moneyness o log-moneyness, Z = IV% BS.
    Mismo método que Volatility Surface de Drosogiannis.
    """
    fig = go.Figure()
    if not chains or not spot: return fig
    lo, hi = moneyness_range

    all_X, all_Y, all_Z = [], [], []
    for exp_str, data in chains.items():
        dte = data["dte"]
        pts = pd.concat([
            data["puts"][data["puts"]["moneyness"].between(lo, 1.02)],
            data["calls"][data["calls"]["moneyness"].between(0.98, hi)],
        ]).drop_duplicates("strike")
        if len(pts) < 4 or "iv" not in pts.columns: continue
        if y_mode == "log":
            y_vals = np.log(pts["strike"].values / spot)
        else:
            y_vals = pts["moneyness"].values * 100 - 100   # % vs spot
        all_X.extend([dte] * len(pts))
        all_Y.extend(y_vals.tolist())
        all_Z.extend((pts["iv"].values * 100).tolist())

    if len(all_X) < 8: return fig
    X = np.array(all_X); Y = np.array(all_Y); Z = np.array(all_Z)

    # Grid regular
    xi = np.linspace(X.min(), X.max(), n_grid)
    yi = np.linspace(Y.min(), Y.max(), n_grid)
    xi_g, yi_g = np.meshgrid(xi, yi)

    zi = griddata((X, Y), Z, (xi_g, yi_g), method="linear")
    zi2 = griddata((X, Y), Z, (xi_g, yi_g), method="nearest")
    zi  = np.where(np.isnan(zi), zi2, zi)   # fill NaN con nearest

    y_label = "Log-moneyness ln(K/S)" if y_mode == "log" else "% vs Spot"

    fig.add_trace(go.Surface(
        x=xi, y=yi, z=zi,
        colorscale=[
            [0.0,"#3A3F47"],[0.15,"#5E656E"],[0.30,"#C7CCD1"],
            [0.45,"#9AA1A9"],[0.55,"#16C784"],[0.65,"#F5A623"],
            [0.80,"#C9821A"],[1.0,"#EA3943"],
        ],
        colorbar=dict(title=dict(text="IV %", font=dict(color="#9AA1A9",size=10)),
                      tickfont=dict(color="#9AA1A9",size=9), len=0.6, thickness=12),
        hovertemplate="DTE: %{x:.0f}d<br>Y: %{y:.2f}<br>IV: %{z:.1f}%<extra></extra>",
        opacity=0.92,
    ))
    fig.update_layout(
        title=dict(text="<b>Implied Volatility Surface</b><sup>  IV Black-Scholes · griddata interpolation</sup>",
                   font=dict(size=14, color="#F4F5F6", family="Inter"), x=0.5),
        scene=dict(
            xaxis=dict(title="DTE (días)", gridcolor="#262A30", backgroundcolor="#0A0B0D",
                       tickfont=dict(size=9, color="#9AA1A9")),
            yaxis=dict(title=y_label, gridcolor="#262A30", backgroundcolor="#0A0B0D",
                       tickfont=dict(size=9, color="#9AA1A9")),
            zaxis=dict(title="IV (%)", gridcolor="#262A30", backgroundcolor="#0A0B0D",
                       tickfont=dict(size=9, color="#9AA1A9")),
            bgcolor="#0A0B0D",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9), up=dict(x=0,y=0,z=1)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=520, margin=dict(l=0,r=0,t=50,b=0),
    )
    return fig


# ─── Chart: IV Heatmap 2D ──────────────────────────────────────────────────
def build_iv_heatmap(chains: dict, spot: float,
                     moneyness_range=(0.82, 1.18), n_bins=35) -> go.Figure:
    fig = go.Figure()
    if not chains or not spot: return fig
    lo, hi = moneyness_range
    mon_grid = np.linspace(lo, hi, n_bins)
    dte_vals, iv_rows = [], []

    for exp_str, data in sorted(chains.items(), key=lambda x: x[1]["dte"]):
        dte = data["dte"]
        pts = pd.concat([
            data["puts"][data["puts"]["moneyness"].between(lo, 1.02)],
            data["calls"][data["calls"]["moneyness"].between(0.98, hi)],
        ]).drop_duplicates("moneyness").sort_values("moneyness")
        if len(pts) < 3 or "iv" not in pts.columns: continue
        iv_interp = np.interp(mon_grid, pts["moneyness"].values,
                              pts["iv"].values * 100, left=np.nan, right=np.nan)
        dte_vals.append(dte); iv_rows.append(iv_interp)

    if not iv_rows: return fig
    Z = np.array(iv_rows)
    labels_x = [f"{(m*100-100):+.0f}%" for m in mon_grid]
    labels_y = [f"{d}d" for d in dte_vals]

    atm_idx = int(np.argmin(np.abs(mon_grid - 1.0)))
    fig.add_trace(go.Heatmap(
        z=Z, x=labels_x, y=labels_y,
        colorscale=[[0.0,"#5E656E"],[0.25,"#C7CCD1"],[0.50,"#16C784"],
                    [0.70,"#F5A623"],[0.85,"#C9821A"],[1.0,"#EA3943"]],
        colorbar=dict(title=dict(text="IV %",font=dict(color="#9AA1A9",size=10)),
                      tickfont=dict(color="#9AA1A9",size=9), len=0.8, thickness=14),
        hoverongaps=False,
        hovertemplate="Δ Spot: %{x}<br>DTE: %{y}<br>IV(BS): %{z:.1f}%<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.add_vline(x=labels_x[atm_idx], line_dash="dash", line_color="#9AA1A9",
                  line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=9, color="#9AA1A9"))
    fig.update_layout(
        title=dict(text="<b>IV Surface — Heatmap</b><sup>  Filas=DTE · Columnas=%Spot · Color=IV(BS)%</sup>",
                   font=dict(size=13, color="#F4F5F6", family="Inter"), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=55, r=20, t=60, b=60),
        xaxis=dict(tickfont=dict(size=8,color="#9AA1A9",family="JetBrains Mono"),
                   title=dict(text="Distancia al Spot",font=dict(size=10,color="#9AA1A9")),
                   tickangle=-45),
        yaxis=dict(tickfont=dict(size=9,color="#9AA1A9",family="JetBrains Mono"),
                   title=dict(text="DTE",font=dict(size=10,color="#9AA1A9")),
                   autorange="reversed"),
    )
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE EXTENSION — SPY + VIX desde yfinance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["yahoo_spot"])
def fetch_live_spy_vix() -> pd.DataFrame:
    """
    Extiende el parquet con datos SPY y VIX de los últimos 45 días.
    Soluciona el gap entre la última fecha del parquet y hoy.
    TTL=55s → misma cadencia que los precios de futuros.
    """
    log = logging.getLogger("vix_controller")
    frames = {}
    for col, sym in [("SPY_Close", "SPY"), ("VIX_Close", "^VIX"),
                     ("VVIX_Live", "^VVIX")]:
        h = _yf_history_safe(sym, period="45d")
        if h is None or h.empty or "Close" not in h.columns:
            continue
        try:
            s = h["Close"].copy()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = pd.DatetimeIndex(s.index).normalize()
            frames[col] = s
        except Exception as ex:
            log.warning(f"fetch_live_spy_vix {sym} normalize: {ex}")
    if "SPY_Close" in frames and "VIX_Close" in frames:
        df = pd.DataFrame(frames)
        log.info(f"Live extension: {len(df)} rows, last={df.index[-1].date()}")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["har_rv"])
def _compute_har_rv_asymmetric(spy_close: pd.Series, vix_close: pd.Series,
                                window: int = 252, h: int = 22) -> dict:
    """
    ════════════════════════════════════════════════════════════════
    MODELO: HAR-RV-A — Heterogeneous Autoregressive Realized
            Volatility with Asymmetry (Good/Bad Volatility)
    ════════════════════════════════════════════════════════════════

    REFERENCIA: Patton & Sheppard (2015) "Good Volatility, Bad Volatility:
    Signed Jumps and the Persistence of Volatility"
    Review of Economics and Statistics, 97(3), 683-697.

    POR QUÉ ES EL MEJOR MODELO PARA ESTE PROPÓSITO:
    ─────────────────────────────────────────────────
    1. EFECTO LEVERAGE: Las caídas del mercado (ret < 0) generan más vol
       futura que subidas de igual magnitud. HAR ignora esto; HAR-A lo captura
       separando "buena vol" (días up) de "mala vol" (días down).

    2. PARSIMONIA: 5 parámetros estimables con OLS simple. GARCH requiere MLE.
       EGARCH más complejo aún. HAR-A es comparablemente preciso con menos costo.

    3. MEMORIA LARGA: Las tres frecuencias (daily/weekly/monthly) capturan la
       persistencia fraccional de la vol sin ARFIMA.

    4. EVIDENCIA EMPÍRICA: Supera HAR, GARCH, EGARCH en la mayoría de mercados
       de renta variable en QLIKE y RMSE OOS (Patton & Sheppard 2015).

    5. HORIZONTE CORRECTO: Predice E[RV_{t,t+22d}] que matchea el VIX
       (~30 días calendario). RV20 trailing NO hace esto.

    ESPECIFICACIÓN:
    ─────────────────────────────────────────────────
    RV_{t+h} = β₀ + β₁·GV_d + β₂·BV_d + β₃·RV_w + β₄·RV_m + ε

      GV_d = |rₜ|·√252·100  si rₜ > 0, else 0  (good vol — días alcistas)
      BV_d = |rₜ|·√252·100  si rₜ ≤ 0, else 0  (bad vol  — días bajistas)
      RV_w = (GV_d+BV_d) promedio últimos 5d
      RV_m = (GV_d+BV_d) promedio últimos 22d
      h    = 22 días hábiles ≈ 30 días calendario

    Empíricamente: β₂ > β₁ → bad vol más persistente = leverage effect.
    """
    log = logging.getLogger("vix_controller")
    log_ret = np.log(spy_close / spy_close.shift(1))

    rv_tot = log_ret.abs() * np.sqrt(252) * 100
    gv_d   = pd.Series(np.where(log_ret > 0,  rv_tot, 0.0), index=spy_close.index)
    bv_d   = pd.Series(np.where(log_ret <= 0, rv_tot, 0.0), index=spy_close.index)
    rv_w   = rv_tot.rolling(5,  min_periods=3).mean()
    rv_m   = rv_tot.rolling(22, min_periods=10).mean()
    rv_fwd = log_ret.rolling(h, min_periods=int(h*0.8)).std().shift(-h) * np.sqrt(252) * 100

    n    = len(spy_close)
    gv_a = gv_d.values; bv_a = bv_d.values
    rw_a = rv_w.values; rm_a = rv_m.values
    rvf_a= rv_fwd.values; rt_a = rv_tot.values

    forecasts  = np.full(n, np.nan)
    betas_hist = []

    # FIX look-ahead: rv_fwd[j] = std(log_ret[j:j+h]) solo es conocido en t=j+h.
    # Para entrenar al tiempo i, solo podemos usar pares (X_t, y_t) con t+h ≤ i.
    # → training slice es [t0 : i-h], no [t0 : i]. Antes, las últimas h filas
    # del fit usaban labels construidas con datos posteriores a i → bias de hindsight.
    for i in range(window + h, n):
        t0 = i - window
        train_end = i - h
        if train_end - t0 < 60:
            continue
        X  = np.column_stack([np.ones(train_end - t0),
                               gv_a[t0:train_end], bv_a[t0:train_end],
                               rw_a[t0:train_end], rm_a[t0:train_end]])
        y  = rvf_a[t0:train_end]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if ok.sum() < 60:
            continue
        beta, _, _, _ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        xi = np.array([1.0, gv_a[i], bv_a[i], rw_a[i], rm_a[i]])
        if np.isfinite(xi).all():
            forecasts[i] = max(float(np.dot(beta, xi)), 1.0)
        betas_hist.append(beta.copy())

    # ── Out-of-sample backtest — últimos 504 días como test ────────
    test_start = max(window + 22, n - 504)
    # Para el backtest necesitamos rv_fwd (que requiere h días futuros)
    # Solo evaluamos donde hay tanto forecast como rv_fwd disponible
    fc_s  = forecasts[test_start:n-h]
    rv_s  = rvf_a[test_start:n-h]
    ok_bt = np.isfinite(fc_s) & np.isfinite(rv_s)
    fc_bt = fc_s[ok_bt]; rv_bt = rv_s[ok_bt]
    idx_bt= spy_close.index[test_start:n-h][ok_bt]

    backtest = {}
    if len(fc_bt) >= 30:
        ss_res = np.sum((rv_bt - fc_bt)**2)
        ss_tot = np.sum((rv_bt - rv_bt.mean())**2)
        r2_oos = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        rmse   = float(np.sqrt(np.mean((rv_bt - fc_bt)**2)))
        mae    = float(np.mean(np.abs(rv_bt - fc_bt)))
        # QLIKE: log(f²) + rv²/f² — penaliza subestimación asimétricamente
        eps    = 1e-6
        qlike  = float(np.mean(np.log(np.maximum(fc_bt,eps)**2) + rv_bt**2/np.maximum(fc_bt,eps)**2))
        # Dirección (¿sube o baja?)
        dir_acc= float(np.mean(np.sign(np.diff(rv_bt)) == np.sign(np.diff(fc_bt)))*100) if len(fc_bt) > 1 else np.nan
        # Mincer-Zarnowitz: RV_actual = a + b*forecast + ε  (ideal: a≈0, b≈1)
        Xmz = np.column_stack([np.ones(len(fc_bt)), fc_bt])
        mz_b, _, _, _ = np.linalg.lstsq(Xmz, rv_bt, rcond=None)

        # Benchmarks
        ewma_arr = np.full(n, np.nan); ewma = 0.0
        for j in range(n):
            ewma = 0.94*ewma + 0.06*rt_a[j]**2 if np.isfinite(rt_a[j]) else ewma
            ewma_arr[j] = np.sqrt(ewma*252) if ewma > 0 else np.nan
        ew_bt = ewma_arr[test_start:n-h][ok_bt]
        rm_bt = rm_a[test_start:n-h][ok_bt]

        def _rmse_b(f, a): return float(np.sqrt(np.nanmean((np.where(np.isfinite(f),a-f,np.nan))**2)))
        def _r2_b(f, a):
            mask=np.isfinite(f); ss=np.nansum((a[mask]-f[mask])**2)
            tot=np.nansum((a[mask]-a[mask].mean())**2)
            return float(1-ss/tot) if tot>0 else np.nan

        backtest = {
            'r2_oos':     round(float(r2_oos), 4),
            'rmse':       round(rmse, 3),
            'mae':        round(mae, 3),
            'qlike':      round(qlike, 4),
            'dir_acc':    round(dir_acc, 1),
            'mz_alpha':   round(float(mz_b[0]), 3),
            'mz_beta':    round(float(mz_b[1]), 3),
            'n_test':     int(len(fc_bt)),
            'rmse_naive': round(_rmse_b(rm_bt, rv_bt), 3),
            'rmse_ewma':  round(_rmse_b(ew_bt, rv_bt), 3),
            'r2_naive':   round(_r2_b(rm_bt, rv_bt), 4),
            'r2_ewma':    round(_r2_b(ew_bt, rv_bt), 4),
            'fc_test':    fc_bt.tolist(),
            'rv_test':    rv_bt.tolist(),
            'idx_test':   [str(d.date()) for d in idx_bt],
        }
        log.info(f"HAR-A BT: R²={r2_oos:.3f} RMSE={rmse:.2f} n={len(fc_bt)}")

    last_beta = betas_hist[-1] if betas_hist else None
    beta_dict = {}
    if last_beta is not None:
        beta_dict = {
            'β₀ intercepto': round(float(last_beta[0]), 3),
            'β₁ GoodVol':    round(float(last_beta[1]), 3),
            'β₂ BadVol':     round(float(last_beta[2]), 3),
            'β₃ RV_weekly':  round(float(last_beta[3]), 3),
            'β₄ RV_monthly': round(float(last_beta[4]), 3),
        }

    idx = spy_close.index
    df_out = pd.DataFrame({
        'rv_tot':         rt_a,
        'gv_d':           gv_d.values,
        'bv_d':           bv_d.values,
        'rv_w':           rw_a,
        'rv_m':           rm_a,
        'rv_fwd':         rvf_a,
        'har_a_forecast': forecasts,
        'vix':            vix_close.values,
        'vrp_har_a':      vix_close.values - forecasts,
    }, index=idx)

    return {'df': df_out, 'beta': beta_dict, 'backtest': backtest}


def compute_edge_analytics(df, edge_extra):
    log = logging.getLogger("vix_controller")
    out = {}

    # ── Normalizar índice del parquet ────────────────────────────────
    bt = df[df['VIX_Close'].notna() & df['SPY_Close'].notna()].copy()
    if bt.index.tz is not None:
        bt.index = bt.index.tz_localize(None)
    bt.index = pd.DatetimeIndex(bt.index).normalize()

    # ── EXTENSIÓN LIVE: rellenar gap parquet → hoy ───────────────────
    try:
        live_ext = fetch_live_spy_vix()
        if not live_ext.empty:
            cutoff = bt.index[-1]
            new_rows = live_ext[live_ext.index > cutoff].copy()
            if not new_rows.empty:
                # Solo llevar las columnas SPY_Close, VIX_Close (y VVIX_Live si existe)
                for col in ['SPY_Close', 'VIX_Close']:
                    if col in new_rows.columns:
                        pass  # se añaden via concat
                bt = pd.concat([bt, new_rows[new_rows.columns.intersection(bt.columns.tolist() + ['SPY_Close','VIX_Close','VVIX_Live'])]])
                bt = bt[~bt.index.duplicated(keep='last')].sort_index()
                # Si hay VVIX_Live en la extensión, mantenerlo en bt
                if 'VVIX_Live' in new_rows.columns and 'VVIX_Live' not in bt.columns:
                    bt['VVIX_Live'] = np.nan
                    bt.update(new_rows[['VVIX_Live']])
                log.info(f"Live extension: +{len(new_rows)} rows → bt ends {bt.index[-1].date()}")
    except Exception as ex:
        log.warning(f"Live extension failed: {ex}")

    if len(bt) < 60:
        return out

    log_ret = np.log(bt['SPY_Close'] / bt['SPY_Close'].shift(1))
    bt['RV5']  = log_ret.rolling(5).std()  * np.sqrt(252) * 100
    bt['RV10'] = log_ret.rolling(10).std() * np.sqrt(252) * 100
    bt['RV20'] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    bt['RV60'] = log_ret.rolling(60).std() * np.sqrt(252) * 100
    bt['VRP']  = bt['VIX_Close'] - bt['RV20']

    # ── HAR-RV-A: modelo asimétrico (Patton & Sheppard 2015) ─────────
    try:
        har_result = _compute_har_rv_asymmetric(bt['SPY_Close'], bt['VIX_Close'])
        har_df         = har_result['df']
        bt['HAR_Forecast'] = har_df['har_a_forecast'].values
        bt['VRP_HAR']      = har_df['vrp_har_a'].values
        bt['RV_Fwd_22']    = har_df['rv_fwd'].values
        bt['GV_d']         = har_df['gv_d'].values   # good vol
        bt['BV_d']         = har_df['bv_d'].values   # bad vol
        out['har_beta']    = har_result['beta']
        out['har_backtest']= har_result['backtest']
        log.info("HAR-A model OK")
    except Exception as e:
        log.warning(f"HAR-A error: {e}")
        bt['HAR_Forecast'] = np.nan
        bt['VRP_HAR']      = bt['VRP']

    # VRP percentile (usa HAR si disponible)
    vrp_col = 'VRP_HAR' if 'VRP_HAR' in bt.columns and bt['VRP_HAR'].notna().sum() > 20 else 'VRP'
    vrp_2y = bt[vrp_col].tail(504).dropna()
    if len(vrp_2y) > 20:
        out['vrp_percentile'] = round((vrp_2y < vrp_2y.iloc[-1]).mean() * 100, 0)

    if 'M1_Price' in bt.columns and 'M1_DTE' in bt.columns:
        m1 = bt['M1_Price']; dte = bt['M1_DTE']; spot = bt['VIX_Close']
        # dte >= 1: con DTE fraccional (día de expiración) la anualización
        # 365/dte explota (p.ej. 730× con dte=0.5) y mete spikes absurdos.
        valid = (m1 > 0) & (dte >= 1) & m1.notna() & dte.notna() & spot.notna()
        bt['Roll_Yield'] = np.where(valid, (m1 - spot) / m1 * (365 / dte) * 100, np.nan)

    # ── VVIX live desde yfinance ────────────────────────────────────
    # Nota: bt puede ya tener VVIX_Live de fetch_live_spy_vix → no re-join
    if 'VVIX' in edge_extra and not edge_extra['VVIX'].empty:
        vvix_s = edge_extra['VVIX'][['Close']].rename(columns={'Close': 'VVIX_Live'})
        if 'VVIX_Live' not in bt.columns:
            bt = bt.join(vvix_s, how='left')
        else:
            # Actualizar NaN del parquet con datos del edge_extra (más completos en historia)
            bt['VVIX_Live'] = bt['VVIX_Live'].fillna(vvix_s['VVIX_Live'])
        bt['VVIX_VIX'] = np.where(
            (bt['VIX_Close'] > 0) & bt['VVIX_Live'].notna(),
            bt['VVIX_Live'] / bt['VIX_Close'], np.nan
        )
    elif 'VVIX_Close' in bt.columns:
        bt['VVIX_VIX'] = np.where(bt['VIX_Close'] > 0, bt['VVIX_Close'] / bt['VIX_Close'], np.nan)

    # ── SKEW live ───────────────────────────────────────────────────
    if 'SKEW' in edge_extra and not edge_extra['SKEW'].empty:
        skew_s = edge_extra['SKEW'][['Close']].rename(columns={'Close': 'SKEW'})
        if 'SKEW' not in bt.columns:
            bt = bt.join(skew_s, how='left')
        else:
            bt['SKEW'] = bt['SKEW'].fillna(skew_s['SKEW'])
        log.info(f"SKEW: {bt['SKEW'].notna().sum()} valid rows")

    # ── Credit Spread live (HYG vs IEF) ─────────────────────────────
    if 'HYG' in edge_extra and 'IEF' in edge_extra:
        hyg = edge_extra['HYG'][['Close']].rename(columns={'Close': 'HYG'})
        ief = edge_extra['IEF'][['Close']].rename(columns={'Close': 'IEF'})
        if 'HYG' not in bt.columns:
            bt = bt.join(hyg, how='left')
        if 'IEF' not in bt.columns:
            bt = bt.join(ief, how='left')
        if 'HYG' in bt.columns and 'IEF' in bt.columns:
            bt['Credit_Spread'] = -(
                bt['HYG'].pct_change().rolling(20).sum() -
                bt['IEF'].pct_change().rolling(20).sum()
            ) * 100

    # Calendario de eventos 2026
    today = pd.Timestamp(now_cdmx().date())
    upcoming = []
    events = {
        'FOMC': ['2026-01-28','2026-03-18','2026-05-06','2026-06-17',
                  '2026-07-29','2026-09-16','2026-10-28','2026-12-16'],
        'CPI':  ['2026-01-14','2026-02-12','2026-03-11','2026-04-14','2026-05-13',
                  '2026-06-10','2026-07-15','2026-08-12','2026-09-10','2026-10-13',
                  '2026-11-12','2026-12-10'],
        'NFP':  ['2026-01-09','2026-02-06','2026-03-06','2026-04-03','2026-05-08',
                  '2026-06-05','2026-07-02','2026-08-07','2026-09-04','2026-10-02',
                  '2026-11-06','2026-12-04'],
    }
    for ev_name, dates in events.items():
        for d in dates:
            dt = pd.Timestamp(d)
            diff = (dt - today).days
            if 0 <= diff <= 14:
                upcoming.append((ev_name, dt, diff))
    upcoming.sort(key=lambda x: x[2])
    out['upcoming_events'] = upcoming
    out['bt'] = bt
    return out


def build_vrp_chart(bt, window=252):
    """
    VRP con modelo HAR-RV.
    Muestra VIX vs HAR_Forecast (E[vol futura]) y VRP_HAR como area.
    También muestra la vol realizada ex-post (RV_Fwd_22) para referencia visual.
    """
    from plotly.subplots import make_subplots

    use_har = 'HAR_Forecast' in bt.columns and bt['HAR_Forecast'].notna().sum() > 20

    if use_har:
        col_vrp = 'VRP_HAR'
        p = bt.tail(window).copy()
        p = p[p['VIX_Close'].notna()]
    else:
        col_vrp = 'VRP'
        p = bt.tail(window).dropna(subset=['VRP'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.03)

    # Panel 1: VIX, HAR forecast, RV realizada ex-post
    fig.add_trace(go.Scatter(
        x=p.index, y=p['VIX_Close'], name='VIX (IV implícita)',
        line=dict(color='#EA3943', width=2.5),
        hovertemplate='VIX: %{y:.1f}<extra></extra>'), row=1, col=1)

    if use_har:
        fig.add_trace(go.Scatter(
            x=p.index, y=p['HAR_Forecast'], name='HAR-RV Forecast (E[vol])',
            line=dict(color='#F4F5F6', width=2, dash='dash'),
            hovertemplate='HAR Forecast: %{y:.1f}<extra></extra>'), row=1, col=1)
        if 'RV_Fwd_22' in p.columns:
            fig.add_trace(go.Scatter(
                x=p.index, y=p['RV_Fwd_22'], name='RV Realizada 22d (ex-post)',
                line=dict(color='#9AA1A9', width=1.5, dash='dot'),
                hovertemplate='RV realizada: %{y:.1f}<extra></extra>'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=p.index, y=p['RV20'], name='RV20 (trailing)',
            line=dict(color='#F4F5F6', width=2),
            hovertemplate='RV20: %{y:.1f}<extra></extra>'), row=1, col=1)

    # Panel 2: VRP como área
    if col_vrp in p.columns and p[col_vrp].notna().sum() > 5:
        vrp_vals = p[col_vrp].fillna(0)
        colors_vrp = ['#16C784' if v >= 0 else '#EA3943' for v in vrp_vals]
        fig.add_trace(go.Bar(
            x=p.index, y=vrp_vals,
            name='VRP = VIX − E[RV]' if use_har else 'VRP = VIX − RV20',
            marker_color=colors_vrp, opacity=0.7,
            hovertemplate='VRP: %{y:+.1f} pts<extra></extra>'), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='#3A3F47',
                      line_width=1.5, row=2, col=1)

        # Líneas de percentil P25/P75 en VRP
        vrp_clean = vrp_vals[vrp_vals.notna()]
        if len(vrp_clean) > 20:
            p25 = float(vrp_clean.quantile(0.25))
            p75 = float(vrp_clean.quantile(0.75))
            fig.add_hline(y=p25, line_dash='dot', line_color='#F5A623',
                          line_width=1, row=2, col=1,
                          annotation_text=f' P25: {p25:.1f}',
                          annotation_font=dict(size=8, color='#F5A623'))
            fig.add_hline(y=p75, line_dash='dot', line_color='#16C784',
                          line_width=1, row=2, col=1,
                          annotation_text=f' P75: {p75:.1f}',
                          annotation_font=dict(size=8, color='#16C784'))

    subtitle = ('VIX vs HAR-RV Forecast · VRP = prima pagada sobre vol esperada'
                if use_har else 'VIX - RV20 trailing (definición simplificada)')
    fig.update_layout(
        title=dict(
            text=f'<b>Volatility Risk Premium</b><sup>  {subtitle}</sup>',
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=480, margin=dict(l=55, r=30, t=60, b=40),
        xaxis2=dict(gridcolor='#1C1F24',
                    tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        xaxis=dict(gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Vol %', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis2=dict(title='VRP (pts)', gridcolor='#1C1F24',
                    tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
                    zeroline=True, zerolinecolor='#262A30'),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#F4F5F6', family='JetBrains Mono')),
        hovermode='x unified', bargap=0)
    return fig


def build_har_backtest_charts(backtest: dict) -> tuple:
    """
    Retorna dos figuras:
    1. Time-series: HAR-A forecast vs RV realizada (test set)
    2. Scatter Mincer-Zarnowitz: predicho vs actual
    """
    if not backtest or 'fc_test' not in backtest:
        return go.Figure(), go.Figure()

    fc  = np.array(backtest['fc_test'])
    rv  = np.array(backtest['rv_test'])
    idx = pd.to_datetime(backtest['idx_test'])

    # ── Fig 1: Time-series ─────────────────────────────────────────
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=idx, y=rv, name='RV Realizada (ex-post, 22d)',
        line=dict(color='#9AA1A9', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>RV real: %{y:.1f}%<extra></extra>'))
    fig_ts.add_trace(go.Scatter(
        x=idx, y=fc, name='HAR-A Forecast',
        line=dict(color='#C9821A', width=2, dash='dash'),
        hovertemplate='HAR-A: %{y:.1f}%<extra></extra>'))
    # Error band
    err = fc - rv
    fig_ts.add_trace(go.Scatter(
        x=list(idx)+list(idx[::-1]),
        y=list(np.maximum(fc,rv))+list(np.minimum(fc,rv)[::-1]),
        fill='toself', fillcolor='rgba(240,136,62,0.08)',
        line=dict(width=0), name='Error band', hoverinfo='skip'))
    r2   = backtest.get('r2_oos', np.nan)
    rmse = backtest.get('rmse', np.nan)
    fig_ts.update_layout(
        title=dict(
            text=f'<b>HAR-A Backtest — Forecast vs RV Realizada</b>'
                 f'<sup>  OOS R²={r2:.3f} · RMSE={rmse:.2f} · n={backtest["n_test"]}d</sup>',
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=55, r=30, t=60, b=40),
        xaxis=dict(gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Vol % (anualizada)', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#F4F5F6', family='JetBrains Mono')),
        hovermode='x unified')

    # ── Fig 2: Mincer-Zarnowitz scatter ───────────────────────────
    # Ideal: todos los puntos sobre la línea de 45°
    vmin = float(min(fc.min(), rv.min())) * 0.9
    vmax = float(max(fc.max(), rv.max())) * 1.05
    alpha = backtest.get('mz_alpha', 0)
    beta  = backtest.get('mz_beta', 1)

    fig_mz = go.Figure()
    fig_mz.add_trace(go.Scatter(
        x=fc, y=rv, mode='markers',
        marker=dict(color='#F4F5F6', size=4, opacity=0.5),
        name='Observaciones',
        hovertemplate='Forecast: %{x:.1f}%<br>Real: %{y:.1f}%<extra></extra>'))
    # Línea ideal 45°
    fig_mz.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax], mode='lines',
        name='Ideal (a=0, b=1)',
        line=dict(color='#16C784', width=2, dash='dot')))
    # Línea MZ regresión
    x_line = np.linspace(vmin, vmax, 100)
    y_line = alpha + beta * x_line
    fig_mz.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        name=f'MZ fit: a={alpha:.2f}, b={beta:.2f}',
        line=dict(color='#C9821A', width=2)))
    fig_mz.update_layout(
        title=dict(
            text=f'<b>Mincer-Zarnowitz</b>'
                 f'<sup>  a={alpha:.2f} (↓0) · b={beta:.2f} (↑1) · sin sesgo si a≈0, b≈1</sup>',
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=55, r=30, t=60, b=55),
        xaxis=dict(title='HAR-A Forecast (%)', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9')),
        yaxis=dict(title='RV Realizada (%)', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#F4F5F6', family='JetBrains Mono')))

    return fig_ts, fig_mz


def build_rv_chart(bt, window=252):
    p = bt.tail(window).dropna(subset=['RV20'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['VIX_Close'], name='VIX',
        line=dict(color='#EA3943', width=2.5)))
    for col, lbl, clr in [('RV5','RV5 (1w)','#F5A623'), ('RV10','RV10 (2w)','#C9821A'),
                           ('RV20','RV20 (1m)','#F4F5F6'), ('RV60','RV60 (3m)','#C9821A')]:
        if col in p.columns:
            fig.add_trace(go.Scatter(x=p.index, y=p[col], name=lbl, line=dict(color=clr, width=1.2)))
    fig.update_layout(
        title=dict(text='<b>Implied vs Realized Vol</b><sup>  VIX encima = VRP positivo</sup>',
                   font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Vol %', gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_roll_yield_chart(bt, window=252):
    if 'Roll_Yield' not in bt.columns:
        return go.Figure()
    p = bt.tail(window).dropna(subset=['Roll_Yield'])
    colors = ['#16C784' if v > 0 else '#EA3943' for v in p['Roll_Yield']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p.index, y=p['Roll_Yield'], marker_color=colors,
        name='Roll Yield %', opacity=0.7))
    fig.add_trace(go.Scatter(x=p.index, y=p['Roll_Yield'].rolling(20).mean(),
        name='SMA(20)', line=dict(color='#9AA1A9', width=2)))
    fig.add_hline(y=0, line_dash='dash', line_color='#9AA1A9', line_width=1)
    fig.update_layout(
        title=dict(text='<b>Roll Yield</b><sup>  Carry anualizado · Verde=cobras</sup>',
                   font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Ann. %', gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_vvix_ratio_chart(bt, window=252):
    """VVIX/VIX ratio — usa VVIX_Live (yfinance) cuando está disponible."""
    # Detectar fuente: preferir live, caer a parquet
    vvix_col = None
    if 'VVIX_Live' in bt.columns and bt['VVIX_Live'].notna().sum() > 10:
        vvix_col = 'VVIX_Live'
        src_label = "yfinance live"
    elif 'VVIX_VIX' in bt.columns and bt['VVIX_VIX'].notna().sum() > 10:
        vvix_col = 'VVIX_VIX'   # ya es el ratio calculado
        src_label = "parquet (ratio)"
    else:
        return go.Figure()

    fig = go.Figure()
    if vvix_col == 'VVIX_VIX':
        p = bt.tail(window).dropna(subset=['VVIX_VIX'])
        y_vals = p['VVIX_VIX']
    else:
        # Calcular ratio con VVIX_Live directamente
        sub = bt[['VIX_Close', 'VVIX_Live']].tail(window).dropna()
        if sub.empty or (sub['VIX_Close'] == 0).all():
            return go.Figure()
        y_vals = sub['VVIX_Live'] / sub['VIX_Close'].replace(0, np.nan)
        p = sub  # índice para x-axis
        src_label = "^VVIX / ^VIX (yfinance)"

    fig.add_trace(go.Scatter(
        x=p.index, y=y_vals, name='VVIX/VIX',
        line=dict(color='#C9821A', width=2),
        fill='tozeroy', fillcolor='rgba(188,140,255,0.07)',
        hovertemplate='%{x|%Y-%m-%d}<br>VVIX/VIX: %{y:.2f}<extra></extra>'))

    # Bands contextuales
    fig.add_hrect(y0=6, y1=max(float(y_vals.max(skipna=True)) + 1, 8),
                  fillcolor='rgba(234,57,67,0.07)', line_width=0)
    fig.add_hline(y=6, line_dash='dash', line_color='#EA3943', line_width=1.5,
        annotation_text='  Danger > 6', annotation_font=dict(color='#EA3943', size=10))
    fig.add_hline(y=5, line_dash='dot', line_color='#F5A623', line_width=1,
        annotation_text='  Warning > 5', annotation_font=dict(color='#F5A623', size=9))
    fig.add_hline(y=4, line_dash='dot', line_color='#16C784', line_width=0.8,
        annotation_text='  Calm < 4', annotation_font=dict(color='#16C784', size=8))

    # SMA 20d
    sma = pd.Series(y_vals.values, index=p.index).rolling(20, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=p.index, y=sma, name='SMA(20)',
        line=dict(color='#9AA1A9', width=1.2, dash='dot'), showlegend=True,
        hovertemplate='SMA20: %{y:.2f}<extra></extra>'))

    fig.update_layout(
        title=dict(
            text=f'<b>VVIX / VIX Ratio</b><sup>  Fuente: {src_label} · > 6 = dealers anticipan spike</sup>',
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Ratio', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_skew_chart(bt, window=252):
    """CBOE SKEW — usa datos live de yfinance (^SKEW) via join en bt."""
    if 'SKEW' not in bt.columns:
        return go.Figure()
    p = bt[['SKEW', 'VIX_Close']].tail(window).dropna(subset=['SKEW'])
    if len(p) < 10:
        return go.Figure()
    mean_skew = float(p['SKEW'].mean())
    std_skew  = float(p['SKEW'].std())
    fig = go.Figure()
    # Banda ±1σ
    fig.add_hrect(
        y0=mean_skew - std_skew, y1=mean_skew + std_skew,
        fillcolor='rgba(244,245,246,0.06)', line_width=0)
    fig.add_trace(go.Scatter(
        x=p.index, y=p['SKEW'], name='CBOE SKEW (^SKEW · yfinance)',
        line=dict(color='#C9821A', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>SKEW: %{y:.0f}<extra></extra>'))
    # SMA 20
    skew_sma = p['SKEW'].rolling(20, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=p.index, y=skew_sma, name='SMA(20)',
        line=dict(color='#9AA1A9', width=1.2, dash='dot'),
        hovertemplate='SMA: %{y:.0f}<extra></extra>'))
    fig.add_hline(y=mean_skew, line_dash='dot', line_color='#F4F5F6', line_width=1,
        annotation_text=f'  μ={mean_skew:.0f}',
        annotation_font=dict(color='#F4F5F6', size=9))
    fig.add_hline(y=150, line_dash='dash', line_color='#EA3943', line_width=1.5,
        annotation_text='  Extremo > 150 (tail-risk hedging)',
        annotation_font=dict(color='#EA3943', size=9))
    fig.add_hline(y=130, line_dash='dot', line_color='#F5A623', line_width=1,
        annotation_text='  Elevado > 130',
        annotation_font=dict(color='#F5A623', size=8))
    fig.update_layout(
        title=dict(text='<b>CBOE SKEW Index</b><sup>  ^SKEW yfinance · > 150 = demanda extrema de cola</sup>',
                   font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='SKEW', gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        hovermode='x unified')
    return fig
def build_credit_chart(bt, window=252):
    """
    Credit Spread vs VIX — HYG/IEF live desde yfinance.
    Credit Spread = HYG yield spread proxy: -(HYG_ret_20 - IEF_ret_20)
    Spread positivo = crédito se amplía = risk-off → alertar al VRP trader.
    """
    needed = ['Credit_Spread', 'VIX_Close']
    if not all(c in bt.columns for c in needed):
        return go.Figure()
    p = bt[needed].tail(window).dropna(subset=['Credit_Spread'])
    if len(p) < 10:
        return go.Figure()

    # Percentiles para bandas de contexto
    p75 = float(p['Credit_Spread'].quantile(0.75))
    p90 = float(p['Credit_Spread'].quantile(0.90))

    fig = go.Figure()
    # Área credit spread
    colors_cs = ['#EA3943' if v > 0 else '#16C784' for v in p['Credit_Spread']]
    fig.add_trace(go.Scatter(
        x=p.index, y=p['Credit_Spread'].clip(lower=0),
        name='Spread widening (risk-off)',
        fill='tozeroy', line=dict(color='#EA3943', width=0),
        fillcolor='rgba(234,57,67,0.15)',
        hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=p.index, y=p['Credit_Spread'], name='Credit Spread (HYG-IEF · yfinance)',
        line=dict(color='#F5A623', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Spread: %{y:.2f}<extra></extra>'))
    # VIX en eje derecho
    fig.add_trace(go.Scatter(
        x=p.index, y=p['VIX_Close'], name='VIX (^VIX)',
        yaxis='y2', line=dict(color='#EA3943', width=1.5, dash='dot'),
        hovertemplate='VIX: %{y:.1f}<extra></extra>'))
    # Líneas de percentil
    fig.add_hline(y=0, line_dash='dash', line_color='#3A3F47', line_width=1)
    fig.add_hline(y=p75, line_dash='dot', line_color='#F5A623', line_width=1,
        annotation_text=f'  P75: {p75:.2f}',
        annotation_font=dict(color='#F5A623', size=8))
    fig.add_hline(y=p90, line_dash='dash', line_color='#EA3943', line_width=1,
        annotation_text=f'  P90: {p90:.2f} (stress)',
        annotation_font=dict(color='#EA3943', size=8))
    fig.update_layout(
        title=dict(
            text='<b>Credit Spread vs VIX</b><sup>  HYG/IEF yfinance · Divergencia credit/VIX = warning</sup>',
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=50, r=60, t=55, b=40),
        xaxis=dict(gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(title='Credit Spread (20d momentum)',
                   gridcolor='#1C1F24', tickfont=dict(size=9, color='#9AA1A9')),
        yaxis2=dict(title='VIX', overlaying='y', side='right',
                    tickfont=dict(size=9, color='#EA3943'), showgrid=False),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        hovermode='x unified')
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MONITOR OPERATIVO — DATA LAYER (parquet local del repo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARQUET_PATH      = "data/master.parquet"
BARO_PARQUET_PATH = "data/baro_history.parquet"
VX_CURVE_PARQUET_PATH = "data/vx_curve_history.parquet"

@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["parquet"])
def load_vx_curve_parquet() -> pd.DataFrame:
    """
    Histórico DIARIO de la curva de futuros del VIX (M1..M8, DTE, OI, VIX,
    contango/basis/roll). Lo escribe scripts/update_vx_curve.py desde el CDN
    público de CBOE, vía GitHub Action diaria. Vacío si aún no existe.
    """
    log = logging.getLogger("vix_controller")
    try:
        df = pd.read_parquet(VX_CURVE_PARQUET_PATH)
        df.index = pd.DatetimeIndex(df.index).normalize()
        df = df.sort_index()
        # Calidad de la curva: ~1 mes entre M1 y M2. Si no, falta un contrato
        # en el CDN de CBOE y "M2" es en realidad el de dos meses.
        if {"dias_m1", "dias_m2"} <= set(df.columns) and "gap_ok" not in df.columns:
            gap = df["dias_m2"] - df["dias_m1"]
            df["gap_ok"] = (gap >= 20) & (gap <= 45)
        return df
    except FileNotFoundError:
        log.info("vx_curve parquet no encontrado en %s", VX_CURVE_PARQUET_PATH)
        return pd.DataFrame()
    except Exception as e:                       # noqa: BLE001
        log.warning("Error leyendo vx_curve parquet: %s", e)
        return pd.DataFrame()


def build_curve_signal_history_chart(hist: pd.DataFrame, curve: pd.DataFrame,
                                     days: int = 504) -> go.Figure:
    """
    Cómo ha cambiado la lectura de la curva en el tiempo:
      panel 1: M1 y VIX (la base) con bandas de color por señal
      panel 2: contango M1→M2 (%) y su percentil rolling
    """
    h = hist.tail(days)
    c = curve.reindex(h.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.42], vertical_spacing=0.06,
                        specs=[[{}], [{"secondary_y": True}]],
                        subplot_titles=("<b>VIX spot y M1 · fondo = señal de curva</b>",
                                        "<b>Contango M1→M2 y percentil rolling 5a</b>"))
    # Marca: relleno ámbar al 14 % = zona favorable al carry; rojo tenue = long vol.
    colors = {tsig.SIGNAL_SHORT: "rgba(245,166,35,0.14)",
              tsig.SIGNAL_LONG: "rgba(234,57,67,0.16)",
              tsig.SIGNAL_NEUTRAL: "rgba(0,0,0,0)"}
    codes = h["signal"].to_numpy()
    idx = h.index
    fig.add_trace(go.Scatter(x=idx, y=c["m1"].tolist(), name="M1 (settle)", mode="lines",
                             line=dict(color="#F5A623", width=1.8),
                             hovertemplate="%{x|%Y-%m-%d}<br>M1: %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=c["VIX"].tolist(), name="VIX spot", mode="lines",
                             line=dict(color="#F4F5F6", width=1.4),
                             hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    # Bandas por señal — DESPUÉS de los traces del panel: add_vrect(row=1) es
    # un no-op silencioso si el subplot aún no tiene datos (plotly >= 6).
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            fig.add_vrect(x0=idx[start], x1=idx[min(i, len(codes) - 1)],
                          fillcolor=colors.get(codes[start], "rgba(0,0,0,0)"),
                          line_width=0, layer="below", row=1, col=1)
            start = i
    bar_clr = ["#16C784" if v > 0 else "#EA3943" for v in h["contango_pct"].fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=h["contango_pct"].tolist(), name="Contango M1→M2 %",
                         marker_color=bar_clr, opacity=0.8,
                         hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}%<extra></extra>"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=h["ct_pctile"].tolist(), name="Percentil 5a (eje dcho.)",
                             mode="lines",
                             line=dict(color="#9AA1A9", width=1.2, dash="dot"),
                             hovertemplate="%{x|%Y-%m-%d}<br>p%{y:.0f}<extra></extra>"),
                  row=2, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="#3A3F47", line_width=1, row=2, col=1, secondary_y=False)
    # Proxies de leyenda para las bandas (x con fecha válida — ver bug plotly.js)
    for name, key in (("Short vol favorable", tsig.SIGNAL_SHORT),
                      ("Long vol", tsig.SIGNAL_LONG), ("Neutral", tsig.SIGNAL_NEUTRAL)):
        fig.add_trace(go.Scatter(x=[idx[0]], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, symbol="square",
                                             color={tsig.SIGNAL_SHORT: "rgba(245,166,35,0.6)",
                                                    tsig.SIGNAL_LONG: "rgba(234,57,67,0.6)",
                                                    tsig.SIGNAL_NEUTRAL: "#3A3F47"}[key]),
                                 hoverinfo="skip"), row=1, col=1)
    fig.update_layout(
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=55, r=25, t=45, b=60), hovermode="x unified",
        bargap=0,
        legend=dict(orientation="h", yanchor="top", y=-0.07, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#F4F5F6", family="JetBrains Mono")))
    for ann in fig["layout"]["annotations"][:2]:
        ann["font"] = dict(size=11, color="#9AA1A9", family="Inter")
        ann["xanchor"] = "left"; ann["x"] = 0.01
    fig.update_xaxes(gridcolor="#1C1F24", tickfont=dict(size=10, color="#9AA1A9"))
    fig.update_yaxes(gridcolor="#1C1F24", tickfont=dict(size=9.5, color="#9AA1A9"))
    fig.update_yaxes(ticksuffix="%", row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 100], showgrid=False, ticksuffix="", row=2, col=1,
                     secondary_y=True, title=dict(text="percentil",
                                                  font=dict(size=9, color="#9AA1A9")))
    return fig


def load_baro_parquet() -> pd.DataFrame:
    """
    Lee el parquet histórico del Barómetro VTS.
    Contiene 15+ años de data de yfinance para todos los tickers del barómetro.
    Actualizado diariamente vía GitHub Action (scripts/update_baro_parquet.py).

    Columnas esperadas: VIX, VIX9D, VIX3M, VIX6M, VIX1Y, VVIX, SKEW,
                        SPY, HYG, IEF, VXX, SVXY
    """
    log = logging.getLogger("vix_controller")
    try:
        df = pd.read_parquet(BARO_PARQUET_PATH)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        log.info(f"Baro parquet: {len(df):,} filas · {len(df.columns)} tickers · "
                 f"{df.index[-1].strftime('%Y-%m-%d')}")
        return df
    except FileNotFoundError:
        log.info(f"Baro parquet no encontrado en {BARO_PARQUET_PATH} — usando yfinance live")
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"Error leyendo baro parquet: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["parquet"])
def load_master_parquet() -> pd.DataFrame:
    """
    Lee el histórico desde data/master.parquet (repo de GitHub).
    Instantáneo — sin red, sin Drive, sin gdown.
    El notebook exporta: df.to_parquet('data/master.parquet') y hace push.
    Columnas clave: VXX_Close, M1_Price, In_Contango, Contango_pct, VIX_Close
    """
    log = logging.getLogger("vix_controller")
    try:
        df = pd.read_parquet(PARQUET_PATH)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        log.info(f"Parquet: {len(df):,} filas · {df.index[-1].strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        log.error(f"Error parquet: {e}")
        return pd.DataFrame()




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WALK-FORWARD BACKTEST ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━













@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["strategy"])
def build_strategy_cached(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica BB(20, 2σ) + Contango Rule sobre el histórico completo.
    Cacheado 1h — mismo TTL que el parquet.

    Lógica exacta del notebook:
      Entrada : VXX < SMA(20)       → pos=1 (BB timing)
      Salida  : VXX > BB_Upper(2σ)  → pos=0 (salida por BB)
               O In_Contango == 0   → pos=0 (salida por CT)
      Filtro  : contango_filter = In_Contango (sin shift — es dato del cierre)
      sig_final = sig_bb × ct_filter  (shift ya aplicado en sig_bb)
    """
    bt = df[df['VXX_Close'].notna() & df['M1_Price'].notna()].copy()

    vxx = bt['VXX_Close']
    bt['BB_SMA20'] = vxx.rolling(20).mean()
    bt['BB_STD20'] = vxx.rolling(20).std()
    bt['BB_Upper'] = bt['BB_SMA20'] + 2.0 * bt['BB_STD20']
    bt['BB_Lower'] = bt['BB_SMA20'] - 2.0 * bt['BB_STD20']

    # Señal BB pura — máquina de estados vectorizada en numpy (misma lógica,
    # antes iteraba con .iloc fila a fila)
    sig = pd.Series(
        bb_position_state(bt['VXX_Close'].to_numpy(dtype=float),
                          bt['BB_SMA20'].to_numpy(dtype=float),
                          bt['BB_Upper'].to_numpy(dtype=float)),
        index=bt.index)

    bt['sig_bb']    = sig.shift(1).fillna(0).astype(int)
    bt['ct_filter'] = bt['In_Contango'].fillna(0).astype(int)
    bt['sig_final'] = (bt['sig_bb'] * bt['ct_filter']).astype(int)
    return bt




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VTS VOLATILITY BAROMETER — 13 métricas de volatilidad
# Inspirado en VolatilityTradingStrategies.com (Brent Osachoff)
# Cada métrica se convierte a percentil rolling (252d o lifetime),
# luego se promedian para obtener un score 0-100%.
#
# Lectura:
#   0-20%   : Vol BAJA     → SVXY/SVIX net short vol (agresivo)
#   20-40%  : Vol moderada → SVXY (posición normal)
#   40-60%  : Vol mid      → Cash / parcial
#   60-80%  : Vol ELEVADA  → Cash / defensivo
#   80-100% : Vol EXTREMA  → Long VIX / hedge / short equities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rolling_percentile(s: pd.Series, window: int = cfg.VTS_ROLLING_WINDOW) -> pd.Series:
    """
    Percentil rolling del último valor vs la ventana histórica.
    Devuelve 0-100. Implementación vectorizada con numpy (mucho más rápida
    que .apply con lambda).

    Cada valor en la salida es: "qué % de los últimos `window` valores
    son <= al valor actual".

    Robust to NaN: valores NaN en la entrada producen NaN en la salida.

    B7 fix: min_obs sube de max(30, window//5) a max(60, window//3) para
    evitar percentiles con soporte insuficiente (antes los primeros 30 días
    de una serie producían percentiles basados en <30 obs válidas).
    """
    # Delegado al módulo vectorizado (rolling.rank). Mismos semánticos:
    # min_obs = max(60, window//3), NaN → NaN, empates cuentan como <=.
    return _rolling_percentile_mod(s, window=window)


@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["barometer"])
def compute_vts_barometer(
    bt: pd.DataFrame,
    edge_extra: dict,
    gex_summary: dict | None = None,
    skew_metrics: dict | None = None,
    window: int = 252,
) -> dict:
    """
    VTS Volatility Barometer — 13 indicadores alineados al original
    de volatilitytradingstrategies.com (Brent Osachoff).

    Indicadores (según la imagen oficial del gauge VTS):
        1.  M1:M2 VIX FUTURES          — contango front
        2.  M4-M7 VIX FUTURES          — contango curva larga
        3.  VX30:VIX ROLL YIELD        — roll yield constant-maturity
        4.  VIX / VVIX / VOLI          — VIX·VVIX composite (VOLI → RV20 proxy)
        5.  CASH VIX OSCILLATOR        — combinación de ratios VIX term structure
        6.  VIX - VOLI RESIDUAL        — VIX menos VOLI (RV20 proxy)
        7.  TRADERS VRP                — WMA(VX30 - HV5, w=1..5)
        8.  SDEX / TDEX / SKEW         — CBOE SKEW (+ proxies SDEX/TDEX)
        9.  VIX:VIX3M MEDIUM           — ratio medium term
        10. EXTREME PUT/CALL RATIO     — skew extremo (proxy)
        11. VIX9D:VIX FAST             — ratio fast term
        12. VOLPOCALYPSE THRESHOLD     — trigger compuesto de vol extrema
        13. (el core: VIX level)       — nivel absoluto VIX

    Todas las métricas se convierten a percentil rolling (window días),
    normalizadas a 0-100 donde 100 = máximo stress histórico.

    Returns
    -------
    dict con score (0-100), regime, position, metrics (lista de dicts),
    history (serie), window, date.
    """
    if bt.empty or len(bt) < 60:
        return {}

    df = bt.copy()
    metrics = []
    log = logging.getLogger("vix_controller")

    # ══════════════════════════════════════════════════════
    # EXTENDER ÍNDICE HACIA ATRÁS usando yfinance
    # VTS usa percentiles lifetime (desde 2011+). Para replicarlo,
    # extendemos el índice con toda la historia disponible de
    # los tickers yfinance (SPY, VIX, VIX3M, etc.) antes de calcular
    # percentiles. Así la ventana rolling cubre años antes del parquet.
    # ══════════════════════════════════════════════════════
    extended_index = df.index
    if edge_extra:
        # Encontrar el índice más largo disponible
        all_indices = [df.index]
        for key in ('VIX', 'VIX3M', 'SKEW', 'VVIX'):
            if key in edge_extra and not edge_extra[key].empty:
                all_indices.append(edge_extra[key].index)
        if all_indices:
            earliest = min(idx.min() for idx in all_indices if len(idx) > 0)
            latest   = df.index.max()
            # Business days desde earliest a latest
            extended_index = pd.bdate_range(earliest, latest)
            log.info(f"BARO: índice extendido {earliest.date()} → {latest.date()} "
                     f"({len(extended_index):,} días hábiles)")
            # Reindexar df al nuevo índice
            df = df.reindex(extended_index)

    # ── Helpers para series de la familia VIX ──────────────────
    def _get_series(key: str) -> pd.Series | None:
        """Intenta obtener una serie de edge_extra alineada al índice extendido."""
        if key in edge_extra and not edge_extra[key].empty:
            return edge_extra[key]['Close'].reindex(df.index).ffill()
        return None

    # VIX desde yfinance si está disponible (preferimos este al parquet
    # porque tiene historia completa desde 1990)
    vix_yf = _get_series('VIX')
    vix_parquet = df.get('VIX_Close')

    if vix_yf is not None and vix_yf.notna().sum() > 1000:
        # yfinance tiene muchos más datos → usar como fuente primaria
        vix = vix_yf
        df['VIX_Close'] = vix
        log.info(f"BARO: usando VIX de yfinance ({vix.notna().sum():,} filas)")
    elif vix_parquet is not None and vix_parquet.notna().sum() > 60:
        vix = vix_parquet
    else:
        # Fallback: descargar en caliente (vía wrapper safe → respeta circuit breaker)
        vix = None
        vh = _yf_history_safe("^VIX", period="max")
        if vh is not None and not vh.empty and "Close" in vh.columns:
            try:
                vh.index = pd.DatetimeIndex(vh.index).tz_localize(None).normalize()
                vix = vh['Close'].reindex(df.index).ffill()
                df['VIX_Close'] = vix
                log.info(f"BARO: VIX descargado fresh ({vix.notna().sum():,} filas)")
            except Exception as e:
                log.warning(f"BARO: VIX post-process failed: {e}")
        else:
            log.warning("BARO: VIX no disponible (yfinance vacío o rate-limit)")

    # SPY con historia completa también (para RV20)
    spy = df.get('SPY_Close')
    if spy is None or spy.notna().sum() < 100:
        sh = _yf_history_safe("SPY", period="max")
        if sh is not None and not sh.empty and "Close" in sh.columns:
            try:
                sh.index = pd.DatetimeIndex(sh.index).tz_localize(None).normalize()
                spy = sh['Close'].reindex(df.index).ffill()
                df['SPY_Close'] = spy
                log.info(f"BARO: SPY descargado ({spy.notna().sum():,} filas)")
            except Exception as e:
                log.warning(f"BARO: SPY post-process failed: {e}")
        else:
            log.warning("BARO: SPY no disponible (yfinance vacío o rate-limit)")

    vvix_s  = _get_series('VVIX')
    if vvix_s is None and 'VVIX_Live' in df.columns:
        vvix_s = df['VVIX_Live']
    vix9d   = _get_series('VIX9D')
    vix3m   = _get_series('VIX3M')
    vix6m   = _get_series('VIX6M')
    vix1y   = _get_series('VIX1Y')
    skew_s  = _get_series('SKEW')

    # Log de diagnóstico
    log.info(
        f"BARO inputs: VIX={(vix is not None and vix.notna().sum())}, "
        f"SPY={(spy is not None and spy.notna().sum())}, "
        f"VVIX={(vvix_s is not None and vvix_s.notna().sum())}, "
        f"VIX3M={(vix3m is not None and vix3m.notna().sum())}, "
        f"VIX9D={(vix9d is not None and vix9d.notna().sum())}, "
        f"SKEW={(skew_s is not None and skew_s.notna().sum())}, "
        f"M1={'M1_Price' in df.columns}, M2={'M2_Price' in df.columns}, "
        f"M4={'M4_Price' in df.columns}, M7={'M7_Price' in df.columns}"
    )

    # VX30 constant-maturity: interpolación lineal entre M1 y M2 a 30 días
    vx30 = None
    if all(c in df.columns for c in ['M1_Price', 'M2_Price', 'M1_DTE']):
        dte1 = df['M1_DTE'].clip(lower=0)
        # Si M2_DTE no existe, asumimos ~30 días después de M1
        m2_dte = df['M2_DTE'] if 'M2_DTE' in df.columns else (dte1 + 30)
        # Interpolación a 30 días
        t = (30 - dte1) / (m2_dte - dte1).replace(0, np.nan)
        t = t.clip(0, 1)
        vx30 = df['M1_Price'] * (1 - t) + df['M2_Price'] * t
        vx30 = vx30.where(vx30 > 0)
    elif 'M1_Price' in df.columns and 'M2_Price' in df.columns:
        # Sin DTE, usamos promedio simple M1-M2 como proxy de VX30
        vx30 = 0.5 * (df['M1_Price'] + df['M2_Price'])

    # HV (historical volatility) del SPY
    hv5 = hv10 = hv20 = None
    if spy is not None:
        log_ret = np.log(spy / spy.shift(1))
        hv5  = log_ret.rolling(5).std()  * np.sqrt(252) * 100
        hv10 = log_ret.rolling(10).std() * np.sqrt(252) * 100
        hv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100

    # ══════════════════════════════════════════════════════
    # 1. M1:M2 VIX FUTURES — contango frontal
    # Valor alto de backwardation (ratio > 1) = stress
    # Invertimos: contango alto → percentil bajo (vol baja)
    # ══════════════════════════════════════════════════════
    if 'M1_Price' in df.columns and 'M2_Price' in df.columns:
        m1m2 = df['M1_Price'] / df['M2_Price']  # >1 = backwardation
        m1m2_pct = _rolling_percentile(m1m2, window)
        metrics.append({
            'name': 'M1:M2 VIX FUTURES',
            'value': f"{m1m2.iloc[-1]:.3f}" if pd.notna(m1m2.iloc[-1]) else '—',
            'percentile': m1m2_pct.iloc[-1] if pd.notna(m1m2_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["m1_m2"],
            'interpretation': 'Ratio M1/M2 — >1 indica backwardation (stress)',
            'series': m1m2_pct,
        })

    # ══════════════════════════════════════════════════════
    # 2. M4-M7 VIX FUTURES — contango curva larga
    # ══════════════════════════════════════════════════════
    if 'M4_Price' in df.columns and 'M7_Price' in df.columns:
        m4m7 = df['M4_Price'] / df['M7_Price']  # >1 = backwardation en curva larga
        m4m7_pct = _rolling_percentile(m4m7, window)
        metrics.append({
            'name': 'M4-M7 VIX FUTURES',
            'value': f"{m4m7.iloc[-1]:.3f}" if pd.notna(m4m7.iloc[-1]) else '—',
            'percentile': m4m7_pct.iloc[-1] if pd.notna(m4m7_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["m4_m7"],
            'interpretation': 'Ratio M4/M7 — inversión en curva larga = stress estructural',
            'series': m4m7_pct,
        })

    # ══════════════════════════════════════════════════════
    # 3. VX30:VIX ROLL YIELD — roll yield constant-maturity
    # Formula: (VX30 - VIX) / VIX
    # Positivo (contango) = vol baja. Invertimos para score.
    # ══════════════════════════════════════════════════════
    if vx30 is not None and vix is not None:
        roll_y = (vx30 - vix) / vix * 100  # % roll yield
        ry_pct = _rolling_percentile(roll_y, window)
        inv_ry = 100 - ry_pct  # alto roll yield = vol baja (invertimos)
        metrics.append({
            'name': 'VX30:VIX ROLL YIELD',
            'value': f"{roll_y.iloc[-1]:+.2f}%" if pd.notna(roll_y.iloc[-1]) else '—',
            'percentile': inv_ry.iloc[-1] if pd.notna(inv_ry.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["vx30_vix_roll"],
            'interpretation': 'Roll yield constant-maturity 30d · Alto = vol baja',
            'series': inv_ry,
        })

    # ══════════════════════════════════════════════════════
    # 4. VIX / VVIX / VOLI — composite
    # Tomamos el percentil ponderado del trio. VOLI → RV20 proxy
    # ══════════════════════════════════════════════════════
    trio_pctiles = []
    if vix is not None:
        trio_pctiles.append(_rolling_percentile(vix, window).iloc[-1])
    if vvix_s is not None and vvix_s.notna().sum() > 60:
        trio_pctiles.append(_rolling_percentile(vvix_s, window).iloc[-1])
    if hv20 is not None and hv20.notna().sum() > 60:
        # VOLI proxy = RV20 (historial de vol realizada 20d)
        trio_pctiles.append(_rolling_percentile(hv20, window).iloc[-1])
    if trio_pctiles:
        trio_pctiles = [p for p in trio_pctiles if pd.notna(p)]
        trio_val = np.mean(trio_pctiles) if trio_pctiles else 50
        # Serie histórica: promedio de los 3 percentiles
        trio_series = []
        if vix is not None:
            trio_series.append(_rolling_percentile(vix, window))
        if vvix_s is not None and vvix_s.notna().sum() > 60:
            trio_series.append(_rolling_percentile(vvix_s, window))
        if hv20 is not None and hv20.notna().sum() > 60:
            trio_series.append(_rolling_percentile(hv20, window))
        trio_hist = pd.concat(trio_series, axis=1).mean(axis=1) if trio_series else None
        metrics.append({
            'name': 'VIX / VVIX / VOLI',
            'value': (f"V={vix.iloc[-1]:.1f}" if vix is not None else '') +
                     (f" VV={vvix_s.iloc[-1]:.0f}" if vvix_s is not None and pd.notna(vvix_s.iloc[-1]) else ''),
            'percentile': trio_val,
            'weight': cfg.VTS_WEIGHTS["vix_vvix_voli"],
            'interpretation': 'Composite nivel: VIX + VVIX + VOLI proxy (RV20)',
            'series': trio_hist,
        })

    # ══════════════════════════════════════════════════════
    # 5. CASH VIX OSCILLATOR — combinación de ratios term structure
    # CVO = promedio de (VIX9D/VIX, VIX/VIX3M, VIX/VIX6M, VIX/VIX1Y)
    # Alto = inversión de curva = stress
    # ══════════════════════════════════════════════════════
    ratios = []
    ratio_series = []
    if vix9d is not None and vix is not None:
        r = vix9d / vix
        if r.notna().sum() > 30:
            ratios.append(r); ratio_series.append(r)
    if vix is not None and vix3m is not None:
        r = vix / vix3m
        if r.notna().sum() > 30:
            ratios.append(r); ratio_series.append(r)
    if vix is not None and vix6m is not None:
        r = vix / vix6m
        if r.notna().sum() > 30:
            ratios.append(r); ratio_series.append(r)
    if vix is not None and vix1y is not None:
        r = vix / vix1y
        if r.notna().sum() > 30:
            ratios.append(r); ratio_series.append(r)
    if ratios:
        cvo = pd.concat(ratios, axis=1).mean(axis=1)
        cvo_pct = _rolling_percentile(cvo, window)
        metrics.append({
            'name': 'CASH VIX OSCILLATOR',
            'value': f"{cvo.iloc[-1]:.3f}" if pd.notna(cvo.iloc[-1]) else '—',
            'percentile': cvo_pct.iloc[-1] if pd.notna(cvo_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["cvo"],
            'interpretation': 'Promedio ratios VIX term structure · Alto = curva invertida',
            'series': cvo_pct,
        })

    # ══════════════════════════════════════════════════════
    # 6. VIX - VOLI RESIDUAL — spread VIX menos VOLI (proxy RV20)
    # VIX - RV20: bajo o negativo = vol exigida = stress
    # Invertimos: residual bajo = score alto
    # ══════════════════════════════════════════════════════
    if vix is not None and hv20 is not None:
        residual = vix - hv20
        res_pct = _rolling_percentile(residual, window)
        inv_res = 100 - res_pct
        metrics.append({
            'name': 'VIX - VOLI RESIDUAL',
            'value': f"{residual.iloc[-1]:+.2f}" if pd.notna(residual.iloc[-1]) else '—',
            'percentile': inv_res.iloc[-1] if pd.notna(inv_res.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["vix_voli_resid"],
            'interpretation': 'VIX menos VOLI (RV20 proxy) · Bajo = mercado exige prima',
            'series': inv_res,
        })

    # ══════════════════════════════════════════════════════
    # 7. TRADERS VRP — WMA(VX30 - HV5, pesos 1,2,3,4,5)
    # Traders VRP bajo = prima comprimida = stress. Invertimos.
    # ══════════════════════════════════════════════════════
    if vx30 is not None and hv5 is not None:
        vrp = vx30 - hv5
        # WMA 5-day con pesos 1,2,3,4,5 (normalizado)
        weights = np.array([1, 2, 3, 4, 5], dtype=float)
        weights /= weights.sum()
        traders_vrp = vrp.rolling(5).apply(
            lambda x: np.dot(x, weights) if len(x) == 5 else np.nan,
            raw=True,
        )
        tvrp_pct = _rolling_percentile(traders_vrp, window)
        inv_tvrp = 100 - tvrp_pct  # VRP bajo = stress alto
        metrics.append({
            'name': 'TRADERS VRP',
            'value': f"{traders_vrp.iloc[-1]:+.2f}" if pd.notna(traders_vrp.iloc[-1]) else '—',
            'percentile': inv_tvrp.iloc[-1] if pd.notna(inv_tvrp.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["traders_vrp"],
            'interpretation': 'WMA(VX30 - HV5) · Bajo = prima comprimida (stress)',
            'series': inv_tvrp,
        })

    # ══════════════════════════════════════════════════════
    # 8. SDEX / TDEX / SKEW — CBOE SKEW Index + proxies
    # SKEW alto = demanda de puts OTM = stress
    # SDEX proxy = skew normalizado · TDEX proxy = SKEW rolling
    # ══════════════════════════════════════════════════════
    if skew_s is not None and skew_s.notna().sum() > 60:
        skew_pct = _rolling_percentile(skew_s, window)
        metrics.append({
            'name': 'SDEX / TDEX / SKEW',
            'value': f"{skew_s.iloc[-1]:.1f}" if pd.notna(skew_s.iloc[-1]) else '—',
            'percentile': skew_pct.iloc[-1] if pd.notna(skew_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["skew"],
            'interpretation': 'CBOE SKEW Index · Alto = demanda de puts OTM (cola izq)',
            'series': skew_pct,
        })

    # ══════════════════════════════════════════════════════
    # 9. VIX:VIX3M MEDIUM — ratio medium term
    # >1 = backwardation medio plazo = stress
    # ══════════════════════════════════════════════════════
    if vix is not None and vix3m is not None:
        vv3m = vix / vix3m
        if vv3m.notna().sum() > 30:
            vv3m_pct = _rolling_percentile(vv3m, window)
            metrics.append({
                'name': 'VIX:VIX3M MEDIUM',
                'value': f"{vv3m.iloc[-1]:.3f}" if pd.notna(vv3m.iloc[-1]) else '—',
                'percentile': vv3m_pct.iloc[-1] if pd.notna(vv3m_pct.iloc[-1]) else 50,
                'weight': cfg.VTS_WEIGHTS["vix_vix3m"],
                'interpretation': 'VIX/VIX3M · >1 = backwardation medio plazo',
                'series': vv3m_pct,
            })

    # ══════════════════════════════════════════════════════
    # 10. EXTREME PUT/CALL RATIO — nivel elevado del SKEW
    # Usamos SKEW suavizado (media 10d) para evitar ruido diario.
    # SKEW > 140 históricamente indica demanda extrema de puts OTM.
    # ══════════════════════════════════════════════════════
    if skew_s is not None and skew_s.notna().sum() > 60:
        # SKEW suavizado — promedio 10d para eliminar ruido
        skew_smooth = skew_s.rolling(10, min_periods=3).mean()
        skew_level_pct = _rolling_percentile(skew_smooth, window)
        last_val = skew_smooth.iloc[-1]
        metrics.append({
            'name': 'EXTREME PUT/CALL RATIO',
            'value': f"{last_val:.1f}" if pd.notna(last_val) else '—',
            'percentile': skew_level_pct.iloc[-1] if pd.notna(skew_level_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["extreme_pc"],
            'interpretation': 'SKEW suavizado 10d · Alto = demanda de puts elevada',
            'series': skew_level_pct,
        })

    # ══════════════════════════════════════════════════════
    # 11. VIX9D:VIX FAST — ratio fast term
    # >1 = vol cola corta > vol 30d = stress inmediato
    # ══════════════════════════════════════════════════════
    if vix9d is not None and vix is not None:
        fast = vix9d / vix
        if fast.notna().sum() > 30:
            fast_pct = _rolling_percentile(fast, window)
            metrics.append({
                'name': 'VIX9D:VIX FAST',
                'value': f"{fast.iloc[-1]:.3f}" if pd.notna(fast.iloc[-1]) else '—',
                'percentile': fast_pct.iloc[-1] if pd.notna(fast_pct.iloc[-1]) else 50,
                'weight': cfg.VTS_WEIGHTS["vix9d_vix"],
                'interpretation': 'VIX9D/VIX · >1 = cola corta caliente (stress inmediato)',
                'series': fast_pct,
            })

    # ══════════════════════════════════════════════════════
    # 12. VOLPOCALYPSE THRESHOLD — trigger compuesto vol extrema
    # Combinación: VIX absoluto + backwardation M1/M2 + momentum VIX 5d
    # Score 0-100 directo. Diseñado para activarse FUERTE en crisis,
    # suave en rangos normales (VIX 15-25 histórico = ~40-50 percentil).
    # ══════════════════════════════════════════════════════
    if vix is not None and vix.notna().sum() > 30:
        vix_nona = vix.dropna()
        last_vix = float(vix_nona.iloc[-1])

        # Ratio M1/M2 si existe; si no, neutral
        if ('M1_Price' in df.columns and 'M2_Price' in df.columns
            and df['M1_Price'].notna().sum() > 0
            and df['M2_Price'].notna().sum() > 0):
            m1m2_ser = df['M1_Price'] / df['M2_Price']
            m1m2_last = float(m1m2_ser.dropna().iloc[-1]) if m1m2_ser.notna().any() else 1.0
        else:
            m1m2_ser = pd.Series([1.0] * len(vix), index=vix.index)
            m1m2_last = 1.0

        vix_5d_ser = vix.pct_change(5).fillna(0) * 100
        vix_5d_last = float(vix_5d_ser.iloc[-1])

        def _volp_score(v, r, ch):
            """
            Mapeo menos agresivo en rango normal:
              VIX < 13 → 3    VIX 13-16 → 10   VIX 16-20 → 25
              VIX 20-25 → 42  VIX 25-30 → 58   VIX 30-40 → 75   VIX 40+ → 92
            """
            if pd.isna(v): return np.nan
            if   v < 13: s = 3
            elif v < 16: s = 10
            elif v < 20: s = 25
            elif v < 25: s = 42
            elif v < 30: s = 58
            elif v < 40: s = 75
            else:        s = 92
            # Backwardation boost (solo si es real)
            if pd.notna(r):
                if r > 1.02:   s = min(100, s + 12)
                elif r > 1.00: s = min(100, s + 4)
            # Momentum boost (solo si muy fuerte)
            if pd.notna(ch):
                if ch > 40:    s = min(100, s + 10)
                elif ch > 20:  s = min(100, s + 4)
            return s

        vix_score = _volp_score(last_vix, m1m2_last, vix_5d_last)

        # Serie histórica vectorizada (numpy)
        m1m2_arr = m1m2_ser.reindex(vix.index).fillna(1.0).values
        ch5_arr = vix_5d_ser.values
        v_arr = vix.values
        hist = np.full(len(vix), np.nan)
        for i in range(len(v_arr)):
            hist[i] = _volp_score(v_arr[i], m1m2_arr[i], ch5_arr[i])
        hist_scores = pd.Series(hist, index=vix.index)

        metrics.append({
            'name': 'VOLPOCALYPSE THRESHOLD',
            'value': f"VIX={last_vix:.1f} · M1/M2={m1m2_last:.3f}",
            'percentile': vix_score,
            'weight': cfg.VTS_WEIGHTS["volpocalypse"],
            'interpretation': 'Trigger compuesto · VIX + backwardation + momentum',
            'series': hist_scores,
        })

    # ══════════════════════════════════════════════════════
    # 13. VIX LEVEL (core — nivel absoluto del VIX)
    # Percentil rolling del nivel absoluto · mantenemos el VIX como ancla
    # ══════════════════════════════════════════════════════
    if vix is not None:
        vix_pct = _rolling_percentile(vix, window)
        metrics.append({
            'name': 'VIX LEVEL (core)',
            'value': f"{vix.iloc[-1]:.2f}",
            'percentile': vix_pct.iloc[-1] if pd.notna(vix_pct.iloc[-1]) else 50,
            'weight': cfg.VTS_WEIGHTS["vix_level"],
            'interpretation': 'Nivel absoluto del VIX · Ancla del barómetro',
            'series': vix_pct,
        })

    # ══════════════════════════════════════════════════════
    # SCORE FINAL — promedio ponderado de los percentiles
    # ══════════════════════════════════════════════════════
    valid_metrics = [m for m in metrics if pd.notna(m['percentile'])]
    log.info(f"BARO: {len(valid_metrics)}/{len(metrics)} métricas válidas")

    if not valid_metrics:
        # Fallback de emergencia: si ninguna métrica calculó pero tenemos VIX,
        # al menos damos una lectura basada en el VIX absoluto (heurística)
        if vix is not None and vix.notna().sum() > 0:
            last_vix = vix.dropna().iloc[-1]
            # Score heurístico simple
            if   last_vix < 13: s = 10
            elif last_vix < 16: s = 25
            elif last_vix < 20: s = 40
            elif last_vix < 25: s = 55
            elif last_vix < 30: s = 70
            elif last_vix < 40: s = 85
            else:               s = 95
            log.warning(f"BARO: fallback heurístico VIX-only · score={s}")
            metrics.append({
                'name': 'VIX LEVEL (fallback)',
                'value': f"{last_vix:.2f}",
                'percentile': s,
                'weight': 1.0,
                'interpretation': 'Fallback: solo VIX absoluto (datos insuficientes)',
                'series': None,
            })
            valid_metrics = metrics
        else:
            log.error("BARO: retornando vacío — no se pudo calcular ninguna métrica")
            return {}

    total_w    = sum(m['weight'] for m in valid_metrics)
    weighted_s = sum(m['percentile'] * m['weight'] for m in valid_metrics)
    score = weighted_s / total_w if total_w > 0 else 50.0

    # Régimen (alineado a VTS: lower = low vol env, higher = high vol env)
    if   score < 20: regime, position = "VOL BAJA",  "Aggressive short vol (SVXY/SVIX)"
    elif score < 40: regime, position = "MODERADA",  "Short vol estándar (SVXY)"
    elif score < 60: regime, position = "MID",       "Cash / posición parcial"
    elif score < 80: regime, position = "ELEVADA",   "Cash / defensivo"
    else:            regime, position = "EXTREMA",   "Long VIX / hedge / short equities"

    # Histórico del score
    score_hist = None
    series_list = [(m['series'], m['weight']) for m in metrics
                    if m.get('series') is not None]
    if series_list:
        weighted_df = pd.concat(
            [s * w for s, w in series_list], axis=1
        ).sum(axis=1, min_count=1)
        total_weights = pd.concat(
            [s.notna().astype(float) * w for s, w in series_list], axis=1
        ).sum(axis=1)
        score_hist = (weighted_df / total_weights).dropna()

    return {
        'score':    float(score),
        'regime':   regime,
        'position': position,
        'metrics':  metrics,
        'history':  score_hist,
        'window':   window,
        'date':     df.index[-1],
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — HMM RÉGIMEN + CROSS-ASSET EARLY WARNING (helpers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ee_close(ee: dict, key: str) -> pd.Series | None:
    """Extrae la serie Close de un dict edge_extra, o None si falta."""
    if ee and key in ee and not ee[key].empty and 'Close' in ee[key].columns:
        return ee[key]['Close']
    return None


@st.cache_data(show_spinner="Cargando datos…", ttl=cfg.CACHE_TTL["har_rv"])   # 1h — el régimen se mueve a escala diaria
def compute_regime_cached(spy_close: pd.Series) -> dict:
    """
    HMM de régimen sobre retornos log del SPY (probabilidades filtradas).
    Limitado a ~20 años (5000 obs): suficiente para estimar los 3 estados
    (incluye 2008, 2018, 2020, 2022) y mantiene el fit en ~3-4s.
    """
    rets = np.log(spy_close / spy_close.shift(1)).dropna().tail(5000)
    return fit_volatility_regime(rets)


REGIME_DISPLAY = {
    "calm":       ("CALMA",      "var(--g)", "Cosecha de prima — short vol funciona"),
    "transition": ("TRANSICIÓN", "var(--y)", "Whipsaws — reducir tamaño, no agregar riesgo"),
    "panic":      ("PÁNICO",     "var(--r)", "Clusters de vol — short vol muere aquí; hedge/cash"),
}

_LIGHT_DOT = {"green": "", "yellow": "", "red": ""}


def build_regime_probs_chart(probs: pd.DataFrame, days: int = 504) -> go.Figure:
    """Área apilada de probabilidades filtradas calm/transition/panic."""
    p = probs.tail(days)
    fig = go.Figure()
    colors = {"calm": "rgba(22,199,132,0.75)",
              "transition": "rgba(245,166,35,0.75)",
              "panic": "rgba(234,57,67,0.80)"}
    names = {"calm": "Calma", "transition": "Transición", "panic": "Pánico"}
    for col in ["calm", "transition", "panic"]:
        if col not in p.columns:
            continue
        fig.add_trace(go.Scatter(
            x=p.index, y=p[col], name=names[col],
            mode="lines", stackgroup="one", line=dict(width=0.5),
            fillcolor=colors[col],
            hovertemplate=f"{names[col]}: %{{y:.0%}}<extra></extra>"))
    fig.update_layout(
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=50, r=20, t=30, b=35),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#1C1F24",
                   title=dict(text="P(régimen)", font=dict(size=10, color="#9AA1A9"))),
        xaxis=dict(gridcolor="#1C1F24"),
        title=dict(text="<b>Probabilidad filtrada de régimen (sin look-ahead)</b>",
                   font=dict(size=12, color="#9AA1A9"), x=0.01))
    return fig


def build_vrp_tracker_chart(vrp_df: pd.DataFrame, days: int = 756) -> go.Figure:
    """VRP (vol points) con zona negativa resaltada + percentil rolling.
    (Renombrada: sombreaba a build_vrp_chart(ebt), el chart VRP/HAR original.)"""
    p = vrp_df.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=("<b>VRP = VIX − RV20 (vol points)</b>",
                                        "<b>Percentil rolling del VRP en varianza</b>"))
    vrp_clr = ['#16C784' if v >= 0 else '#EA3943' for v in p['vrp_vol']]
    fig.add_trace(go.Bar(x=p.index, y=p['vrp_vol'], marker_color=vrp_clr,
                         opacity=0.75, name='VRP', showlegend=False,
                         hovertemplate='%{x|%Y-%m-%d}<br>VRP: %{y:+.2f} pts<extra></extra>'),
                  row=1, col=1)
    fig.add_hline(y=0, line_color='#3A3F47', line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(x=p.index, y=p['vrp_pct'], mode='lines',
                             line=dict(color='#9AA1A9', width=1.8),
                             name='Percentil', showlegend=False,
                             hovertemplate='%{x|%Y-%m-%d}<br>Percentil: %{y:.0f}<extra></extra>'),
                  row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(234,57,67,0.08)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor='rgba(22,199,132,0.08)', line_width=0, row=2, col=1)

    fig.update_layout(
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=55, r=20, t=40, b=35),
        hovermode='x unified', bargap=0)
    for ann in fig['layout']['annotations'][:2]:
        ann['font'] = dict(size=11, color='#9AA1A9', family='Inter')
        ann['xanchor'] = 'left'; ann['x'] = 0.01
    fig.update_xaxes(gridcolor='#1C1F24',
                     tickfont=dict(size=9.5, color='#9AA1A9', family='JetBrains Mono'))
    fig.update_yaxes(gridcolor='#1C1F24',
                     tickfont=dict(size=9.5, color='#9AA1A9', family='JetBrains Mono'))
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIX INVERSE — gráficos del panel de seguimiento (modelo congelado)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━







def build_vts_barometer_gauge(score: float, regime: str,
                               date_str: str = "") -> go.Figure:
    """
    Gauge idéntico al VTS Volatility Barometer original.
    Semicírculo con colormap verde→amarillo→rojo y aguja negra.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            suffix="%",
            font=dict(size=40, color='#F4F5F6', family='Inter'),
            valueformat='.2f',
        ),
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dict(
            text=f"<span style='font-size:0.85rem;color:#9AA1A9;font-family:JetBrains Mono'>"
                 f"{date_str}</span><br>"
                 f"<b style='font-size:1.1rem;color:#F4F5F6;font-family:Inter'>"
                 f"VTS Volatility Barometer</b>",
            font=dict(color='#F4F5F6'),
        ),
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': '#9AA1A9',
                'tickfont': dict(size=11, color='#9AA1A9', family='JetBrains Mono'),
                'tickmode': 'array',
                'tickvals': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'ticktext': ['0%','10%','20%','30%','40%','50%',
                             '60%','70%','80%','90%','100%'],
            },
            'bar': {'color': 'rgba(0,0,0,0)', 'thickness': 0.0},
            # Sin borde grueso — el fondo del gauge es limpio
            'bgcolor': '#0A0B0D',
            'borderwidth': 0,
            'bordercolor': '#0A0B0D',
            'steps': [
                {'range': [0, 20],   'color': '#16C784'},    # verde fuerte
                {'range': [20, 35],  'color': '#16C784'},    # verde
                {'range': [35, 50],  'color': '#16C784'},    # verde claro
                {'range': [50, 65],  'color': '#F5A623'},    # amarillo
                {'range': [65, 80],  'color': '#F5A623'},    # naranja
                {'range': [80, 90],  'color': '#EA3943'},    # rojo
                {'range': [90, 100], 'color': '#EA3943'},    # rojo oscuro
            ],
            'threshold': {
                'line': {'color': '#0A0B0D', 'width': 8},
                'thickness': 0.85,
                'value': score,
            },
        },
    ))

    # Etiqueta de régimen debajo
    if   score < 20: clr = '#16C784'
    elif score < 40: clr = '#16C784'
    elif score < 60: clr = '#F5A623'
    elif score < 80: clr = '#F5A623'
    else:            clr = '#EA3943'

    fig.add_annotation(
        x=0.5, y=-0.05, xref='paper', yref='paper',
        text=f"<b style='font-size:1.3rem;color:{clr};font-family:Inter'>{regime}</b>",
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#F4F5F6'),
        height=420,
        margin=dict(l=30, r=30, t=80, b=60),
    )
    return fig


def build_vts_metrics_table(metrics: list, sort_mode: str = 'vts') -> go.Figure:
    """
    Tabla horizontal de cada métrica con su percentil como barra de progreso.
    Similar al desglose que justifica el score del gauge VTS original.

    sort_mode:
      'vts'    → orden canónico VTS (como en la imagen oficial del gauge)
      'stress' → ordenado por percentil descendente (más estrés arriba)
    """
    if not metrics:
        return go.Figure()

    # Orden canónico VTS (según imagen oficial del gauge)
    VTS_ORDER = [
        'M1:M2 VIX FUTURES',
        'M4-M7 VIX FUTURES',
        'VX30:VIX ROLL YIELD',
        'VIX / VVIX / VOLI',
        'CASH VIX OSCILLATOR',
        'VIX - VOLI RESIDUAL',
        'TRADERS VRP',
        'SDEX / TDEX / SKEW',
        'VIX:VIX3M MEDIUM',
        'EXTREME PUT/CALL RATIO',
        'VIX9D:VIX FAST',
        'VOLPOCALYPSE THRESHOLD',
        'VIX LEVEL (core)',
    ]

    if sort_mode == 'vts':
        # Ordenar según VTS_ORDER, con fallback al final para las no conocidas
        order_map = {name: i for i, name in enumerate(VTS_ORDER)}
        sorted_m = sorted(metrics, key=lambda m: order_map.get(m['name'], 999))
    else:  # stress
        sorted_m = sorted(metrics,
                          key=lambda m: m['percentile'] if pd.notna(m['percentile']) else -1,
                          reverse=True)

    n = len(sorted_m)
    names  = [m['name'] for m in sorted_m]
    # Reemplazar NaN por 0 para que Plotly lo renderice (pero flaggeamos en el texto)
    pctls  = [m['percentile'] if pd.notna(m['percentile']) else 0 for m in sorted_m]
    is_nan = [pd.isna(m['percentile']) for m in sorted_m]
    vals   = [m['value']  for m in sorted_m]
    wts    = [m['weight'] for m in sorted_m]

    # Color por bucket
    def bucket_color(p, is_nan_flag):
        if is_nan_flag: return '#3A3F47'
        if p < 20:  return '#16C784'
        if p < 40:  return '#16C784'
        if p < 60:  return '#F5A623'
        if p < 80:  return '#F5A623'
        return '#EA3943'

    bar_colors = [bucket_color(p, nn) for p, nn in zip(pctls, is_nan)]

    fig = go.Figure()

    # Fondo: barra gris hasta 100
    fig.add_trace(go.Bar(
        x=[100] * n, y=names, orientation='h',
        marker=dict(color='#15171A', line=dict(width=0)),
        showlegend=False, hoverinfo='skip',
        width=0.65,
    ))
    # Barra de percentil real
    fig.add_trace(go.Bar(
        x=pctls, y=names, orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"N/A · {v} · w={w:.1f}" if nn else f"{p:.0f}% · {v} · w={w:.1f}"
              for p, v, w, nn in zip(pctls, vals, wts, is_nan)],
        textposition='inside', insidetextanchor='start',
        textfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Percentil: %{x:.1f}%<extra></extra>',
        width=0.65,
    ))

    title_sub = ("Orden canónico VTS (como en el gauge oficial)"
                 if sort_mode == 'vts'
                 else "Ordenado por nivel de stress (mayor percentil arriba)")

    fig.update_layout(
        title=dict(
            text=f"<b>Desglose del Barómetro — Percentil rolling por métrica</b>"
                 f"<br><span style='font-size:0.7rem;color:#9AA1A9;font-family:JetBrains Mono'>"
                 f"{title_sub} · Verde = vol baja · Rojo = vol alta"
                 f"</span>",
            font=dict(size=13, color='#F4F5F6', family='Inter'), x=0.5, xanchor='center',
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode='overlay',
        height=max(420, n * 36 + 110),
        margin=dict(l=180, r=30, t=80, b=40),
        xaxis=dict(
            range=[0, 100], showgrid=True, gridcolor='#1C1F24',
            tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
        ),
        yaxis=dict(
            tickfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
            autorange='reversed',
        ),
        showlegend=False,
    )

    # Línea vertical en 50%
    fig.add_vline(x=50, line_color='#262A30', line_dash='dot', line_width=1)

    return fig


def build_vts_history_chart(history: pd.Series, window: int = 252) -> go.Figure:
    """
    Timeline del score del barómetro con bandas de color para cada régimen.
    """
    fig = go.Figure()

    if history is None or history.empty:
        return fig

    # Filtrar NaN antes de procesar
    h_clean = history.dropna()
    if h_clean.empty:
        return fig

    # Mostrar solo el último año para claridad
    h = h_clean.tail(window)
    if h.empty:
        return fig

    # Bandas de régimen (horizontales)
    for y0, y1, color, label in [
        (0, 20,   'rgba(22,199,132,0.10)',   'Vol Baja'),
        (20, 40,  'rgba(22,199,132,0.08)',   'Moderada'),
        (40, 60,  'rgba(255,211,61,0.08)',  'Mid'),
        (60, 80,  'rgba(251,133,0,0.08)',   'Elevada'),
        (80, 100, 'rgba(234,57,67,0.10)',   'Extrema'),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer='below')

    # Línea del score
    fig.add_trace(go.Scatter(
        x=h.index, y=h.values,
        mode='lines', name='Barómetro VTS',
        line=dict(color='#F4F5F6', width=2.2),
        fill='tozeroy', fillcolor='rgba(244,245,246,0.06)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}%<extra></extra>',
    ))

    # Punto actual
    fig.add_trace(go.Scatter(
        x=[h.index[-1]], y=[h.iloc[-1]],
        mode='markers', name='HOY',
        marker=dict(size=14, color='#F5A623', symbol='diamond',
                    line=dict(width=2, color='white')),
        showlegend=False,
    ))

    # Línea de media histórica
    mean_val = h.mean()
    fig.add_hline(y=mean_val, line_color='#F5A623', line_dash='dash',
                  line_width=1, annotation_text=f"Media: {mean_val:.1f}%",
                  annotation_position='top right',
                  annotation_font=dict(size=9, color='#F5A623'))

    fig.update_layout(
        title=dict(
            text=f"<b>Histórico del Barómetro — últimos {len(h)} días de trading</b>",
            font=dict(size=13, color='#F4F5F6', family='Inter'),
            x=0.5, xanchor='center',
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(gridcolor='#1C1F24',
                   tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono')),
        yaxis=dict(
            range=[0, 100], gridcolor='#1C1F24',
            tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
            title=dict(text="Score %", font=dict(size=10, color='#9AA1A9')),
            tickvals=[0, 20, 40, 60, 80, 100],
        ),
        hovermode='x unified', showlegend=False,
    )
    return fig


def build_vts_history_full(history: pd.Series) -> go.Figure:
    """
    Gráfico histórico completo del Barómetro (lifetime).
    Incluye rangeselector, bandas de régimen y estadísticas.
    """
    fig = go.Figure()
    if history is None or history.empty:
        return fig
    h = history.dropna()
    if h.empty:
        return fig

    # Bandas de régimen (horizontales)
    for y0, y1, color, _ in [
        (0, 20,   'rgba(22,199,132,0.10)',   'Vol Baja'),
        (20, 40,  'rgba(22,199,132,0.08)',   'Moderada'),
        (40, 60,  'rgba(255,211,61,0.08)',  'Mid'),
        (60, 80,  'rgba(251,133,0,0.08)',   'Elevada'),
        (80, 100, 'rgba(234,57,67,0.10)',   'Extrema'),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer='below')

    # Serie principal
    fig.add_trace(go.Scatter(
        x=h.index, y=h.values,
        mode='lines', name='Barómetro VTS',
        line=dict(color='#F4F5F6', width=1.5),
        fill='tozeroy', fillcolor='rgba(244,245,246,0.04)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}%<extra></extra>',
    ))

    # Punto HOY
    fig.add_trace(go.Scatter(
        x=[h.index[-1]], y=[h.iloc[-1]],
        mode='markers', name='HOY',
        marker=dict(size=14, color='#F5A623', symbol='diamond',
                    line=dict(width=2, color='white')),
        showlegend=False,
    ))

    # Media histórica
    mean_val = h.mean()
    fig.add_hline(y=mean_val, line_color='#F5A623', line_dash='dash',
                  line_width=1,
                  annotation_text=f"Media: {mean_val:.1f}%",
                  annotation_position='top right',
                  annotation_font=dict(size=9, color='#F5A623'))

    fig.update_layout(
        title=dict(
            text=f"<b>Histórico completo del Barómetro VTS</b>"
                 f"<br><span style='font-size:0.72rem;color:#9AA1A9;font-family:JetBrains Mono'>"
                 f"{h.index[0].date()} → {h.index[-1].date()} · "
                 f"{len(h):,} días · Media: {mean_val:.1f}% · "
                 f"Min: {h.min():.1f}% · Max: {h.max():.1f}%"
                 f"</span>",
            font=dict(size=13, color='#F4F5F6', family='Inter'),
            x=0.5, xanchor='center',
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=480, margin=dict(l=50, r=30, t=85, b=45),
        xaxis=dict(
            gridcolor='#1C1F24',
            tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1A",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3A",  step="year",  stepmode="backward"),
                    dict(count=5,  label="5A",  step="year",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ],
                bgcolor='#15171A', activecolor='#F5A623', bordercolor='#262A30',
                font=dict(size=9, color='#F4F5F6', family='JetBrains Mono'),
            ),
            rangeslider=dict(visible=True, thickness=0.04, bgcolor='#15171A'),
        ),
        yaxis=dict(
            range=[0, 100], gridcolor='#1C1F24',
            tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
            title=dict(text="Score %", font=dict(size=10, color='#9AA1A9')),
            tickvals=[0, 20, 40, 60, 80, 100],
        ),
        hovermode='x unified', showlegend=False,
    )
    return fig


def build_vts_monthly_heatmap(history: pd.Series) -> go.Figure:
    """
    Heatmap de la media mensual del score VTS.
    Año × Mes — detecta patrones estacionales (sell-in-May, etc).
    """
    fig = go.Figure()
    if history is None or history.empty:
        return fig

    h = history.dropna()
    if len(h) < 60:
        return fig

    # Agrupar por año/mes y tomar la media mensual
    df_m = pd.DataFrame({'score': h.values}, index=h.index)
    df_m['year']  = df_m.index.year
    df_m['month'] = df_m.index.month
    monthly = df_m.groupby(['year', 'month'])['score'].mean().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='score')

    # Asegurar todos los meses 1-12
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Escala de colores idéntica a las bandas del barómetro
    z_vals = pivot.values
    text_vals = [[f"{v:.0f}%" if pd.notna(v) else "" for v in row] for row in z_vals]

    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
        y=pivot.index.tolist(),
        text=text_vals,
        texttemplate='%{text}',
        textfont=dict(size=10, color='#0A0B0D', family='JetBrains Mono'),
        colorscale=[
            [0.00, '#16C784'],
            [0.20, '#16C784'],
            [0.40, '#F5A623'],
            [0.60, '#F5A623'],
            [0.80, '#EA3943'],
            [1.00, '#EA3943'],
        ],
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="Score %", font=dict(color='#9AA1A9', size=10)),
            tickfont=dict(size=9, color='#9AA1A9'),
            thickness=12,
        ),
        hovertemplate='<b>%{y} · %{x}</b><br>Score medio: %{z:.1f}%<extra></extra>',
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Heatmap mensual del Barómetro — media por mes/año</b>"
                 "<br><span style='font-size:0.72rem;color:#9AA1A9;font-family:JetBrains Mono'>"
                 "Detecta patrones estacionales (ej: 'sell in May', vol de septiembre-octubre)"
                 "</span>",
            font=dict(size=13, color='#F4F5F6', family='Inter'),
            x=0.5, xanchor='center',
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(340, len(pivot) * 24 + 120),
        margin=dict(l=60, r=30, t=85, b=40),
        xaxis=dict(tickfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
                   side='top'),
        yaxis=dict(tickfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
                   autorange='reversed', dtick=1),
    )
    return fig


def build_vts_regime_distribution(history: pd.Series) -> tuple[go.Figure, dict]:
    """
    Distribución del tiempo pasado en cada régimen + histograma del score.
    Retorna (figura, stats dict).
    """
    fig = go.Figure()
    stats = {}

    if history is None or history.empty:
        return fig, stats

    h = history.dropna()
    if h.empty:
        return fig, stats

    # Buckets de régimen
    buckets = [
        (0,   20,  'Vol BAJA',    '#16C784'),
        (20,  40,  'MODERADA',    '#16C784'),
        (40,  60,  'MID',         '#F5A623'),
        (60,  80,  'ELEVADA',     '#F5A623'),
        (80,  100, 'EXTREMA',     '#EA3943'),
    ]

    total = len(h)
    counts = []
    labels = []
    colors = []
    for lo, hi, label, color in buckets:
        mask = (h >= lo) & (h < hi if hi < 100 else h <= hi)
        n = int(mask.sum())
        pct = n / total * 100 if total > 0 else 0
        counts.append(pct)
        labels.append(f"{label}<br>({lo}-{hi}%)")
        colors.append(color)
        stats[label] = {'days': n, 'pct': pct}

    # Barras horizontales de tiempo en cada régimen
    fig.add_trace(go.Bar(
        x=counts, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c:.1f}% del tiempo" for c in counts],
        textposition='outside',
        textfont=dict(size=11, color='#F4F5F6', family='JetBrains Mono'),
        hovertemplate='%{y}<br>%{x:.2f}%<extra></extra>',
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Distribución del tiempo por régimen</b>"
                 f"<br><span style='font-size:0.72rem;color:#9AA1A9;font-family:JetBrains Mono'>"
                 f"{total:,} días · {h.index[0].date()} → {h.index[-1].date()}"
                 f"</span>",
            font=dict(size=13, color='#F4F5F6', family='Inter'),
            x=0.5, xanchor='center',
        ),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=130, r=80, t=80, b=40),
        xaxis=dict(
            range=[0, max(counts) * 1.2],
            gridcolor='#1C1F24', showgrid=True,
            tickfont=dict(size=9, color='#9AA1A9', family='JetBrains Mono'),
            ticksuffix='%',
        ),
        yaxis=dict(
            tickfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
            autorange='reversed',
        ),
    )
    return fig, stats


def cpct(p1, p2):
    if p1 and p2 and p1 > 0:
        return round((p2 - p1) / p1 * 100, 2)
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_term_chart(vix_spot, df_vx, show_prev=True):
    """VIXCentral-faithful term structure chart using scraped CBOE data."""
    fig = go.Figure()
    if df_vx.empty:
        return fig

    # Month labels from expiration
    labels = []
    for _, r in df_vx.iterrows():
        exp = r.get('Expiration')
        if pd.notna(exp):
            labels.append(MN.get(exp.month, str(exp.month)[:3]))
        else:
            labels.append(str(r.get('Symbol','')))

    xpos = list(range(len(df_vx)))
    prices = df_vx['Price'].tolist()

    # Previous close = Price - Change
    prev_prices = []
    for _, r in df_vx.iterrows():
        p = r['Price']
        c = r.get('Change', 0)
        if pd.notna(p) and p > 0 and pd.notna(c):
            prev_prices.append(round(p - c, 4))
        else:
            prev_prices.append(None)

    # Today's curve
    vx = [x for x, y in zip(xpos, prices) if pd.notna(y) and y > 0]
    vy = [y for y in prices if pd.notna(y) and y > 0]

    if vy:
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode='lines+markers+text',
            name='Último', line=dict(color='#F5A623', width=3, shape='spline'),
            marker=dict(size=9, color='#F5A623', line=dict(width=2, color='#0A0B0D')),
            text=[f"{v:.3f}" for v in vy],
            textposition='top center',
            textfont=dict(size=10, color='#F4F5F6', family='JetBrains Mono'),
            hovertemplate='%{text}<extra></extra>',
        ))

    # Previous day
    if show_prev:
        pvx = [x for x, y in zip(xpos, prev_prices) if y and y > 0]
        pvy = [y for y in prev_prices if y and y > 0]
        if len(pvy) >= 2:
            fig.add_trace(go.Scatter(
                x=pvx, y=pvy, mode='lines+markers',
                name='Cierre anterior',
                line=dict(color='#9AA1A9', width=1.5, dash='dot', shape='spline'),
                marker=dict(size=5, color='#9AA1A9', symbol='diamond'),
                hovertemplate='Prev: %{y:.3f}<extra></extra>',
            ))

    # VIX Index dashed line
    if vix_spot:
        fig.add_hline(y=vix_spot['price'], line_dash="dash", line_color="#9AA1A9", line_width=1.5,
                      annotation_text=f"  {vix_spot['price']:.2f}",
                      annotation_position="right",
                      annotation_font=dict(size=11, color="#9AA1A9", family="JetBrains Mono"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='VIX contado',
                                 line=dict(color='#9AA1A9', width=1.5, dash='dash'), showlegend=True))

    all_y = vy + ([vix_spot['price']] if vix_spot else [])
    y_min = min(all_y) - 1.5 if all_y else 15
    y_max = max(all_y) + 1.5 if all_y else 30

    fig.update_layout(
        title=dict(
            text="Curva de futuros del VIX",
            font=dict(size=15, color='#F4F5F6', family="'Space Grotesk', Inter"), x=0, xanchor="left"),
        template="stc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=50, r=30, t=65, b=50),
        xaxis=dict(tickvals=xpos, ticktext=labels,
                   tickfont=dict(size=11, color='#9AA1A9', family='JetBrains Mono'),
                   gridcolor='#1C1F24', showline=True, linecolor='#262A30',
                   title=dict(text="Vencimiento", font=dict(size=11, color='#9AA1A9', family='Inter'))),
        yaxis=dict(range=[y_min, y_max],
                   title=dict(text="Volatilidad", font=dict(size=11, color='#9AA1A9', family='Inter')),
                   tickfont=dict(size=11, color='#9AA1A9', family='JetBrains Mono'),
                   gridcolor='#1C1F24', showline=True, linecolor='#262A30'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    font=dict(size=10, color='#F4F5F6', family='JetBrains Mono')),
        hoverlabel=dict(bgcolor='#15171A', bordercolor='#262A30',
                        font=dict(size=11, family='JetBrains Mono', color='#F4F5F6')),
        hovermode='x unified',
    )
    return fig





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATOS GLOBALES (ligeros: un JSON de CBOE y dos cotizaciones)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_vx_full = fetch_cboe_curve()
df_vx = df_vx_full
vix_spot = fetch_vix_spot()
vix3m_live = fetch_index_live("VIX3M")
etps = fetch_etps()

m1p = df_vx['Price'].iloc[0] if not df_vx.empty and pd.notna(df_vx['Price'].iloc[0]) else None
m2p = df_vx['Price'].iloc[1] if len(df_vx) > 1 and pd.notna(df_vx['Price'].iloc[1]) else None
front_ct = cpct(m1p, m2p)
r_m2m1_live = (m2p / m1p) if (m1p and m2p) else None
r_vix3m_live = (vix3m_live["price"] / vix_spot["price"]) if (vix3m_live and vix_spot) else None


def fv(v):
    return f"{v:.2f}" if v is not None and pd.notna(v) and v != 0 else "—"


def vc(v):
    if v is None:
        return "nt"
    return "up" if v >= 0 else "dn"


def fp(v):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}%"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CABECERA DE MARCA + TIRA DE ESTADO (comunes a todas las páginas)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_now = now_cdmx()
_et = datetime.now(cfg.NYSE_TZ)
_mins = _et.hour * 60 + _et.minute
_mkt_open = _alerts.es_sesion(_et.date()) and 9 * 60 + 30 <= _mins < 16 * 60
st.markdown(f"""<div class="stc-head">
  <div class="stc-lockup">{T.icon(46)}<div class="stc-word"><b>SPREAD</b><span>TRADING CLUB</span></div></div>
  <div class="stc-product"><b>VIX CONTROLLER</b><span>Volatilidad y estrategia VIX Inverse</span></div>
  <div class="stc-meta"><span class="dot {'on' if _mkt_open else 'off'}"></span><span class="{'on' if _mkt_open else 'off'}">
    {'MERCADO ABIERTO' if _mkt_open else 'MERCADO CERRADO'}</span><br>
    {_now.strftime('%d/%m/%Y %H:%M')} CDMX · CBOE con ~15 min de retraso</div>
</div>""", unsafe_allow_html=True)


def _sb_item(label, value, cls="", hl=False):
    return (f'<div class="sb-item{" hl" if hl else ""}"><div class="sb-l">{label}</div>'
            f'<div class="sb-v {cls}">{value}</div></div>')


_sb = []
if vix_spot:
    _chg = vix_spot.get("chg", 0) or 0
    _sb.append(_sb_item("VIX", f"{vix_spot['price']:.2f}"))
    _sb.append(_sb_item("Δ día", f"{'▲' if _chg > 0 else '▼'} {_chg:+.2f}", "dn" if _chg > 0 else "up"))
if m1p is not None:
    _sb.append(_sb_item("VX M1", f"{m1p:.2f}"))
if front_ct is not None:
    _sb.append(_sb_item("Contango M1→M2" if front_ct > 0 else "Backwardation",
                        f"{'▲' if front_ct > 0 else '▼'} {front_ct:+.2f}%", "up" if front_ct > 0 else "dn"))
if r_vix3m_live is not None:
    _sb.append(_sb_item("VIX3M/VIX", f"{'▲' if r_vix3m_live > 1 else '▼'} {r_vix3m_live:.3f}",
                        "up" if r_vix3m_live > 1 else "dn"))
for _sym in ("VXX", "SVXY"):
    _e = (etps or {}).get(_sym)
    if _e and _e.get("close") and _e.get("prev"):
        _d = (_e["close"] / _e["prev"] - 1) * 100
        _sb.append(_sb_item(_sym, f"{_e['close']:.2f} {'▲' if _d >= 0 else '▼'}{abs(_d):.1f}%",
                            "up" if _d >= 0 else "dn"))
try:
    _vh = strategy_page.load_history(strategy_page._mtime(strategy_page.HIST_PATH))
    _ve = _vl.estado(_vh, strategy_page.load_curve(strategy_page._mtime(strategy_page.CURVE_PATH)))
    _sb.append(_sb_item("VIX Inverse", "DENTRO" if _ve["dentro"] else "FUERA",
                        "am" if _ve["dentro"] else "", hl=True))
except Exception as _e_vinv:                      # noqa: BLE001
    logging.getLogger("vix_controller").warning("estado VIX Inverse: %s", _e_vinv)
if _sb:
    st.markdown(f'<div class="statusbar">{"".join(_sb)}</div>', unsafe_allow_html=True)
if st.session_state.get("curve_error"):
    st.markdown(f'<div class="stc-alert loss"><b>Curva de CBOE no disponible</b>'
                f'{st.session_state["curve_error"]}. Se muestran los datos que sí cargaron.</div>',
                unsafe_allow_html=True)


def page_curva():
    st.markdown(f"""{T.eyebrow("Curva de futuros")}
        <div class="stc-h1">Estructura temporal del VIX</div>
        <p class="stc-lead">La curva en vivo de CBOE y lo que dice sobre el carry: contango, base frente
        al contado, roll yield del ETP y su contexto histórico.</p>""", unsafe_allow_html=True)
    _c1, _c2, _c3, _c4 = st.columns([1.2, 1, 1, 1])
    with _c1:
        N_MONTHS = st.segmented_control("Meses", options=[4, 6, 8], default=8, key="curva_meses") or 8
    with _c2:
        SHOW_PREV = st.toggle("Cierre anterior", value=True, key="curva_prev")
    with _c3:
        SHOW_TABLE = st.toggle("Tabla de contratos", value=True, key="curva_tabla")
    with _c4:
        if st.button("Actualizar CBOE", key="btn_refresh_cboe", width="stretch"):
            fetch_cboe_curve.clear()
            fetch_index_live.clear()
            fetch_vix_spot.clear()
            fetch_etps.clear()
            st.rerun()
    df_vx = df_vx_full.head(N_MONTHS).reset_index(drop=True)

    vix_p = vix_spot['price'] if vix_spot else None

    # Metrics
    last_price_col = df_vx['Price'].tolist() if not df_vx.empty else []
    total_ct = cpct(vix_p, last_price_col[-1]) if vix_p and last_price_col else None
    spot_m1 = cpct(vix_p, m1p)

    m1_lbl = ""
    m1_dte = "?"
    m2_lbl = ""
    if not df_vx.empty:
        exp1 = df_vx['Expiration'].iloc[0]
        if pd.notna(exp1):
            m1_lbl = MN.get(exp1.month, "")
            m1_dte = df_vx['DTE'].iloc[0] if 'DTE' in df_vx.columns else "?"
        if len(df_vx) > 1:
            exp2 = df_vx['Expiration'].iloc[1]
            if pd.notna(exp2):
                m2_lbl = MN.get(exp2.month, "")

    st.markdown(f"""
    <div class="mrow">
        <div class="mpill"><div class="ml">VIX Index</div><div class="mv nt">{fv(vix_p)}</div></div>
        <div class="mpill"><div class="ml">M1 · {m1_lbl} · {m1_dte} DTE</div><div class="mv nt">{fv(m1p)}</div></div>
        <div class="mpill"><div class="ml">M2 · {m2_lbl}</div><div class="mv nt">{fv(m2p)}</div></div>
        <div class="mpill"><div class="ml">VIX → M1</div><div class="mv {vc(spot_m1)}">{fp(spot_m1)}</div></div>
        <div class="mpill"><div class="ml">M1 → M2 Contango</div><div class="mv {vc(front_ct)}">{fp(front_ct)}</div></div>
        <div class="mpill"><div class="ml">Total Curve</div><div class="mv {vc(total_ct)}">{fp(total_ct)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # PANEL DE DECISIÓN — ¿la curva favorece LONG VOL o SHORT VOL?
    # Traduce la term structure a métricas operables:
    #   · Percentil histórico del contango (¿el carry de hoy es rico o pobre?)
    #   · Roll yield anualizado del M1 (el carry que cobra el short / paga el long)
    #   · Basis VIX→M1 (descuento = el mercado espera vol YA)
    #   · Checklist transparente → señal compuesta
    # ═══════════════════════════════════════════════════════════
    _dfh = load_master_parquet()
    _vxh = load_vx_curve_parquet()          # histórico diario de CBOE (M1..M8)

    # Percentil del contango M1→M2 vs últimos 5 años. Fuente preferente: el
    # histórico diario de futuros de CBOE; si aún no existe, master.parquet.
    ct_pctile = None
    _hist_ct = pd.Series(dtype=float)
    if not _vxh.empty and 'contango_pct' in _vxh.columns:
        _hist_ct = _vxh['contango_pct'].dropna().tail(tsig.PCTILE_WINDOW)
    elif not _dfh.empty and 'Contango_pct' in _dfh.columns:
        _hist_ct = _dfh['Contango_pct'].dropna().tail(tsig.PCTILE_WINDOW)
    if front_ct is not None and len(_hist_ct) > tsig.PCTILE_MIN_OBS:
        ct_pctile = float((_hist_ct < front_ct).mean() * 100)

    # Roll yield anualizado del ETP: VXX/SVXY rollean DIARIAMENTE de M1 a M2,
    # así que su carry es el contango M1→M2 normalizado por los días entre
    # vencimientos (Simon & Campasano 2014). No usamos (M1−VIX)/DTE_M1
    # porque con DTE chico la anualización explota (+300%/año a 6 días
    # de expiración) y no representa el carry real del vehículo.
    roll_ann = None
    try:
        _dte1 = float(m1_dte)
        _dte2 = float(df_vx['DTE'].iloc[1]) if len(df_vx) > 1 else None
        if front_ct is not None and _dte2 and _dte2 > _dte1:
            roll_ann = front_ct * (365.0 / (_dte2 - _dte1))
    except (TypeError, ValueError, KeyError, IndexError):
        pass

    # ── Checklist → señal compuesta (lógica en quant/term_structure.py, la
    #    misma que se reproduce sobre el histórico) ─────────────────────
    try:
        _dte1_live = float(m1_dte)
    except (TypeError, ValueError):
        _dte1_live = None
    _cs = tsig.curve_signal(front_ct, spot_m1, ct_pctile, vix_p, dte1=_dte1_live)
    checks, n_ok, ts_sig, ts_desc = _cs["checks"], _cs["n_ok"], _cs["signal"], _cs["desc"]
    ts_clr = {tsig.SIGNAL_LONG: "var(--r)", tsig.SIGNAL_SHORT: "var(--g)",
              tsig.SIGNAL_NEUTRAL: "var(--y)"}[ts_sig]

    chk_html = "".join(
        f'<div class="chk"><span class="{"ok" if ok else "no"}">'
        f'{"✓" if ok else "✗"}</span> {name} '
        f'<span style="margin-left:auto;color:var(--dim)">{val}</span></div>'
        for name, ok, val in checks)

    roll_html = ""
    if roll_ann is not None:
        _rc = "var(--g)" if roll_ann > 0 else "var(--r)"
        roll_html = (f'<div class="ic-row"><span class="ic-label">Roll yield del ETP '
                     f'(M1→M2 anualizado) — el carry que decae VXX</span>'
                     f'<span class="ic-val" style="color:{_rc};font-weight:700">'
                     f'{roll_ann:+.1f}%/año</span></div>')

    pct_html = ""
    if ct_pctile is not None:
        _pc = "var(--g)" if 25 <= ct_pctile <= 90 else "var(--y)"
        pct_html = (f'<div class="ic-row"><span class="ic-label">Percentil del contango '
                    f'vs 5 años</span><span class="ic-val" style="color:{_pc}">'
                    f'p{ct_pctile:.0f}</span></div>')

    col_sig, col_chk = st.columns([1, 1.6])
    with col_sig:
        st.markdown(f"""<div class="icard" style="border-left:3px solid {ts_clr}">
            <div class="ic-title">Lectura de la curva</div>
            <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:1.7rem;
                        color:{ts_clr};text-align:center;padding:0.3rem 0">{ts_sig}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                        color:var(--dim);line-height:1.5">{ts_desc}</div>
        </div>""", unsafe_allow_html=True)
    with col_chk:
        st.markdown(f"""<div class="icard">
            <div class="ic-title">Checklist de curva ({n_ok}/{len(checks)})</div>
            {chk_html}{roll_html}{pct_html}
        </div>""", unsafe_allow_html=True)

    # ── Curva de índices spot (VIX9D → VIX1Y) ──────────────────
    # Complementa los futuros: las inversiones VIX9D/VIX aparecen ANTES
    # de que la curva de futuros se invierta.
    _ee_ts = fetch_edge_extra()
    _spot_curve = []
    for _k in ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"):
        _s = _ee_ts.get(_k)
        if _k == "VIX" and vix_p:
            _spot_curve.append((_k, float(vix_p)))
        elif _s is not None and not _s.empty and 'Close' in _s.columns \
                and _s['Close'].notna().any():
            _spot_curve.append((_k, float(_s['Close'].dropna().iloc[-1])))
    if len(_spot_curve) >= 3:
        _vals = dict(_spot_curve)
        _r9d = _vals.get("VIX9D", np.nan) / _vals.get("VIX", np.nan) \
            if _vals.get("VIX") else np.nan
        _r3m = _vals.get("VIX", np.nan) / _vals.get("VIX3M", np.nan) \
            if _vals.get("VIX3M") else np.nan
        _pills = "".join(
            f'<div class="mpill"><div class="ml">{k}</div>'
            f'<div class="mv nt" style="font-size:1.05rem">{v:.2f}</div></div>'
            for k, v in _spot_curve)
        _c9 = "dn" if pd.notna(_r9d) and _r9d > 1 else "up"
        _c3 = "dn" if pd.notna(_r3m) and _r3m > 1 else "up"
        _pills += (f'<div class="mpill"><div class="ml">VIX9D/VIX · >1 = stress</div>'
                   f'<div class="mv {_c9}" style="font-size:1.05rem">{_r9d:.3f}</div></div>'
                   if pd.notna(_r9d) else "")
        _pills += (f'<div class="mpill"><div class="ml">VIX/VIX3M · >1 = inversión</div>'
                   f'<div class="mv {_c3}" style="font-size:1.05rem">{_r3m:.3f}</div></div>'
                   if pd.notna(_r3m) else "")
        st.markdown(f'<div class="mrow">{_pills}</div>', unsafe_allow_html=True)

    # Chart
    fig = build_term_chart(vix_spot, df_vx, show_prev=SHOW_PREV)
    st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=True, displaylogo=False))

    # ── Histórico de la señal de curva (histórico diario de futuros CBOE) ──
    if not _vxh.empty and {"m1", "m2", "VIX"} <= set(_vxh.columns):
        try:
            _sh = tsig.signal_history(_vxh)
            _opts_sh = sorted(set([126, 252, 504, 1260, len(_sh)]))
            _n_days_sh = st.segmented_control(
                "Ventana del histórico de señal", options=_opts_sh,
                default=504 if 504 in _opts_sh else _opts_sh[-1], key="sl_ts_hist",
                format_func=lambda x: {126: "6 meses", 252: "1 año", 504: "2 años",
                                       1260: "5 años"}.get(x, "todo"))
            _n_days_sh = _n_days_sh or (504 if 504 in _opts_sh else _opts_sh[-1])
            st.plotly_chart(build_curve_signal_history_chart(_sh, _vxh, days=_n_days_sh),
                            width="stretch", config=dict(displayModeBar=False))
            _cnt = _sh["signal"].tail(_n_days_sh).value_counts()
            _tot = max(int(_cnt.sum()), 1)
            st.caption(
                f"Señal reproducida día a día con las mismas reglas que la lectura en vivo, "
                f"sobre settles diarios de CBOE ({_vxh.index[0].strftime('%d/%m/%Y')} → "
                f"{_vxh.index[-1].strftime('%d/%m/%Y')}). Reparto en la ventana: "
                f"short vol favorable {_cnt.get(tsig.SIGNAL_SHORT, 0)/_tot:.0%} · neutral "
                f"{_cnt.get(tsig.SIGNAL_NEUTRAL, 0)/_tot:.0%} · long vol "
                f"{_cnt.get(tsig.SIGNAL_LONG, 0)/_tot:.0%}. El percentil es rolling: "
                f"ningún día ve datos posteriores."
                + (f" {int((~_vxh['gap_ok']).sum())} fechas con contrato intermedio "
                   f"ausente en el CDN de CBOE (M2 real = dos meses); se muestran tal cual."
                   if "gap_ok" in _vxh.columns and (~_vxh["gap_ok"]).any() else ""))
        except Exception as _e:                   # noqa: BLE001
            st.warning(f"No se pudo reproducir el histórico de señal: {_e}")
    else:
        st.info("El histórico diario de futuros (data/vx_curve_history.parquet) aún no "
                "existe — lo genera scripts/update_vx_curve.py (GitHub Action diaria).")

    # ── Contexto histórico del contango (percentiles visuales) ──
    if len(_hist_ct) > 300:
        with st.expander("Contango M1→M2 — últimos 6 meses vs bandas de 5 años"):
            _cth = _hist_ct
            if True:
                _win = _cth.tail(1260)
                _p20, _p50, _p80 = _win.quantile([0.20, 0.50, 0.80])
                _recent = _cth.tail(126)
                _fig_ct = go.Figure()
                _fig_ct.add_hrect(y0=float(_win.min()) - 1, y1=float(_p20),
                                  fillcolor='rgba(234,57,67,0.07)', line_width=0)
                _fig_ct.add_hrect(y0=float(_p80), y1=float(_win.max()) + 1,
                                  fillcolor='rgba(22,199,132,0.07)', line_width=0)
                for _lv, _nm in [(_p20, "p20"), (_p50, "p50"), (_p80, "p80")]:
                    _fig_ct.add_hline(y=float(_lv), line_dash='dot',
                                      line_color='#3A3F47', line_width=1,
                                      annotation_text=_nm,
                                      annotation_font=dict(size=9, color='#9AA1A9'))
                _bar_clrs = ['#16C784' if v > 0 else '#EA3943' for v in _recent]
                _fig_ct.add_trace(go.Bar(x=_recent.index, y=_recent.tolist(),
                                         marker_color=_bar_clrs, opacity=0.8,
                                         hovertemplate='%{x|%Y-%m-%d}<br>%{y:+.2f}%<extra></extra>'))
                if front_ct is not None:
                    _fig_ct.add_trace(go.Scatter(
                        x=[_recent.index[-1]], y=[front_ct], mode='markers',
                        marker=dict(size=13, color='#F5A623', symbol='diamond',
                                    line=dict(width=2, color='white')),
                        hovertemplate=f'HOY (CBOE live): {front_ct:+.2f}%<extra></extra>'))
                _fig_ct.update_layout(
                    template="stc", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=280, showlegend=False,
                    margin=dict(l=50, r=20, t=20, b=35), bargap=0,
                    yaxis=dict(title=dict(text="Contango %",
                               font=dict(size=10, color='#9AA1A9')),
                               gridcolor='#1C1F24', ticksuffix='%'),
                    xaxis=dict(gridcolor='#1C1F24'))
                st.plotly_chart(_fig_ct, width="stretch",
                                config=dict(displayModeBar=False))
                st.caption(
                    "Bandas p20/p50/p80 de los últimos 5 años. Contango sobre p80 = "
                    "carry rico (entorno generoso para short vol) · bajo p20 = "
                    "comprimido (el carry no paga el riesgo) · negativo = backwardation "
                    "(terreno de long vol). = contango live de CBOE.")

    # Contango & Difference table (VIXCentral style)
    if len(df_vx) >= 2:
        ct_cells = ""
        diff_cells = ""
        for i in range(len(df_vx) - 1):
            n = i + 1
            p1 = df_vx['Price'].iloc[i]
            p2 = df_vx['Price'].iloc[i + 1]
            ct = cpct(p1, p2)
            diff = round(p2 - p1, 2) if pd.notna(p1) and pd.notna(p2) and p1 > 0 and p2 > 0 else None
            ct_cls = "pos" if ct and ct >= 0 else "neg"
            diff_cls = "pos" if diff and diff >= 0 else "neg"
            ct_cells += f'<td>{n}</td><td class="{ct_cls}">{fp(ct)}</td>'
            diff_cells += f'<td>{n}</td><td class="{diff_cls}">{fv(diff)}</td>'

        m74_ct, m74_diff = None, None
        if len(df_vx) >= 7:
            p4 = df_vx['Price'].iloc[3]
            p7 = df_vx['Price'].iloc[6]
            if pd.notna(p4) and pd.notna(p7) and p4 > 0 and p7 > 0:
                m74_ct = cpct(p4, p7)
                m74_diff = round(p7 - p4, 2)

        st.markdown(f"""
        <table class="ctx">
        <tr><td class="hdr-cell">% Contango</td>{ct_cells}</tr>
        <tr><td class="hdr-cell">Difference</td>{diff_cells}</tr>
        </table>
        """, unsafe_allow_html=True)

        if m74_ct is not None:
            m74_cls = "pos" if m74_ct >= 0 else "neg"
            st.markdown(f"""
            <table class="ctx" style="width:auto;margin-top:4px;">
            <tr><td class="hdr-cell">Month 7 to 4 contango</td>
            <td class="{m74_cls}">{fp(m74_ct)}</td><td class="{m74_cls}">{fv(m74_diff)}</td></tr>
            </table>""", unsafe_allow_html=True)

    # Data table
    if SHOW_TABLE and not df_vx.empty:
        rows = ""
        prev_p = vix_p
        for _, r in df_vx.iterrows():
            sym = r.get('Symbol', '')
            exp = r.get('Expiration')
            exp_s = exp.strftime('%m/%d/%Y') if pd.notna(exp) else "—"
            last = r.get('Last', 0)
            chg = r.get('Change', 0)
            hi = r.get('High', 0)
            lo = r.get('Low', 0)
            settle = r.get('Settlement', 0)
            vol = r.get('Volume', 0)
            price = r.get('Price', 0)
            dte = r.get('DTE', '')

            ct = cpct(prev_p, price) if prev_p and pd.notna(price) and price > 0 else None
            chg_c = "color:var(--g)" if pd.notna(chg) and chg > 0 else "color:var(--r)" if pd.notna(chg) and chg < 0 else ""
            ct_c = "color:var(--g)" if ct and ct >= 0 else "color:var(--r)" if ct else ""
            last_s = f"{last:.2f}" if pd.notna(last) and last > 0 else "—"
            chg_s = f"{chg:+.3f}" if pd.notna(chg) and chg != 0 else "—"
            hi_s = f"{hi:.2f}" if pd.notna(hi) and hi > 0 else "—"
            lo_s = f"{lo:.2f}" if pd.notna(lo) and lo > 0 else "—"
            settle_s = f"{settle:.4f}" if pd.notna(settle) and settle > 0 else "—"
            vol_s = f"{int(vol):,}" if pd.notna(vol) and vol > 0 else "0"

            rows += f"""<tr>
                <td style="color:var(--b);font-weight:600">{sym}</td>
                <td>{exp_s}</td>
                <td style="font-weight:600">{last_s}</td>
                <td style="{chg_c}">{chg_s}</td>
                <td>{hi_s}</td><td>{lo_s}</td>
                <td>{settle_s}</td>
                <td style="{ct_c}">{fp(ct) if ct else '—'}</td>
                <td>{dte}</td>
                <td>{vol_s}</td>
            </tr>"""
            if pd.notna(price) and price > 0:
                prev_p = price

        st.markdown(f"""
        <table class="dtbl">
            <thead><tr><th>Symbol</th><th>Expiration</th><th>Last</th><th>Change</th>
            <th>High</th><th>Low</th><th>Settlement</th><th>Contango</th><th>DTE</th><th>Volume</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    if df_vx.empty:
        st.warning("No se pudieron obtener precios de futuros VIX de CBOE. Pulsa «Actualizar CBOE» "
                   "o vuelve a intentarlo en unos minutos.")

    if not df_vx.empty:
        scraped = df_vx['Scraped_At'].iloc[0] if 'Scraped_At' in df_vx.columns else "?"
        st.caption(f"{len(df_vx)} contratos mensuales · actualizado {scraped} CDMX · CBOE con ~15 min de retraso")


def page_regimen():

    st.markdown(f"""{T.eyebrow("Régimen de volatilidad")}
        <div class="stc-h1">¿En qué entorno de volatilidad estamos?</div>
        <p class="stc-lead">El barómetro combina los 13 indicadores de VTS (curva, roll yield, VVIX,
        SKEW, VIX9D, VRP…) en un percentil de 0 a 100: bajo = entorno de venta de volatilidad, alto =
        cobertura. Debajo, el HMM de tres estados y las señales de bonos, crédito y dólar, que suelen
        oler el estrés antes que el VIX.</p>""", unsafe_allow_html=True)
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    # Slider para ventana
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        baro_window = st.select_slider(
            "Ventana rolling para percentiles",
            options=[252, 504, 756, 1260, 2520, 3780],
            value=1260,
            format_func=lambda x: {
                252: "252d (1 año)",
                504: "504d (2 años)",
                756: "756d (3 años)",
                1260: "1260d (5 años) — recomendado",
                2520: "2520d (10 años)",
                3780: "3780d (15 años) — máxima similitud VTS",
            }.get(x, f"{x} días"),
            help="VTS oficial usa percentiles desde ~2011 (15+ años). "
                 "Ventanas más largas se parecen más al VTS real. "
                 "Ventanas cortas reaccionan más rápido pero sobrestiman vol "
                 "cuando el entorno reciente fue volátil.",
        )
    with col_w2:
        st.write("")
        st.write("")
        if st.button("Recalcular", key="btn_baro_recalc"):
            compute_vts_barometer.clear()
            fetch_edge_extra.clear()

    # ── Cargar datos necesarios ──────────────────────────
    with st.spinner("Calculando barómetro..."):
        df_master_baro = load_master_parquet()
        if df_master_baro.empty:
            st.error("No se encontró data/master.parquet")
            st.stop()

        # Aplicar estrategia (añade BB_SMA20 etc.)
        bt_baro = build_strategy_cached(df_master_baro)

        # Extender con datos live
        live_ext = fetch_live_spy_vix()
        if not live_ext.empty:
            # Merge solo donde el parquet no tiene datos
            for col in live_ext.columns:
                if col in bt_baro.columns:
                    bt_baro[col] = bt_baro[col].fillna(
                        live_ext[col].reindex(bt_baro.index))
                else:
                    bt_baro[col] = live_ext[col].reindex(bt_baro.index)

        # Fetch de datos extra (VVIX, SKEW, HYG, IEF, VIX9D, VIX3M, VIX6M, VIX1Y)
        edge_extra_baro = fetch_edge_extra()

        # Calcular el barómetro
        baro = compute_vts_barometer(
            bt=bt_baro,
            edge_extra=edge_extra_baro,
            gex_summary=None,       # opcional — se agrega si está disponible
            skew_metrics=None,      # opcional
            window=baro_window,
        )

    # ── DIAGNÓSTICO: siempre visible para debugging ─────────
    with st.expander("Diagnóstico de datos (clic para ver qué columnas llegan)",
                     expanded=not bool(baro)):
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("**Parquet master (bt_baro):**")
            parquet_info = []
            parquet_info.append(f"- Filas: **{len(bt_baro):,}**")
            parquet_info.append(f"- Rango: {bt_baro.index[0].date()} → {bt_baro.index[-1].date()}" if len(bt_baro) > 0 else "- Vacío")
            # Columnas relevantes para el barómetro
            needed_cols = ['VIX_Close', 'SPY_Close', 'M1_Price', 'M2_Price', 'M4_Price',
                           'M7_Price', 'M1_DTE', 'M2_DTE', 'VVIX_Live']
            parquet_info.append("\n**Columnas necesarias:**")
            for col in needed_cols:
                if col in bt_baro.columns:
                    nv = bt_baro[col].notna().sum()
                    last = bt_baro[col].iloc[-1] if nv > 0 else 'todos NaN'
                    last_str = f"{last:.2f}" if isinstance(last, (int, float)) and pd.notna(last) else str(last)
                    parquet_info.append(f"- `{col}`: {nv:,} válidos · último = {last_str}")
                else:
                    parquet_info.append(f"- `{col}`: **NO EXISTE en parquet**")
            st.markdown("\n".join(parquet_info))

        with diag_col2:
            st.markdown("**edge_extra (yfinance):**")
            extra_info = []
            expected = ['VVIX', 'SKEW', 'HYG', 'IEF', 'VIX9D', 'VIX3M', 'VIX6M', 'VIX1Y']
            for name in expected:
                if name in edge_extra_baro and not edge_extra_baro[name].empty:
                    df_e = edge_extra_baro[name]
                    nv = df_e['Close'].notna().sum() if 'Close' in df_e.columns else 0
                    last_v = df_e['Close'].iloc[-1] if nv > 0 else None
                    last_str = f"{last_v:.2f}" if last_v is not None and pd.notna(last_v) else '—'
                    extra_info.append(f"- `{name}`: {nv:,} filas · último = {last_str}")
                else:
                    extra_info.append(f"- `{name}`: **no se descargó** (rate limit?)")
            st.markdown("\n".join(extra_info))

        # Info del barómetro calculado
        st.markdown("---")
        if baro:
            st.markdown(f"**Barómetro calculado:**")
            st.markdown(f"- Score: **{baro['score']:.2f}%** · Régimen: **{baro['regime']}**")
            st.markdown(f"- Métricas válidas: **{len([m for m in baro['metrics'] if pd.notna(m['percentile'])])}"
                        f" de {len(baro['metrics'])}**")
            st.markdown(f"- History series: {len(baro['history']) if baro['history'] is not None else 0} puntos")
        else:
            st.error("**compute_vts_barometer devolvió vacío.** Razones posibles:")
            st.markdown(
                "- `bt_baro` tiene menos de 60 filas → necesitas parquet más largo\n"
                "- Ninguna métrica logró calcular percentil válido\n"
                "- `VIX_Close` no está en parquet (es requerido)\n"
                "\n**Mira el diagnóstico arriba para saber qué columna falta.**"
            )

    if not baro:
        st.warning("No se pudo calcular el barómetro — revisa el diagnóstico arriba")
        st.stop()

    # ═══════════════════════════════════════════════════════════════
    # CROSS-ASSET EARLY WARNING — el bond market huele el stress antes
    # ═══════════════════════════════════════════════════════════════
    stress = compute_stress_signals(
        move=_ee_close(edge_extra_baro, 'MOVE'),
        hyg=_ee_close(edge_extra_baro, 'HYG'),
        lqd=_ee_close(edge_extra_baro, 'LQD'),
        ief=_ee_close(edge_extra_baro, 'IEF'),
        dxy=_ee_close(edge_extra_baro, 'DXY'),
    )
    if stress:
        comp_dot = _LIGHT_DOT[stress['light']]
        comp_clr = {'green': 'var(--g)', 'yellow': 'var(--y)',
                    'red': 'var(--r)'}[stress['light']]
        sig_pills = "".join(
            f'<div class="mpill"><div class="ml">{_LIGHT_DOT[s["light"]]} {s["name"]}</div>'
            f'<div class="mv nt" style="font-size:0.95rem">{s["value"]}</div>'
            f'<div style="font-size:0.6rem;color:var(--dim)">stress p{s["stress_pct"]:.0f}</div></div>'
            for s in stress['signals'])
        st.markdown(f"""<div class="icard">
            <div class="ic-title">{comp_dot} Cross-Asset Early Warning
                <span style="color:{comp_clr};font-size:0.75rem;margin-left:0.5rem">
                composite p{stress['composite']:.0f}</span></div>
            <div class="mrow" style="margin-bottom:0">{sig_pills}</div>
        </div>""", unsafe_allow_html=True)
        if stress['light'] == 'red':
            st.warning("Señal cross-asset en zona de stress — el mercado de bonos/divisas "
                       "está pagando un riesgo que el VIX todavía no registra. "
                       "Considera reducir el short vol aunque el barómetro esté verde.")

    # ═══════════════════════════════════════════════════════════════
    # RÉGIMEN HMM — calm / transition / panic (probabilidad filtrada)
    # ═══════════════════════════════════════════════════════════════
    regime_hmm = {}
    if 'SPY_Close' in bt_baro.columns and bt_baro['SPY_Close'].notna().sum() > 500:
        with st.spinner("Ajustando HMM de régimen..."):
            try:
                regime_hmm = compute_regime_cached(bt_baro['SPY_Close'].dropna())
            except Exception as e:
                logging.getLogger("vix_controller").warning(f"HMM régimen: {e}")

    if regime_hmm:
        lbl, clr, desc = REGIME_DISPLAY[regime_hmm['state']]
        cur = regime_hmm['current']
        sig_ann = regime_hmm['sigmas_ann']
        dur = regime_hmm['durations']
        probs_html = " · ".join(
            f"{REGIME_DISPLAY[k][0].title()}: <b>{v:.0%}</b>"
            for k, v in cur.items())
        st.markdown(f"""<div class="icard" style="border-left:3px solid {clr}">
            <div class="ic-title">Régimen de Volatilidad (HMM 3 estados)
                <span style="color:{clr};font-size:0.9rem;margin-left:0.5rem">{lbl}</span></div>
            <div class="ic-row"><span class="ic-label">Probabilidades hoy (filtradas, sin look-ahead)</span>
                <span class="ic-val">{probs_html}</span></div>
            <div class="ic-row"><span class="ic-label">σ anualizada por estado</span>
                <span class="ic-val">{sig_ann[0]:.0f}% / {sig_ann[1]:.0f}% / {sig_ann[2]:.0f}%</span></div>
            <div class="ic-row"><span class="ic-label">Duración esperada</span>
                <span class="ic-val">{dur[0]:.0f}d / {dur[1]:.0f}d / {dur[2]:.0f}d</span></div>
            <div class="ic-row"><span class="ic-label">Lectura</span>
                <span class="ic-val" style="color:{clr}">{desc}</span></div>
        </div>""", unsafe_allow_html=True)
        if regime_hmm['state'] == 'panic' or cur.get('panic', 0) > 0.30:
            st.error("P(pánico) > 30% — históricamente el peor entorno para short vol. "
                     "La señal BB×Contango NO debería operarse hasta que el régimen se normalice.")
        with st.expander("Historia de probabilidades de régimen (2 años)"):
            st.plotly_chart(build_regime_probs_chart(regime_hmm['probs']),
                            width="stretch", config=dict(displayModeBar=False))
            st.caption("Probabilidades FILTRADAS P(estado | datos hasta t) — son las honestas "
                       "para decidir en t, sin usar información futura. "
                       "Matriz de transición diaria:")
            st.dataframe(regime_hmm['transmat'].style.format("{:.1%}"))

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # SUB-TABS: Actual vs Histórico
    # ═══════════════════════════════════════════════════════════════
    sub_actual, sub_hist = st.tabs([
        " Actual",
        " Análisis Histórico",
    ])

    # ═══════════════════════════════════════════════════════════════
    # SUB-TAB: ACTUAL (gauge + KPIs + desglose + timeline últimos N días)
    # ═══════════════════════════════════════════════════════════════
    with sub_actual:
        # ── Layout principal: Gauge + KPIs ───────────────────
        col_gauge, col_kpi = st.columns([1.15, 1])

        with col_gauge:
            gauge_fig = build_vts_barometer_gauge(
                score=baro['score'],
                regime=baro['regime'],
                date_str=baro['date'].strftime('%Y-%m-%d'),
            )
            st.plotly_chart(gauge_fig, width="stretch", config=dict(displayModeBar=False))

        with col_kpi:
            score    = baro['score']
            regime   = baro['regime']
            position = baro['position']
            n_metrics = len(baro['metrics'])

            # Determinar color del régimen
            if   score < 20: rc = 'var(--g)'
            elif score < 40: rc = 'var(--g)'
            elif score < 60: rc = 'var(--y)'
            elif score < 80: rc = '#F5A623'
            else:            rc = 'var(--r)'

            # Percentil del score HOY vs su historia
            if baro['history'] is not None and not baro['history'].empty:
                h = baro['history']
                score_pctile = (h <= score).mean() * 100
                h_mean = h.mean()
                h_max = h.max()
                h_min = h.min()
            else:
                score_pctile = 50; h_mean = 50; h_max = 100; h_min = 0

            st.markdown(f"""
            <div style="padding:0.5rem 0;">
                <div class="sig-box" style="background:rgba(245,166,35,0.08);
                     border-color:#F5A623;margin-bottom:0.8rem;">
                    <div class="sl" style="color:{rc};">{score:.2f}%</div>
                    <div class="sd" style="font-size:0.85rem;color:{rc};font-weight:700;">
                        Régimen: {regime}
                    </div>
                    <div class="sd" style="margin-top:4px;">{position}</div>
                </div>
                <div class="icard">
                    <div class="ic-title">Métricas del barómetro</div>
                    <div class="ic-row"><span class="ic-label">Métricas activas</span>
                        <span class="ic-val">{n_metrics}</span></div>
                    <div class="ic-row"><span class="ic-label">Ventana rolling</span>
                        <span class="ic-val">{baro['window']}d</span></div>
                    <div class="ic-row"><span class="ic-label">Score HOY</span>
                        <span class="ic-val" style="color:{rc};font-weight:700">{score:.2f}%</span></div>
                    <div class="ic-row"><span class="ic-label">Percentil histórico</span>
                        <span class="ic-val">{score_pctile:.1f}°</span></div>
                    <div class="ic-row"><span class="ic-label">Media histórica</span>
                        <span class="ic-val">{h_mean:.1f}%</span></div>
                    <div class="ic-row"><span class="ic-label">Rango (min-max)</span>
                        <span class="ic-val">{h_min:.1f}% – {h_max:.1f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ── Timeline del score (últimos N días) ──────────────
        if baro['history'] is not None and not baro['history'].empty:
            hist_fig = build_vts_history_chart(baro['history'], window=baro_window)
            st.plotly_chart(hist_fig, width="stretch", config=dict(displayModeBar=False))

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ── Desglose de métricas (toggle: orden VTS vs stress) ───
        col_mode1, col_mode2 = st.columns([3, 1])
        with col_mode2:
            sort_mode = st.radio(
                "Orden",
                options=['vts', 'stress'],
                format_func=lambda x: 'Canónico VTS' if x == 'vts' else 'Por stress',
                key='baro_sort_mode',
                horizontal=False,
                label_visibility='collapsed',
            )
        metrics_fig = build_vts_metrics_table(baro['metrics'], sort_mode=sort_mode)
        st.plotly_chart(metrics_fig, width="stretch", config=dict(displayModeBar=False))

        # ── Interpretación y guía de lectura ─────────────────
        with st.expander("Metodología y guía de lectura", expanded=False):
            st.markdown(f"""
**Qué mide el barómetro**

El VTS Volatility Barometer combina **{n_metrics} métricas de volatilidad** en un único
score 0-100%, replicando la metodología oficial de
[volatilitytradingstrategies.com](https://www.volatilitytradingstrategies.com).
Cada métrica captura una porción del mercado (futuros VIX, opciones SPX, term structure,
credit, etc.) — combinadas ofrecen la lectura más robusta del régimen de vol actual.

**Los 13 indicadores VTS (alineados al gauge oficial):**

| # | Indicador | Qué mide |
|---|-----------|----------|
| 1 | **M1:M2 VIX FUTURES** | Contango frontal — ratio entre M1 y M2 |
| 2 | **M4-M7 VIX FUTURES** | Contango en la curva larga |
| 3 | **VX30:VIX ROLL YIELD** | Roll yield constant-maturity a 30 días |
| 4 | **VIX / VVIX / VOLI** | Composite nivel absoluto (VIX + VVIX + RV20) |
| 5 | **CASH VIX OSCILLATOR** | Promedio de ratios VIX9D/VIX, VIX/VIX3M, VIX/VIX6M, VIX/VIX1Y |
| 6 | **VIX - VOLI RESIDUAL** | VIX menos VOLI (proxy RV20) |
| 7 | **TRADERS VRP** | WMA 5d con pesos 1-5 de (VX30 − HV5) |
| 8 | **SDEX / TDEX / SKEW** | CBOE SKEW Index + proxies |
| 9 | **VIX:VIX3M MEDIUM** | Ratio medium-term (invirtido en backwardation) |
| 10 | **EXTREME PUT/CALL RATIO** | Momentum del SKEW 5d (proxy demanda de puts) |
| 11 | **VIX9D:VIX FAST** | Ratio fast-term (>1 = cola corta caliente) |
| 12 | **VOLPOCALYPSE THRESHOLD** | Trigger compuesto: VIX + backwardation + momentum |
| 13 | **VIX LEVEL (core)** | Nivel absoluto del VIX — ancla del barómetro |

Cada métrica se convierte a **percentil rolling** (ventana configurable 126–756 días).
El score final es el **promedio ponderado** de todos los percentiles.

**Proxies usados (cuando el dato original no está disponible):**
- **VOLI** → Se usa Realized Volatility 20d del SPY (VOLI no está en yfinance gratis)
- **SDEX / TDEX** → Se usa solo CBOE SKEW (los otros son propietarios)
- **Put/Call Ratio extremo** → Momentum 5d del SKEW Index
- **VX30** → Interpolación lineal M1↔M2 a 30 días

**Interpretación del score (idéntica al VTS original):**

- **0-20% (Vol BAJA)**: Todos los indicadores apuntan a vol estable →
  **Short vol agresivo** (SVXY/SVIX al 100%).
- **20-40% (MODERADA)**: Mayoría verdes pero algunos elevados →
  **Short vol estándar** (SVXY al 75-100%).
- **40-60% (MID)**: Señales mixtas → **Cash o posición parcial** (25-50%).
  Es el rango donde más falsos positivos ocurren.
- **60-80% (ELEVADA)**: Mayoría rojas → **Cash, evitar short vol**.
- **80-100% (EXTREMA)**: Entorno de crisis (COVID 2020, Vol-geddon 2018, Aug 2015) →
  Oportunidad de **long VIX / short equities / long puts**.

**Diferencias vs VTS original:**

VTS tiene 10+ años de refinamiento propietario en pesos y métricas exactas. Esta
implementación usa las mismas fórmulas públicas documentadas por Brent Osachoff
(`volatilitymaster.wordpress.com`) pero los pesos y proxies pueden diferir levemente.
La utilidad principal es como **filtro de régimen** complementario al Monitor Operativo.

**Uso recomendado en combo con Monitor Operativo:**

- Score **< 40%** + señal LONG = **convicción alta**, tamaño completo
- Score **40-60%** + señal LONG = **reducir tamaño o esperar confirmación**
- Score **> 60%** = **NO tomar señal LONG** aunque el Monitor la marque
- Score **> 80%** = considerar **long vol (VXX/UVXY) como hedge táctico**
""")

        st.caption(
            f"VTS Volatility Barometer v1.0 · "
            f"Inspirado en volatilitytradingstrategies.com · "
            f"Ventana: {baro_window}d · Métricas: {n_metrics} · "
            f"Última actualización: {baro['date'].strftime('%Y-%m-%d')}"
        )

    # ═══════════════════════════════════════════════════════════════
    # SUB-TAB: ANÁLISIS HISTÓRICO (serie completa + date picker + heatmap + dist)
    # ═══════════════════════════════════════════════════════════════
    with sub_hist:
        hist = baro.get('history')

        if hist is None or hist.empty or hist.dropna().empty:
            st.warning(
                "No hay serie histórica disponible. Esto puede pasar si:\n\n"
                "- Los datos de yfinance no se descargaron (rate limit)\n"
                "- El parquet del barómetro `data/baro_history.parquet` no existe aún\n"
                "- La ventana rolling es más larga que la data disponible"
            )
            st.info(
                "**Tip:** ejecuta `python scripts/update_baro_parquet.py` para generar "
                "`data/baro_history.parquet` con 15+ años de datos. "
                "Después activa la GitHub Action para mantenerlo actualizado diariamente."
            )
            st.stop()

        h_clean = hist.dropna()

        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
            f"color:#9AA1A9;padding:0.3rem 0 0.8rem;'>"
            f"Serie completa: <b>{len(h_clean):,} días</b> · "
            f"{h_clean.index[0].date()} → {h_clean.index[-1].date()} · "
            f"Ventana rolling: <b>{baro_window}d</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Date picker: mirar el score en una fecha específica ──
        st.markdown("### Score en una fecha específica")
        dp_col1, dp_col2 = st.columns([1, 2])

        with dp_col1:
            selected_date = st.date_input(
                "Selecciona una fecha",
                value=h_clean.index[-1].date(),
                min_value=h_clean.index[0].date(),
                max_value=h_clean.index[-1].date(),
                key='baro_date_picker',
            )

        # Buscar la fecha más cercana en la serie
        sel_ts = pd.Timestamp(selected_date)
        # Primero intentamos exact match, si no, buscamos la más cercana anterior
        if sel_ts in h_clean.index:
            matched_date = sel_ts
            matched_score = float(h_clean.loc[sel_ts])
        else:
            prev_dates = h_clean.index[h_clean.index <= sel_ts]
            if len(prev_dates) > 0:
                matched_date = prev_dates[-1]
                matched_score = float(h_clean.loc[matched_date])
            else:
                matched_date = h_clean.index[0]
                matched_score = float(h_clean.iloc[0])

        # Determinar régimen de esa fecha
        if   matched_score < 20: reg_sel, clr_sel = "VOL BAJA",  '#16C784'
        elif matched_score < 40: reg_sel, clr_sel = "MODERADA",  '#16C784'
        elif matched_score < 60: reg_sel, clr_sel = "MID",       '#F5A623'
        elif matched_score < 80: reg_sel, clr_sel = "ELEVADA",   '#F5A623'
        else:                    reg_sel, clr_sel = "EXTREMA",   '#EA3943'

        # Percentil de esa fecha vs todo el histórico
        pct_rank = (h_clean <= matched_score).mean() * 100

        with dp_col2:
            st.markdown(
                f"""
                <div class="icard" style="margin-top:0;">
                    <div class="ic-title">Score el {matched_date.strftime('%Y-%m-%d')}</div>
                    <div style="display:flex;align-items:center;gap:1.5rem;padding:0.4rem 0;">
                        <div style="font-family:Inter,sans-serif;font-weight:800;
                                    font-size:2.2rem;color:{clr_sel};">
                            {matched_score:.2f}%
                        </div>
                        <div>
                            <div style="font-family:Inter,sans-serif;font-weight:700;
                                        font-size:1rem;color:{clr_sel};">
                                {reg_sel}
                            </div>
                            <div style="font-family:JetBrains Mono,monospace;font-size:0.75rem;
                                        color:#9AA1A9;margin-top:2px;">
                                Percentil histórico: {pct_rank:.1f}°
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ── Gráfico histórico completo ───────────────────────
        st.markdown("### Serie histórica completa")
        full_fig = build_vts_history_full(h_clean)
        st.plotly_chart(full_fig, width="stretch",
                        config=dict(displayModeBar=True, displaylogo=False,
                                    modeBarButtonsToRemove=['select2d', 'lasso2d']))

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ── Heatmap + Distribución ────────────────────────────
        col_hm, col_dist = st.columns([1.3, 1])

        with col_hm:
            st.markdown("### Heatmap estacional")
            hm_fig = build_vts_monthly_heatmap(h_clean)
            st.plotly_chart(hm_fig, width="stretch", config=dict(displayModeBar=False))

        with col_dist:
            st.markdown("### Tiempo por régimen")
            dist_fig, regime_stats = build_vts_regime_distribution(h_clean)
            st.plotly_chart(dist_fig, width="stretch", config=dict(displayModeBar=False))

            # Insight interpretativo
            dominant = max(regime_stats.items(), key=lambda x: x[1]['pct']) if regime_stats else None
            if dominant:
                st.caption(
                    f"**Régimen dominante**: {dominant[0]} "
                    f"({dominant[1]['pct']:.1f}% del tiempo · {dominant[1]['days']:,} días)"
                )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # ── Estadísticas generales + tabla de datos ──────────
        st.markdown("### Estadísticas del barómetro histórico")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Score actual", f"{h_clean.iloc[-1]:.2f}%",
                      delta=f"{h_clean.iloc[-1] - h_clean.iloc[-22]:+.2f}%"
                      if len(h_clean) > 22 else None,
                      help="Cambio vs hace 22 días (1 mes)")
        with col_s2:
            st.metric("Media histórica", f"{h_clean.mean():.2f}%",
                      help=f"Media de los {len(h_clean):,} días")
        with col_s3:
            st.metric("Mediana histórica", f"{h_clean.median():.2f}%")
        with col_s4:
            st.metric("Desviación estándar", f"{h_clean.std():.2f}%")

        col_s5, col_s6, col_s7, col_s8 = st.columns(4)
        with col_s5:
            st.metric("Máximo histórico", f"{h_clean.max():.2f}%",
                      help=f"Alcanzado el {h_clean.idxmax().date()}")
        with col_s6:
            st.metric("Mínimo histórico", f"{h_clean.min():.2f}%",
                      help=f"Alcanzado el {h_clean.idxmin().date()}")
        with col_s7:
            pct_above_60 = (h_clean > 60).mean() * 100
            st.metric("% días con score > 60", f"{pct_above_60:.1f}%",
                      help="Entorno defensivo / cash")
        with col_s8:
            pct_below_40 = (h_clean < 40).mean() * 100
            st.metric("% días con score < 40", f"{pct_below_40:.1f}%",
                      help="Entorno favorable para short vol")

        # ── Tabla de días con score extremo ──────────────────
        with st.expander("Top 20 días con score más extremo (alto y bajo)"):
            col_hi, col_lo = st.columns(2)
            with col_hi:
                st.markdown("**Top 20 más altos (mayor stress)**")
                top_hi = h_clean.nlargest(20).reset_index()
                top_hi.columns = ['Fecha', 'Score']
                top_hi['Fecha'] = top_hi['Fecha'].dt.strftime('%Y-%m-%d')
                top_hi['Score'] = top_hi['Score'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(top_hi, width="stretch", hide_index=True)
            with col_lo:
                st.markdown("**Top 20 más bajos (mayor calma)**")
                top_lo = h_clean.nsmallest(20).reset_index()
                top_lo.columns = ['Fecha', 'Score']
                top_lo['Fecha'] = top_lo['Fecha'].dt.strftime('%Y-%m-%d')
                top_lo['Score'] = top_lo['Score'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(top_lo, width="stretch", hide_index=True)

        # ── Descarga CSV ─────────────────────────────────────
        csv_data = h_clean.to_frame(name='score').to_csv(index_label='date')
        st.download_button(
            "Descargar serie completa CSV",
            data=csv_data,
            file_name=f"vts_barometer_history_{h_clean.index[-1].date()}.csv",
            mime='text/csv',
        )


def page_volatilidad():

    st.markdown(f"""{T.eyebrow("Volatilidad realizada e implícita")}
        <div class="stc-h1">La prima que cobra el vendedor de volatilidad</div>
        <p class="stc-lead">VRP (implícita frente a realizada), previsión HAR-RV de la volatilidad de los
        próximos 22 días, roll yield, VVIX, SKEW y crédito. Es la parte de la ecuación que dice si el
        carry compensa el riesgo.</p>""", unsafe_allow_html=True)
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    df_master_edge = load_master_parquet()

    if df_master_edge.empty:
        st.error("No se pudo cargar el Master para Edge Analytics.")
    else:
        with st.spinner("Calculando edge analytics..."):
            edge_extra = fetch_edge_extra()
            edge = compute_edge_analytics(df_master_edge, edge_extra)

        if 'bt' not in edge:
            st.error("Datos insuficientes para edge analytics.")
        else:
            ebt = edge['bt']
            last_e = ebt.iloc[-1]

            def ecard(label, val, sub, clr="nt"):
                c = "var(--g)" if clr == "up" else "var(--r)" if clr == "dn" else "var(--b)"
                return (f'<div class="mpill"><div class="ml">{label}</div>'
                        f'<div class="mv" style="color:{c}">{val}</div>'
                        f'<div style="font-size:0.6rem;color:var(--dim)">{sub}</div></div>')

            # ── Métricas con HAR-VRP cuando disponible ──────────────
            vrp_har  = last_e.get('VRP_HAR', np.nan)
            har_fc   = last_e.get('HAR_Forecast', np.nan)
            vrp_trad = last_e.get('VRP', np.nan)
            vrp_val  = vrp_har if pd.notna(vrp_har) else vrp_trad
            vrp_pct  = edge.get('vrp_percentile', '?')
            rv20_val = last_e.get('RV20', np.nan)
            def _ultimo_valido(col):
                return float(ebt[col].dropna().iloc[-1]) if col in ebt.columns and ebt[col].notna().any() else np.nan
            ry_val   = _ultimo_valido('Roll_Yield')
            skew_val = _ultimo_valido('SKEW')
            # Roll yield con la curva en vivo (master.parquet no se actualiza a diario)
            if not df_vx.empty and vix_spot and len(df_vx) > 0:
                _m1, _dte1 = float(df_vx['Price'].iloc[0]), float(df_vx['DTE'].iloc[0])
                if _m1 > 0 and _dte1 >= 1:
                    ry_val = (_m1 - vix_spot['price']) / _m1 * (365 / _dte1) * 100
            if (skew_val != skew_val) and (q := fetch_index_live("SKEW")):
                skew_val = q["price"]
            # VVIX: preferir live, caer a ratio calculado
            vvix_live = last_e.get('VVIX_Live', np.nan)
            vvix_r    = last_e.get('VVIX_VIX', np.nan)
            vvix_disp = vvix_r if pd.notna(vvix_r) else (
                vvix_live / last_e['VIX_Close'] if pd.notna(vvix_live) and last_e['VIX_Close'] > 0 else np.nan
            )

            use_har = pd.notna(vrp_har) and pd.notna(har_fc)
            vrp_label = "VRP·HAR" if use_har else "VRP·RV20"
            vrp_sub   = (f"P{vrp_pct} · VIX:{last_e['VIX_Close']:.0f} vs E[RV]:{har_fc:.0f}"
                         if use_har and pd.notna(har_fc)
                         else f"P{vrp_pct} hist" if vrp_pct != '?' else "")

            vrp_str  = f"{vrp_val:+.1f}" if pd.notna(vrp_val) else "N/A"
            vrp_clr  = "up" if pd.notna(vrp_val) and vrp_val > 2 else "dn" if pd.notna(vrp_val) and vrp_val < 0 else "nt"
            har_str  = f"{har_fc:.1f}" if pd.notna(har_fc) else "N/A"
            ry_str   = f"{ry_val:+.1f}%" if pd.notna(ry_val) else "N/A"
            ry_clr   = "up" if pd.notna(ry_val) and ry_val > 0 else "dn" if pd.notna(ry_val) and ry_val < 0 else "nt"
            vvix_str = f"{vvix_disp:.2f}" if pd.notna(vvix_disp) else "N/A"
            vvix_clr = "dn" if pd.notna(vvix_disp) and vvix_disp > 6 else "up" if pd.notna(vvix_disp) and vvix_disp < 5 else "nt"
            skew_str = f"{skew_val:.0f}" if pd.notna(skew_val) else "N/A"
            skew_clr = "dn" if pd.notna(skew_val) and skew_val > 150 else "up" if pd.notna(skew_val) and skew_val < 130 else "nt"

            st.markdown(f"""<div class="mrow">
                {ecard(vrp_label, vrp_str, vrp_sub, vrp_clr)}
                {ecard("HAR Forecast", har_str, "E[RV futura 22d]", "nt")}
                {ecard("RV20 (trailing)", f"{rv20_val:.1f}" if pd.notna(rv20_val) else "N/A", "Pasado (ref)", "nt")}
                {ecard("Roll Yield", ry_str, "Carry anualizado", ry_clr)}
                {ecard("VVIX/VIX", vvix_str, "live yfinance · >6=peligro", vvix_clr)}
                {ecard("SKEW", skew_str, "live yfinance · >150=extremo", skew_clr)}
                {ecard("VIX", f"{last_e['VIX_Close']:.1f}", "spot", "nt")}
            </div>""", unsafe_allow_html=True)

            # ═══════════════════════════════════════════════════════
            # VRP TRACKER — ¿la prima que cosechas está cara o barata?
            # ═══════════════════════════════════════════════════════
            vrp_track = compute_vrp_tracker(ebt['VIX_Close'], ebt['RV20'])
            if vrp_track:
                _vt = vrp_track
                _reg_clr = {"COMPRIMIDA": "var(--y)", "NORMAL": "var(--b)",
                            "RICA": "var(--g)", "NEGATIVA": "var(--r)"}[_vt['regime']]
                _reg_desc = {
                    "COMPRIMIDA": "Prima en p<20 — mala compensación para asumir riesgo de varianza",
                    "NORMAL":     "Prima en rango histórico normal",
                    "RICA":       "Prima en p>80 — entorno generoso para cosechar short vol",
                    "NEGATIVA":   "RV > VIX — el short vol está PAGANDO por perder (stress agudo)",
                }[_vt['regime']]
                st.markdown(f"""<div class="icard" style="border-left:3px solid {_reg_clr}">
                    <div class="ic-title">VRP Tracker
                        <span style="color:{_reg_clr};font-size:0.9rem;margin-left:0.5rem">{_vt['regime']}</span></div>
                    <div class="ic-row"><span class="ic-label">VRP hoy (VIX − RV20)</span>
                        <span class="ic-val">{_vt['vrp_vol']:+.2f} vol pts</span></div>
                    <div class="ic-row"><span class="ic-label">VRP en varianza (VIX²−RV20²)/100 · Carr-Wu</span>
                        <span class="ic-val">{_vt['vrp_var']:+.2f} · percentil <b>{_vt['percentile']:.0f}</b></span></div>
                    <div class="ic-row"><span class="ic-label">Cosecha típica 1 año / % días negativos</span>
                        <span class="ic-val">{_vt['mean_1y']:+.2f} pts · {_vt['pct_negative_1y']:.0f}% neg</span></div>
                    <div class="ic-row"><span class="ic-label">Lectura</span>
                        <span class="ic-val" style="color:{_reg_clr}">{_reg_desc}</span></div>
                </div>""", unsafe_allow_html=True)
                with st.expander("VRP histórico (3 años)", expanded=False):
                    st.plotly_chart(build_vrp_tracker_chart(_vt['df']), width="stretch",
                                    config=dict(displayModeBar=False))
                    st.caption(
                        "Panel superior: VRP diario en vol points (verde = cosechando, rojo = pagando). "
                        "Panel inferior: percentil rolling del VRP en unidades de varianza — "
                        "la métrica de Carr-Wu castiga más los episodios de alta vol. "
                        "Zona roja (p<20) = prima comprimida; zona verde (p>80) = prima rica.")

            # ── Expander: HAR model diagnostics ─────────────────────
            har_beta = edge.get('har_beta', {})
            if har_beta or use_har:
                with st.expander("Modelo HAR-RV — Detalles y coeficientes", expanded=False):
                    st.markdown("""
**¿Qué es HAR-RV?** (Corsi 2009 — *A Simple Approximate Long-Memory Model of Realized Volatility*)

El VIX mide volatilidad **implícita** de los próximos 30 días. Compararlo con RV20 trailing es incorrecto porque mezcla horizontes temporales distintos.

El modelo HAR-RV estima directamente `E[RV_{t, t+22}]` — la **volatilidad esperada** para los próximos 22 días hábiles:

```
E[RV_futura] = β₀ + β₁·RV_diaria + β₂·RV_semanal(5d) + β₃·RV_mensual(22d)
```

**¿Por qué funciona?**
- Captura la *heterogeneidad* de los agentes: day traders (β₁), gestores de semana (β₂), institucionales de mes (β₃)
- La volatilidad tiene *memoria larga* — las tres frecuencias juntas la capturan mejor que cualquiera sola
- Outperforms GARCH en forecasting fuera de muestra (Andersen, Bollerslev, Diebold 2003)

**VRP correcto:** `VIX_t - HAR_forecast_t`
- Positivo → el mercado sobreestima la vol futura → **puedes vender vol con descuento**
- Negativo → el mercado subestima la vol → **la estrategia inverse vol está en zona de riesgo**
                    """)
                    if har_beta:
                        cols_b = st.columns(len(har_beta))
                        for i, (k, v) in enumerate(har_beta.items()):
                            cols_b[i].metric(k, f"{v:.3f}")

            # Calendario de eventos
            upcoming = edge.get('upcoming_events', [])
            if upcoming:
                ev_html = ""
                for name, dt, days in upcoming:
                    ev_clr = "var(--r)" if days <= 2 else "var(--y)" if days <= 5 else "var(--dim)"
                    ev_tag = "HOY" if days == 0 else f"en {days}d"
                    ev_html += (f'<span style="background:var(--card);border:1px solid {ev_clr};'
                               f'border-radius:4px;padding:0.2rem 0.6rem;margin-right:0.4rem;'
                               f'font-family:JetBrains Mono;font-size:0.75rem;color:{ev_clr}">'
                               f'{name} {dt.strftime("%b %d")} · {ev_tag}</span>')
                st.markdown(f'<div style="margin:0.4rem 0 0.8rem">{ev_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:JetBrains Mono;font-size:0.75rem;color:#16C784;'
                            'margin:0.4rem 0 0.8rem">Sin eventos macro en los proximos 14 dias</div>',
                            unsafe_allow_html=True)

            # Edge Verdict
            warnings_e = []
            if pd.notna(vrp_val) and vrp_val < 0:
                warnings_e.append(f"{'VRP·HAR' if use_har else 'VRP'} negativo ({vrp_val:+.1f} pts) — estas pagando por estar posicionado")
            if pd.notna(vvix_r) and vvix_r > 6:
                warnings_e.append("VVIX/VIX > 6 — dealers anticipan spike")
            if pd.notna(ry_val) and ry_val < 0:
                warnings_e.append("Roll Yield negativo — backwardation erosiona el carry")
            if pd.notna(skew_val) and skew_val > 150:
                warnings_e.append("SKEW extremo — alta demanda de proteccion")
            if any(ev[2] <= 2 for ev in upcoming):
                warnings_e.append("Evento macro inminente — considerar reducir exposicion")

            if len(warnings_e) >= 3:
                verdict, v_clr, v_bg = "EDGE COMPROMETIDO", "var(--r)", "var(--rbg)"
            elif len(warnings_e) >= 1:
                verdict, v_clr, v_bg = "EDGE ACTIVO — CON PRECAUCION", "var(--y)", "#2A1F08"
            else:
                verdict, v_clr, v_bg = "EDGE SALUDABLE", "var(--g)", "var(--gbg)"

            st.markdown(f"""<div style="background:{v_bg};border:1px solid {v_clr};
                border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.8rem">
                <span style="font-family:Inter;font-weight:800;font-size:1rem;color:{v_clr}">{verdict}</span>
                <span style="font-family:JetBrains Mono;font-size:0.7rem;color:var(--dim);margin-left:1rem">
                {len(warnings_e)} warning{'s' if len(warnings_e) != 1 else ''}</span>
            </div>""", unsafe_allow_html=True)
            for w in warnings_e:
                st.warning(w)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            # ── Última fecha disponible ──────────────────────────────
            last_date_ebt = ebt.index[-1].date()
            st.markdown(
                f'<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#9AA1A9;margin-bottom:0.4rem">'
                f'Datos hasta: <b style="color:#F4F5F6">{last_date_ebt}</b>'
                f'{"  Al día" if last_date_ebt >= (now_cdmx().date() - timedelta(days=3)) else "  parquet desactualizado"}'
                f'</div>',
                unsafe_allow_html=True)

            # ── VRP + Backtest ───────────────────────────────────────
            try:
                st.plotly_chart(build_vrp_chart(ebt), width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error VRP: {e}")

            # Backtest del modelo HAR-A
            har_bt = edge.get('har_backtest', {})
            if har_bt:
                with st.expander("Backtest HAR-A — ¿Qué tan bien predice la volatilidad futura?", expanded=False):
                    # Métricas en tabla
                    brow1 = {
                        "Métrica":      ["R² OOS", "RMSE", "MAE", "QLIKE", "Dir. Accuracy"],
                        "HAR-A":        [f"{har_bt.get('r2_oos','—'):.3f}",
                                         f"{har_bt.get('rmse','—'):.2f}",
                                         f"{har_bt.get('mae','—'):.2f}",
                                         f"{har_bt.get('qlike','—'):.4f}",
                                         f"{har_bt.get('dir_acc','—'):.1f}%"],
                        "RV_m (naive)": [f"{har_bt.get('r2_naive','—'):.3f}",
                                         f"{har_bt.get('rmse_naive','—'):.2f}", "—", "—", "—"],
                        "EWMA(0.94)":   [f"{har_bt.get('r2_ewma','—'):.3f}",
                                         f"{har_bt.get('rmse_ewma','—'):.2f}", "—", "—", "—"],
                    }
                    mz_a = har_bt.get('mz_alpha', '—'); mz_b = har_bt.get('mz_beta', '—')
                    n_t  = har_bt.get('n_test', '—')
                    st.dataframe(pd.DataFrame(brow1), width="stretch", hide_index=True)
                    st.markdown(f"""
<div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#9AA1A9;margin:0.3rem 0">
<b style="color:#F4F5F6">Mincer-Zarnowitz:</b> α={mz_a} (ideal 0) · β={mz_b} (ideal 1) ·
<b style="color:#F4F5F6">Muestra test:</b> {n_t} días (~2 años)
</div>
<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#9AA1A9;margin-top:0.3rem">
<b>Interpretación:</b>
R² OOS > 0.20 = buena predicción (vol es difícil de predecir) ·
QLIKE penaliza asimétricamente subestimaciones ·
Dir. Acc. > 55% = útil para timing ·
β ≈ 1.0 en MZ = sin sesgo sistemático
</div>""", unsafe_allow_html=True)

                    try:
                        fig_bts, fig_mz = build_har_backtest_charts(har_bt)
                        col_bts1, col_bts2 = st.columns([1.4, 1])
                        with col_bts1:
                            if fig_bts.data:
                                st.plotly_chart(fig_bts, width="stretch", config=dict(displayModeBar=False))
                        with col_bts2:
                            if fig_mz.data:
                                st.plotly_chart(fig_mz, width="stretch", config=dict(displayModeBar=False))
                    except Exception as ex:
                        st.error(f"Error backtest charts: {ex}")

            try:
                st.plotly_chart(build_rv_chart(ebt), width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error RV: {e}")

            col_ry, col_vv = st.columns(2)
            with col_ry:
                try:
                    st.plotly_chart(build_roll_yield_chart(ebt), width="stretch", config=dict(displayModeBar=False))
                except Exception as e:
                    st.error(f"Error Roll Yield: {e}")
            with col_vv:
                try:
                    st.plotly_chart(build_vvix_ratio_chart(ebt), width="stretch", config=dict(displayModeBar=False))
                except Exception as e:
                    st.error(f"Error VVIX: {e}")

            col_sk, col_cr = st.columns(2)
            with col_sk:
                try:
                    fig_sk = build_skew_chart(ebt)
                    if fig_sk.data:
                        st.plotly_chart(fig_sk, width="stretch", config=dict(displayModeBar=False))
                    else:
                        st.info("SKEW data no disponible")
                except Exception as e:
                    st.error(f"Error SKEW: {e}")
            with col_cr:
                try:
                    fig_cr = build_credit_chart(ebt)
                    if fig_cr.data:
                        st.plotly_chart(fig_cr, width="stretch", config=dict(displayModeBar=False))
                    else:
                        st.info("Credit spread data no disponible")
                except Exception as e:
                    st.error(f"Error Credit: {e}")

            har_src = 'HAR-A (Patton & Sheppard 2015)' if use_har else 'RV20 trailing'
            st.caption(
                f"Edge Analytics · VRP: {har_src} · "
                f"Datos: SPY+VIX parquet extendido con yfinance live (hasta {ebt.index[-1].date()}) · "
                f"VVIX / SKEW / Credit: ^VVIX ^SKEW HYG IEF yfinance")


def _vista_skew():

    # ── Controles ─────────────────────────────────────────────
    col_c1, col_c2, col_c3, col_c4 = st.columns([1,1,1,1])
    with col_c1:
        skew_ticker = st.selectbox(
            "Subyacente", ["SPY","QQQ","IWM","GLD","TLT"], index=0,
            help="SPY = mayor liquidez de opciones",
        )
    with col_c2:
        n_exps = st.slider("Nº Vencimientos", 2, 6, 4,
                           help="Cada vencimiento tarda ~0.6-1.5s")
    with col_c3:
        skew_rfr = st.number_input("Risk-Free Rate (r)", 0.0, 0.15,
                                   value=float(get_risk_free_rate()), step=0.001, format="%.3f",
                                   help="Tasa libre de riesgo (live desde ^IRX; edita si deseas)")
    with col_c4:
        skew_div = st.number_input("Dividend Yield (q)", 0.0, 0.10,
                                   value=float(get_dividend_yield("SPY")), step=0.001, format="%.3f",
                                   help="Dividend yield live (SPY TTM); edita si deseas")

    col_c5, col_c6, col_c7, col_c8 = st.columns([1,1,1,1])
    with col_c5:
        mon_lo = st.slider("Strike mín (%spot)", 75, 95, 85,
                           help="Solo strikes con bid+ask activos — defecto 85% filtra iqlíquidos") / 100

        mon_hi = st.slider("Strike máx (%spot)", 105, 130, 115,
                           help="Defecto 115% — más allá hay bid=0 en la mayoría de chains") / 100
    with col_c7:
        y_axis_mode = st.selectbox("Eje Y", ["% vs Spot", "Log-moneyness ln(K/S)"],
                                   help="Log-moneyness es la convención académica de BS")
        y_mode = "moneyness" if y_axis_mode.startswith("%") else "log"
    with col_c8:
        view_mode = st.selectbox("Vista superficie", ["3D Surface", "Heatmap 2D"])

    if st.button("Actualizar opciones", key="refresh_options"):
        fetch_options_chains.clear()
        st.rerun()

    # ── Fetch raw data ────────────────────────────────────────
    est_secs = n_exps * 1.2 + 2
    with st.spinner(f"Descargando {n_exps} vencimientos de {skew_ticker} (~{est_secs:.0f}s)…"):
        opt_chains_raw, opt_spot = fetch_options_chains(skew_ticker, n_exp=n_exps)

    if not opt_chains_raw or not opt_spot:
        st.error(f"**Yahoo Finance rate-limited** — no se cargaron opciones de **{skew_ticker}**.")
        st.info(
            "Streamlit Cloud comparte IP con miles de apps, Yahoo aplica rate limits.\n\n"
            "**Qué puedes hacer:**\n"
            "- Esperar 15-30 min (cache negativo automático)\n"
            "- Reducir a 2 vencimientos y reintentar\n"
            "- Intentar en horario de mercado (9:30-16:00 ET) cuando la API de Yahoo es más estable\n"
            "- Ejecutar la app localmente donde tu IP no está bloqueada"
        )
        if st.button("Reintentar ahora (limpia cache negativo)", key="btn_skew_retry"):
            for k in list(st.session_state.keys()):
                if k.startswith("_opts_fail_"):
                    del st.session_state[k]
            fetch_options_chains.clear()
            st.rerun()
        st.stop()

    # ── Calcular IV con Black-Scholes (Brent's method) ───────
    with st.spinner("Calculando IV Black-Scholes…"):
        opt_chains = compute_bs_iv_for_chains(opt_chains_raw, opt_spot,
                                              r=skew_rfr, q=skew_div)

    if not opt_chains:
        st.warning("No se pudo calcular IV BS para ningún vencimiento. "
                   "Ajusta r/q o amplía el rango de strikes.")
        st.stop()

    spot_disp = f"${opt_spot:.2f}"
    n_valid   = len(opt_chains)

    # ── Métricas de skew ─────────────────────────────────────
    sk = compute_skew_metrics(opt_chains, opt_spot)

    def _fmt(v, sfx="", sign=False):
        return f"{'+' if sign and v>=0 else ''}{v:.2f}{sfx}" if v is not None else "—"

    rr_raw = sk.get("rr25"); pc_raw = sk.get("pc_ratio")
    rr_clr = "var(--r)" if (rr_raw and rr_raw < -3) else "var(--y)" if (rr_raw and rr_raw < 0) else "var(--g)"
    pc_clr = "var(--r)" if (pc_raw and pc_raw > 1.5) else "var(--y)" if (pc_raw and pc_raw > 1.0) else "var(--g)"

    st.markdown(f"""
    <div class="mrow">
        <div class="mpill"><div class="ml">{skew_ticker} Spot</div><div class="mv nt">{spot_disp}</div></div>
        <div class="mpill"><div class="ml">ATM IV BS · {sk.get('exp','—')} ({sk.get('dte','?')}d)</div>
            <div class="mv nt">{_fmt(sk.get('atm_iv'),'%')}</div></div>
        <div class="mpill"><div class="ml">25Δ Risk Reversal</div>
            <div class="mv" style="color:{rr_clr}">{_fmt(sk.get('rr25'),' pts',True)}</div></div>
        <div class="mpill"><div class="ml">25Δ Butterfly</div>
            <div class="mv nt">{_fmt(sk.get('bf25'),' pts',True)}</div></div>
        <div class="mpill"><div class="ml">Skew Slope /10%</div>
            <div class="mv nt">{_fmt(sk.get('skew_slope'),' pts')}</div></div>
        <div class="mpill"><div class="ml">P/C Vol Ratio</div>
            <div class="mv" style="color:{pc_clr}">{_fmt(sk.get('pc_ratio'))}</div></div>
        <div class="mpill"><div class="ml">Vencimientos BS</div>
            <div class="mv nt">{n_valid}</div></div>
        <div class="mpill"><div class="ml">r / q</div>
            <div class="mv nt">{skew_rfr:.1%} / {skew_div:.1%}</div></div>
    </div>
    """, unsafe_allow_html=True)

    interp_parts = []
    if rr_raw is not None:
        if rr_raw < -4: interp_parts.append("**Risk Reversal muy negativo** — put skew extremo, fear elevado")
        elif rr_raw < -2: interp_parts.append("**Risk Reversal negativo moderado** — demanda de cobertura activa")
        else: interp_parts.append("**Risk Reversal neutro** — apetito por riesgo presente")
    if pc_raw is not None:
        if pc_raw > 1.5: interp_parts.append("**P/C Ratio > 1.5** — flujo dominante en puts, hedging institucional")
        elif pc_raw > 1.0: interp_parts.append("**P/C Ratio > 1.0** — ligero sesgo defensivo")
        else: interp_parts.append("**P/C Ratio < 1.0** — flujo en calls, risk-on")
    if interp_parts:
        with st.expander("Lectura del Skew", expanded=True):
            for l in interp_parts: st.markdown(l)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Skew Curves + ATM Term Structure ─────────────────────
    col_sk, col_atm = st.columns([1.6, 1])
    with col_sk:
        try:
            fig_sk = build_skew_curves(opt_chains, opt_spot,
                                       moneyness_range=(mon_lo, mon_hi),
                                       y_mode=y_mode,
                                       r=skew_rfr, q=skew_div)
            if fig_sk.data:
                st.plotly_chart(fig_sk, width="stretch",
                                config=dict(displayModeBar=True,
                                            modeBarButtonsToRemove=["lasso2d","select2d"]))
            else: st.info("No hay suficientes datos para graficar el skew.")
        except Exception as e: st.error(f"Error skew: {e}")

    with col_atm:
        try:
            fig_atm = build_atm_term_structure(opt_chains, opt_spot)
            if fig_atm.data:
                st.plotly_chart(fig_atm, width="stretch", config=dict(displayModeBar=False))
            else: st.info("No hay datos ATM.")
        except Exception as e: st.error(f"Error ATM TS: {e}")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── IV Surface ────────────────────────────────────────────
    if view_mode == "3D Surface":
        try:
            fig_surf = build_iv_surface(opt_chains, opt_spot,
                                        moneyness_range=(mon_lo, mon_hi),
                                        y_mode=y_mode)
            if fig_surf.data:
                st.plotly_chart(fig_surf, width="stretch", config=dict(displayModeBar=True))
            else: st.info("No hay suficientes datos para la superficie 3D.")
        except Exception as e: st.error(f"Error IV Surface: {e}")
    else:
        try:
            fig_hm = build_iv_heatmap(opt_chains, opt_spot,
                                      moneyness_range=(mon_lo, mon_hi))
            if fig_hm.data:
                st.plotly_chart(fig_hm, width="stretch", config=dict(displayModeBar=False))
            else: st.info("No hay suficientes datos para el heatmap.")
        except Exception as e: st.error(f"Error IV Heatmap: {e}")

    # ── Tabla por vencimiento ─────────────────────────────────
    with st.expander("Tabla resumen por vencimiento"):
        rows_tbl = []
        for exp_str, data in sorted(opt_chains.items(), key=lambda x: x[1]["dte"]):
            dte_t = data["dte"]
            puts_t  = data["puts"];  calls_t = data["calls"]
            atm_all = pd.concat([
                puts_t[puts_t["moneyness"].between(0.97,1.03)],
                calls_t[calls_t["moneyness"].between(0.97,1.03)],
            ])
            atm_iv_t = (float(np.average(atm_all["iv"].values,
                                          weights=atm_all["openInterest"].values+1))*100
                        if not atm_all.empty and "iv" in atm_all.columns else np.nan)
            p90  = puts_t[puts_t["moneyness"].between(0.88,0.92)]["iv"].mean()
            c110 = calls_t[calls_t["moneyness"].between(1.08,1.12)]["iv"].mean()
            rr_t = (c110-p90)*100 if pd.notna(p90) and pd.notna(c110) else np.nan
            rows_tbl.append({
                "Vencimiento": exp_str, "DTE": dte_t,
                "ATM IV (BS)": f"{atm_iv_t:.1f}%" if not np.isnan(atm_iv_t) else "—",
                "IV 90% put":  f"{p90*100:.1f}%"  if pd.notna(p90)  else "—",
                "IV 110% call":f"{c110*100:.1f}%" if pd.notna(c110) else "—",
                "RR ~25Δ":     f"{rr_t:+.1f} pts" if not np.isnan(rr_t) else "—",
                "Puts": len(puts_t), "Calls": len(calls_t),
            })
        if rows_tbl:
            st.dataframe(pd.DataFrame(rows_tbl), width="stretch", hide_index=True)

    st.caption(
        f"IV calculada con Black-Scholes (Brent) · r={skew_rfr:.1%} · q={skew_div:.1%} · "
        f"Spot {skew_ticker}: {spot_disp} · {now_cdmx().strftime('%H:%M:%S')} CDMX"
    )

    # ════════════════════════════════════════════════════════
    # SECCIÓN: VOL SURFACE FORECAST + SELLING RECOMMENDATIONS
    # ════════════════════════════════════════════════════════
    st.markdown("<div style='border-top:2px solid #F5A623;margin:0.8rem 0 0.4rem'></div>",
                unsafe_allow_html=True)
    st.markdown("## Forecast de Superficie + Oportunidades de Venta de Vol")
    st.markdown("""
    <div style="font-family:'JetBrains Mono';font-size:0.72rem;color:#9AA1A9;margin-bottom:0.6rem">
    <b>Modelo: SVI (Stochastic Volatility Inspired)</b> — Gatheral (2004) ·
    Fitea el smile actual por vencimiento con 5 parámetros (a, b, ρ, m, σ) que satisfacen
    condiciones de no-arbitraje. Luego proyecta el smile del día siguiente según un escenario
    de cambio de IV y calcula el <b>P&L esperado por vender prima</b>.
    </div>""", unsafe_allow_html=True)

    col_fc1, col_fc2, col_fc3 = st.columns([1, 1, 1])
    with col_fc1:
        iv_scenario = st.slider(
            "Escenario: cambio de IV (%)", -40, 10, -15,
            help="-15% = mean-reversion típica de un día con VIX elevado · 0% = sin cambio")
    with col_fc2:
        fc_view = st.selectbox("Vista superficie", ["IV Drop (oportunidad)", "IV Forecasted"],
                                help="Drop = cuánto se derrite la IV · Forecast = nivel esperado")
    with col_fc3:
        fc_exp_sel = st.selectbox(
            "Vencimiento para smile chart",
            sorted(opt_chains.keys(), key=lambda x: opt_chains[x]["dte"]),
            format_func=lambda x: f"{x} ({opt_chains[x]['dte']}d)",
            help="Vencimiento a mostrar en el chart de smile SVI")

    with st.spinner("Fittando SVI y calculando forecast…"):
        fc_result = forecast_vol_surface(
            opt_chains, opt_spot,
            r=skew_rfr, q=skew_div,
            iv_change_pct=iv_scenario/100,
        )

    if not fc_result:
        st.warning("No se pudo fittar el modelo SVI. Se necesitan ≥5 strikes por vencimiento.")
    else:
        svi_fits = fc_result.get("svi_fits", {})
        sell_df  = fc_result.get("sell_df", pd.DataFrame())

        # ── SVI model params ─────────────────────────────────
        with st.expander("Parámetros SVI por vencimiento", expanded=False):
            st.markdown("""
**Guía de lectura:**
- **ρ (rho)**: asimetría del smile. ρ < 0 = put skew dominante (normal en equity). Cuanto más negativo, más pronunciado el skew bajista.
- **b**: pendiente/curvatura total. b alto = smile muy curvado (vol de cola elevada).
- **a**: nivel base de varianza implícita.
- **σ**: suavidad ATM. σ bajo = smile más pronunciado en el dinero.
- **R²**: calidad del fit (>0.90 = excelente, >0.70 = usable).
            """)
            rows_svi = []
            for exp, fit in sorted(svi_fits.items(), key=lambda x: x[1]["dte"]):
                rows_svi.append({
                    "Vencimiento": exp, "DTE": fit["dte"],
                    "a": round(fit["a"],4), "b": round(fit["b"],4),
                    "ρ (rho)": round(fit["rho"],3),
                    "m": round(fit["m"],4), "σ": round(fit["sigma"],4),
                    "R²": round(fit.get("r2",0),3),
                })
            if rows_svi:
                st.dataframe(pd.DataFrame(rows_svi), width="stretch", hide_index=True)

        # ── Smile chart: actual vs forecasted ────────────────
        col_sm1, col_sm2 = st.columns([1.4, 1])
        with col_sm1:
            try:
                fig_smile = build_svi_smile_chart(fc_result, fc_exp_sel, opt_spot)
                if fig_smile.data:
                    st.plotly_chart(fig_smile, width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error smile chart: {e}")
        with col_sm2:
            # Resumen del fit para el vencimiento seleccionado
            fit_sel = svi_fits.get(fc_exp_sel, {})
            if fit_sel:
                rho_s = fit_sel.get("rho", 0)
                b_s   = fit_sel.get("b",   0)
                r2_s  = fit_sel.get("r2",  0)
                skew_interp = (
                    "Put skew muy pronunciado — alta demanda de protección" if rho_s < -0.4
                    else "Put skew moderado — skew normal de equity" if rho_s < -0.2
                    else "Smile casi simétrico — mercado tranquilo"
                )
                st.markdown(f"""
                <div class="icard" style="margin-top:1.5rem">
                    <div class="ic-title">SVI {fc_exp_sel}</div>
                    <div class="ic-row"><span class="ic-label">ρ (asimetría)</span>
                        <span class="ic-val">{rho_s:.3f}</span></div>
                    <div class="ic-row"><span class="ic-label">b (curvatura)</span>
                        <span class="ic-val">{b_s:.4f}</span></div>
                    <div class="ic-row"><span class="ic-label">R² del fit</span>
                        <span class="ic-val" style="color:{'var(--g)' if r2_s>0.85 else 'var(--y)'}">{r2_s:.3f}</span></div>
                    <div class="ic-row"><span class="ic-label">Escenario</span>
                        <span class="ic-val">{iv_scenario:+.0f}% IV</span></div>
                    <div class="ic-row" style="margin-top:0.4rem">
                        <span style="font-size:0.78rem;color:var(--t)">{skew_interp}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Superficie forecasted ─────────────────────────────
        try:
            fc_view_key = "drop" if "Drop" in fc_view else "forecast"
            fig_fc_surf = build_forecast_surface_chart(fc_result, opt_spot, view=fc_view_key)
            if fig_fc_surf.data:
                st.plotly_chart(fig_fc_surf, width="stretch", config=dict(displayModeBar=True))
        except Exception as e:
            st.error(f"Error forecast surface: {e}")

        # ── Selling recommendations ───────────────────────────
        st.markdown("### Opciones Candidatas para Vender Prima")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono';font-size:0.72rem;color:#9AA1A9;margin-bottom:0.5rem">
        Ordenadas por <b>P&L esperado</b> si la IV cae <b>{iv_scenario:+.0f}%</b> hacia el nivel SVI forecasted.
        P&L = Vega × ΔIV · Solo muestra opciones OTM dentro de ±22% del spot con OI > 0.
        </div>""", unsafe_allow_html=True)

        if not sell_df.empty:
            # Top 20
            top_sell = sell_df.head(20).copy()
            # Color coding
            def _style_sell(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for i in df.index:
                    if df.loc[i,"Tipo"] == "PUT":
                        styles.loc[i,"Tipo"] = "color: #9AA1A9"
                    else:
                        styles.loc[i,"Tipo"] = "color: #C9821A"
                    if df.loc[i,"P&L Esp. ($)"] > 50:
                        styles.loc[i,"P&L Esp. ($)"] = "color: #16C784; font-weight:bold"
                    elif df.loc[i,"P&L Esp. ($)"] > 20:
                        styles.loc[i,"P&L Esp. ($)"] = "color: #F5A623"
                return styles

            st.dataframe(
                top_sell.style.apply(_style_sell, axis=None).format({
                    "Strike": "${:.0f}",
                    "Dist Spot %": "{:+.1f}%",
                    "IV Actual %": "{:.1f}%",
                    "IV Forecast %": "{:.1f}%",
                    "IV Drop pts": "{:.2f}",
                    "Vega/ct ($)": "${:.0f}",
                    "P&L Esp. ($)": "${:.0f}",
                    "Mid $": "${:.2f}",
                }),
                width="stretch", hide_index=True
            )

            # ── Best trade summary ────────────────────────────
            best = sell_df.iloc[0]
            st.markdown(f"""
            <div style="background:var(--gbg);border:1px solid var(--g);border-radius:6px;
                        padding:0.7rem 1rem;margin-top:0.5rem">
                <div style="font-family:Inter;font-weight:800;font-size:1rem;color:var(--g)">
                    MEJOR CANDIDATO</div>
                <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#F4F5F6;margin-top:0.3rem">
                    <b>Vender {best['Tipo']} K=${best['Strike']:.0f} exp {best['Exp']} ({best['DTE']}d)</b>
                    · Dist spot: {best['Dist Spot %']:+.1f}%<br>
                    IV actual: {best['IV Actual %']:.1f}% → IV forecast: {best['IV Forecast %']:.1f}%
                    → Drop esperado: <b>{best['IV Drop pts']:.2f} pts</b><br>
                    Mid: ${best['Mid $']:.2f} · OI: {best['OI']:,} · Vega/ct: ${best['Vega/ct ($)']:.0f}
                    · <b>P&L esperado: ${best['P&L Esp. ($)']:.0f}/contrato</b>
                </div>
                <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#9AA1A9;margin-top:0.3rem">
                Solo análisis educativo. No es recomendación financiera.
                El P&L esperado asume que la IV cae exactamente {iv_scenario:+.0f}% — nada garantizado.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No se encontraron candidatos con P&L positivo bajo el escenario seleccionado.")

    st.caption(
        f"SVI: Gatheral (2004) · No-arbitrage butterfly condition: b(1+|ρ|)≤2 · "
        f"Vega = S·√T·N'(d₁)·100 · Opciones con OI>0 en ±22% del spot"
    )


def _vista_gex():

    st.markdown('<p class="stc-note"><b style="color:var(--white)">GEX</b> mide el gamma neto de los '
                'dealers por strike. Positivo: compran caídas y venden subidas, el mercado queda anclado. '
                'Negativo: venden caídas y compran subidas, los movimientos se amplifican.</p>',
                unsafe_allow_html=True)

    # ── Controles ─────────────────────────────────────────────
    col_g1, col_g2, col_g3, col_g4 = st.columns([1, 1, 1, 1])
    with col_g1:
        gex_ticker = st.selectbox(
            "Subyacente", ["SPY", "QQQ", "IWM", "GLD", "TLT"],
            index=0, key="gex_ticker_sel",
            help="SPY tiene el mayor OI de opciones = GEX más fiable",
        )
    with col_g2:
        gex_n_exp = st.slider("Vencimientos a incluir", 1, 6, 3,
                              help="Más vencimientos = GEX más completo pero más lento",
                              key="gex_n_exp")
    with col_g3:
        gex_rfr = st.number_input("Risk-Free Rate (r)", 0.0, 0.15,
                                   float(get_risk_free_rate()),
                                   step=0.001, format="%.3f", key="gex_rfr",
                                   help="Live desde ^IRX; edita si deseas")
    with col_g4:
        gex_div = st.number_input("Dividend Yield (q)", 0.0, 0.10,
                                   float(get_dividend_yield("SPY")),
                                   step=0.001, format="%.3f", key="gex_div",
                                   help="SPY TTM dividend yield live")

    col_g5, col_g6 = st.columns([2, 1])
    with col_g5:
        gex_range = st.slider("Rango de strikes mostrado (±% del spot)", 5, 20, 12,
                              help="Porcentaje del spot hacia arriba y abajo del strike central")
    with col_g6:
        if st.button("Actualizar GEX", key="btn_refresh_gex"):
            fetch_options_chains.clear()
            st.rerun()

    # ── Datos: reusar cache de opciones si el ticker/n_exp coincide ──────
    with st.spinner(f"Cargando opciones {gex_ticker} ({gex_n_exp} vencimientos)…"):
        gex_chains_raw, gex_spot = fetch_options_chains(gex_ticker, n_exp=gex_n_exp)

    if not gex_chains_raw or not gex_spot:
        st.error(f"**Yahoo Finance rate-limited** — no se cargaron opciones de **{gex_ticker}**.")
        st.info(
            "Streamlit Cloud comparte IP con miles de apps, Yahoo aplica rate limits.\n\n"
            "**Qué puedes hacer:**\n"
            "- Esperar 15-30 min (cache negativo automático)\n"
            "- Reducir a 1-2 vencimientos y reintentar\n"
            "- Intentar en horario de mercado (9:30-16:00 ET)\n"
            "- Ejecutar la app localmente"
        )
        if st.button("Reintentar ahora (limpia cache negativo)", key="btn_gex_retry"):
            for k in list(st.session_state.keys()):
                if k.startswith("_opts_fail_"):
                    del st.session_state[k]
            fetch_options_chains.clear()
            st.rerun()
        st.stop()

    # Calcular IV BS (necesaria para gamma precisa)
    with st.spinner("Calculando Gamma (Black-Scholes)…"):
        gex_chains_bs = compute_bs_iv_for_chains(
            gex_chains_raw, gex_spot, r=gex_rfr, q=gex_div
        )
        # Si BS falla, usar chains raw con impliedVolatility de yfinance
        gex_chains_use = gex_chains_bs if gex_chains_bs else gex_chains_raw

    # ── Calcular GEX profile ──────────────────────────────────
    gex_df = compute_gex_profile(
        gex_chains_use, gex_spot, r=gex_rfr, q=gex_div
    )
    gex_summary = compute_gex_summary(gex_df, gex_spot)

    if gex_df.empty or not gex_summary:
        st.warning("No hay suficientes datos de opciones para calcular el GEX.")
        st.stop()

    # ── Métricas ─────────────────────────────────────────────
    total_g   = gex_summary.get("total_gex", 0)
    flip_s    = gex_summary.get("flip_strike")
    call_w    = gex_summary.get("call_wall")
    put_w     = gex_summary.get("put_wall")
    pct_pos   = gex_summary.get("pct_pos_otm")
    regime    = gex_summary.get("regime", "?")
    reg_clr   = "var(--g)" if regime == "POSITIVE" else "var(--r)"

    flip_dist = f"{(flip_s/gex_spot - 1)*100:+.1f}%" if flip_s else "N/A"
    cw_dist   = f"{(call_w/gex_spot - 1)*100:+.1f}%" if call_w else "N/A"
    pw_dist   = f"{(put_w/gex_spot - 1)*100:+.1f}%" if put_w else "N/A"

    st.markdown(f"""
    <div class="mrow">
        <div class="mpill" style="min-width:160px">
            <div class="ml">Régimen GEX</div>
            <div class="mv" style="color:{reg_clr}">{regime}</div>
        </div>
        <div class="mpill">
            <div class="ml">Net GEX Total</div>
            <div class="mv {'up' if total_g>=0 else 'dn'}">${total_g:+.1f}M</div>
        </div>
        <div class="mpill">
            <div class="ml">{gex_ticker} Spot</div>
            <div class="mv nt">${gex_spot:.2f}</div>
        </div>
        <div class="mpill">
            <div class="ml">Gamma Flip</div>
            <div class="mv" style="color:var(--y)">${f"{flip_s:.0f}" if flip_s else "—"} <span style="font-size:0.75rem;color:var(--dim)">({flip_dist})</span></div>
        </div>
        <div class="mpill">
            <div class="ml">Call Wall</div>
            <div class="mv" style="color:#C9821A">${f"{call_w:.0f}" if call_w else "—"} <span style="font-size:0.75rem;color:var(--dim)">({cw_dist})</span></div>
        </div>
        <div class="mpill">
            <div class="ml">Put Wall</div>
            <div class="mv" style="color:#9AA1A9">${f"{put_w:.0f}" if put_w else "—"} <span style="font-size:0.75rem;color:var(--dim)">({pw_dist})</span></div>
        </div>
        <div class="mpill">
            <div class="ml">%OTM con GEX+</div>
            <div class="mv nt">{f"{pct_pos:.0f}%" if pct_pos is not None else "—"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interpretación ────────────────────────────────────────
    with st.expander("Interpretación GEX", expanded=True):
        if regime == "POSITIVE":
            st.markdown(
                f"**Régimen POSITIVO** — Los dealers tienen gamma larga neta. "
                f"Cuando el mercado cae, los dealers **compran** para delta-hedgear → actúa como soporte. "
                f"Cuando el mercado sube, los dealers **venden** → actúa como resistencia. "
                f"Resultado: **volatilidad comprimida**, el mercado tiende a mantenerse anclado cerca del gamma flip (${flip_s:.0f} aprox.)."
                if flip_s else
                f"**Régimen POSITIVO** — Los dealers tienen gamma larga neta → mercado estabilizador."
            )
        else:
            st.markdown(
                f"**Régimen NEGATIVO** — Los dealers tienen gamma corta neta. "
                f"Cuando el mercado cae, los dealers **también venden** para hedgear → los movimientos se **amplifican**. "
                f"El mercado puede tener gaps y movimientos bruscos. "
                f"{'El Gamma Flip está en $' + str(int(flip_s)) + ' — recuperar ese nivel sería la señal de estabilización.' if flip_s else ''}"
            )
        st.markdown("""
**Niveles clave:**
- **Call Wall**: mayor concentración de gamma de calls — los dealers venden agresivamente aquí (techo)
- **Put Wall**: mayor concentración de gamma de puts — los dealers compran agresivamente aquí (soporte)
- **Gamma Flip**: punto donde el régimen cambia de positivo a negativo
        """)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ── Chart 1: GEX Profile (principal) ────────────────────
    try:
        fig_gex = build_gex_profile_chart(gex_df, gex_spot, gex_summary,
                                           ticker=gex_ticker,
                                           strike_range_pct=gex_range/100)
        if fig_gex.data:
            st.plotly_chart(fig_gex, width="stretch",
                            config=dict(displayModeBar=True,
                                        modeBarButtonsToRemove=["lasso2d","select2d"]))
    except Exception as e:
        st.error(f"Error GEX profile: {e}")

    # ── Chart 2: Expected Move + GEX levels ──────────────────
    try:
        fig_em = build_gex_expected_move_chart(gex_df, gex_chains_use, gex_spot,
                                                strike_range_pct=gex_range/100)
        if fig_em.data:
            st.plotly_chart(fig_em, width="stretch", config=dict(displayModeBar=False))
    except Exception as e:
        st.error(f"Error Expected Move: {e}")

    # ── Charts 3 & 4: DEX + Cumulative GEX ──────────────────
    col_dex, col_cum = st.columns(2)
    with col_dex:
        try:
            fig_dex = build_gex_delta_exposure_chart(gex_chains_use, gex_spot,
                                                      r=gex_rfr, q=gex_div,
                                                      strike_range_pct=gex_range/100)
            if fig_dex.data:
                st.plotly_chart(fig_dex, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error DEX: {e}")
    with col_cum:
        try:
            fig_cum = build_gex_cumulative_chart(gex_df, gex_spot,
                                                  strike_range_pct=gex_range/100)
            if fig_cum.data:
                st.plotly_chart(fig_cum, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error GEX Acumulado: {e}")

    # ── Charts 5 & 6: Vanna/Charm + GEX por vencimiento ─────
    col_vc, col_exp = st.columns(2)
    with col_vc:
        try:
            fig_vc = build_gex_vanna_charm_chart(gex_chains_use, gex_spot,
                                                   r=gex_rfr, q=gex_div,
                                                   strike_range_pct=gex_range/100)
            if fig_vc.data:
                st.plotly_chart(fig_vc, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error Vanna/Charm: {e}")
    with col_exp:
        try:
            fig_gex_exp = build_gex_by_expiry_chart(gex_chains_use, gex_spot,
                                                     r=gex_rfr, q=gex_div)
            if fig_gex_exp.data:
                st.plotly_chart(fig_gex_exp, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error GEX por vencimiento: {e}")

    # ── Tabla de strikes clave ────────────────────────────────
    with st.expander("Top strikes por |GEX|"):
        lo_rng = gex_spot * (1 - gex_range/100)
        hi_rng = gex_spot * (1 + gex_range/100)
        top_strikes = (
            gex_df[gex_df["strike"].between(lo_rng, hi_rng)]
            .assign(abs_gex=lambda x: x["net_gex"].abs())
            .nlargest(15, "abs_gex")
            .sort_values("strike")
            [["strike","calls_gex","puts_gex","net_gex"]]
            .rename(columns={"strike":"Strike","calls_gex":"GEX Calls ($M)",
                              "puts_gex":"GEX Puts ($M)","net_gex":"Net GEX ($M)"})
        )
        if not top_strikes.empty:
            st.dataframe(top_strikes.style.format({
                "Strike":"${:.0f}","GEX Calls ($M)":"${:.2f}M",
                "GEX Puts ($M)":"${:.2f}M","Net GEX ($M)":"${:+.2f}M"}),
                width="stretch", hide_index=True)

    st.caption(
        f"GEX/DEX = OI × Greeks_BS × S² × 100 · "
        f"Vanna = ∂Δ/∂σ · Charm = ∂Δ/∂t · "
        f"Max Pain = mínima pérdida agregada de holders · "
        f"r={gex_rfr:.1%} q={gex_div:.1%} · {now_cdmx().strftime('%H:%M:%S')} CDMX"
    )




def page_estrategia():
    strategy_page.render(live={"r1": r_m2m1_live, "r2": r_vix3m_live,
                               "ts": (vix3m_live or {}).get("timestamp")})


def page_opciones():
    st.markdown(f"""{T.eyebrow("Opciones SPY")}
        <div class="stc-h1">Skew, superficie y gamma</div>
        <p class="stc-lead">Volatilidad implícita por strike y vencimiento, y el posicionamiento de gamma
        de los dealers. Depende de Yahoo Finance, que limita peticiones en Streamlit Cloud.</p>""",
                unsafe_allow_html=True)
    vista = st.segmented_control("Vista", options=["Skew y superficie", "Gamma exposure"],
                                 default="Skew y superficie", key="opciones_vista") or "Skew y superficie"
    if vista == "Skew y superficie":
        _vista_skew()
    else:
        _vista_gex()


def page_metodologia():
    methodology_page.render()


_pg = st.navigation([
    st.Page(page_estrategia, title="Estrategia", url_path="estrategia", default=True),
    st.Page(page_curva, title="Curva", url_path="curva"),
    st.Page(page_regimen, title="Régimen", url_path="regimen"),
    st.Page(page_volatilidad, title="Volatilidad", url_path="volatilidad"),
    st.Page(page_opciones, title="Opciones", url_path="opciones"),
    st.Page(page_metodologia, title="Metodología", url_path="metodologia"),
], position="top")
_pg.run()

st.markdown(f"""<div class="stc-foot"><span>SPREAD TRADING CLUB · VIX CONTROLLER</span>
<span>Donde el riesgo se define.</span>
<span>Herramienta de seguimiento · no es asesoramiento de inversión</span></div>""",
            unsafe_allow_html=True)
