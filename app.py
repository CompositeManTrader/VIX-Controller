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
from datetime import datetime, date
import re, time, warnings, logging
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
from vix_controller.quant import term_structure as tsig
from vix_controller.quant import vix_inverse_live as _vl
from vix_controller.data import cboe as _cboe
from vix_controller import alerts as _alerts
from vix_controller.ui import theme as T
from vix_controller.ui import strategy_page, methodology_page, vol_page

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
                                     days: int | None = None) -> go.Figure:
    """
    Cómo ha cambiado la lectura de la curva en el tiempo:
      panel 1: M1 y VIX (la base)
      panel 2: franja con la señal de cada día (una sola traza: con 20 años
               de historia, sombrear con un rectángulo por tramo serían ~900 formas)
      panel 3: contango M1→M2 (%) y su percentil rolling
    """
    h = hist.tail(days) if days else hist
    c = curve.reindex(h.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.52, 0.07, 0.41], vertical_spacing=0.035,
                        specs=[[{}], [{}], [{"secondary_y": True}]],
                        subplot_titles=("<b>VIX y M1 (liquidación)</b>", "",
                                        "<b>Contango M1→M2 y percentil móvil de 5 años</b>"))
    idx = h.index
    fig.add_trace(go.Scatter(x=idx, y=c["m1"].tolist(), name="M1", mode="lines",
                             line=dict(color="#F5A623", width=1.4),
                             hovertemplate="M1 %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=c["VIX"].tolist(), name="VIX", mode="lines",
                             line=dict(color="#F4F5F6", width=1.0),
                             hovertemplate="VIX %{y:.2f}<extra></extra>"), row=1, col=1)
    code = h["signal"].map({tsig.SIGNAL_LONG: 0, tsig.SIGNAL_NEUTRAL: 1, tsig.SIGNAL_SHORT: 2})
    fig.add_trace(go.Heatmap(x=idx, y=["señal"], z=[code.tolist()], zmin=0, zmax=2,
                             colorscale=[[0, "#EA3943"], [0.33, "#EA3943"], [0.34, "#3A3F47"],
                                         [0.66, "#3A3F47"], [0.67, "#F5A623"], [1, "#F5A623"]],
                             showscale=False, customdata=[h["signal"].tolist()],
                             hovertemplate="%{customdata}<extra></extra>"), row=2, col=1)
    bar_clr = ["#16C784" if v > 0 else "#EA3943" for v in h["contango_pct"].fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=h["contango_pct"].tolist(), name="Contango M1→M2",
                         marker_color=bar_clr, marker_line_width=0, opacity=0.85,
                         hovertemplate="contango %{y:+.2f} %<extra></extra>"),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=idx, y=h["ct_pctile"].tolist(), name="Percentil 5 años",
                             mode="lines", line=dict(color="#9AA1A9", width=0.8 if len(h) > 1500 else 1.1,
                                                     dash="dot"), opacity=0.55 if len(h) > 1500 else 1.0,
                             hovertemplate="p%{y:.0f}<extra></extra>"),
                  row=3, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="#3A3F47", line_width=1, row=3, col=1, secondary_y=False)
    # Leyenda de la franja (x con fecha válida — bug de plotly.js con x vacío)
    for name, clr in (("Short vol favorable", "#F5A623"), ("Neutral", "#3A3F47"),
                      ("Long vol", "#EA3943")):
        fig.add_trace(go.Scatter(x=[idx[0]], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, symbol="square", color=clr),
                                 hoverinfo="skip"), row=1, col=1)
    fig.update_layout(
        template="stc", height=600, margin=dict(l=55, r=25, t=45, b=60), hovermode="x unified",
        bargap=0,
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center",
                    font=dict(size=10, color="#F4F5F6", family="JetBrains Mono")))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=11, color="#9AA1A9", family="Inter")
        ann["xanchor"] = "left"; ann["x"] = 0.01
    fig.update_xaxes(gridcolor="#1C1F24", tickfont=dict(size=10, color="#9AA1A9"))
    fig.update_yaxes(gridcolor="#1C1F24", tickfont=dict(size=9.5, color="#9AA1A9"))
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    fig.update_yaxes(ticksuffix=" %", row=3, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 100], showgrid=False, row=3, col=1, secondary_y=True,
                     title=dict(text="percentil", font=dict(size=9, color="#9AA1A9")))
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — HMM RÉGIMEN + CROSS-ASSET EARLY WARNING (helpers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIX INVERSE — gráficos del panel de seguimiento (modelo congelado)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━







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
            _todo = f"Todo · {_sh.index[0].year}"
            _ventanas = {"1 año": 252, "3 años": 756, "5 años": 1260, "10 años": 2520,
                         _todo: len(_sh)}
            _v_sh = st.segmented_control("Ventana del histórico de señal", options=list(_ventanas),
                                         default=_todo, key="sl_ts_hist_v") or _todo
            _n_days_sh = _ventanas[_v_sh]
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
                + (f" {int((~_vxh['gap_ok']).sum())} fechas de 2004-2006 en que CBOE aún no "
                   f"listaba todos los meses: el «M2» era el siguiente contrato listado, a dos o "
                   f"tres meses; se muestran tal cual."
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



def page_volatilidad():
    vol_page.render()


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
    st.Page(page_volatilidad, title="Volatilidad", url_path="volatilidad"),
    st.Page(page_opciones, title="Opciones", url_path="opciones"),
    st.Page(page_metodologia, title="Metodología", url_path="metodologia"),
], position="top")
_pg.run()

st.markdown(f"""<div class="stc-foot"><span>SPREAD TRADING CLUB · VIX CONTROLLER</span>
<span>Donde el riesgo se define.</span>
<span>Herramienta de seguimiento · no es asesoramiento de inversión</span></div>""",
            unsafe_allow_html=True)
