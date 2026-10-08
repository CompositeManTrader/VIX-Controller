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
from datetime import datetime
import re, time, warnings, logging

# ── Módulos quant propios (refactor) ──────────────────────────
from vix_controller import config as cfg
from vix_controller.quant import term_structure as tsig
from vix_controller.quant import vix_inverse_live as _vl
from vix_controller.data import cboe as _cboe
from vix_controller import alerts as _alerts
from vix_controller.ui import theme as T
from vix_controller.ui import strategy_page, methodology_page, vol_page, options_page

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
# CURVA EN VIVO — API JSON de CBOE
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
# ÍNDICES DE VOLATILIDAD — baro_history.parquet + últimos días de yfinance
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
# DATOS — parquets del repo (los actualizan las GitHub Actions)
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


def page_estrategia():
    strategy_page.render(live={"r1": r_m2m1_live, "r2": r_vix3m_live,
                               "ts": (vix3m_live or {}).get("timestamp")})


def page_opciones():
    options_page.render(vix_spot=vix_spot["price"] if vix_spot else None)


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
