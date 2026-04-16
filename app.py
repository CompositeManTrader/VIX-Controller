"""
VIX Controller — Bloomberg-Style Term Structure + Operational Monitor
Data: CBOE Delayed Quotes via Playwright (browser por llamada, install cacheado)
Auto-refresh: every 60 seconds via JS injection
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, date
from io import StringIO
import re, time, warnings, logging, os
from zoneinfo import ZoneInfo
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

CDMX_TZ = ZoneInfo("America/Mexico_City")
def now_cdmx():
    return datetime.now(CDMX_TZ)

st.set_page_config(page_title="VIX Controller", page_icon="🔴", layout="wide",
                   initial_sidebar_state="collapsed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLAYWRIGHT — solo verifica instalación UNA vez (no lanza browser aquí)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def check_playwright_installed() -> bool:
    """
    Instala Chromium si no existe y verifica que funcione.
    Se ejecuta UNA sola vez por deployment (cache_resource).
    """
    log = logging.getLogger("vix_controller")
    try:
        import subprocess
        result = subprocess.run(
            ["playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log.info("Playwright install chromium: OK")
        else:
            log.warning(f"Playwright install output: {result.stderr[:300]}")

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
            )
            browser.close()
        log.info("Playwright check: Chromium OK")
        return True
    except Exception as e:
        log.error(f"Playwright check failed: {e}")
        return False

pw_ready = check_playwright_installed()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOOMBERG CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0D1117;--card:#161B22;--border:#30363D;--g:#3FB950;--r:#F85149;--y:#D29922;--b:#58A6FF;--c:#39D2C0;--t:#C9D1D9;--dim:#8B949E;--w:#F0F6FC;--gbg:#0B2E13;--rbg:#3B1218;}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 1.5rem;max-width:1400px;}
.hdr{display:flex;align-items:center;padding:0.6rem 0;border-bottom:2px solid var(--border);margin-bottom:0.8rem;gap:1rem;}
.hdr .logo-box{display:flex;align-items:center;gap:0.6rem;}
.hdr .logo-icon{width:32px;height:32px;background:linear-gradient(135deg,#F7931A,#FF6B35);border-radius:4px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:14px;color:#0D1117;font-family:'Inter',sans-serif;letter-spacing:-0.5px;}
.hdr .logo-text{font-family:'Inter',sans-serif;font-weight:800;font-size:1.1rem;color:#F0F6FC;letter-spacing:0.8px;}
.hdr .logo-tag{font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:#F7931A;letter-spacing:1.5px;text-transform:uppercase;margin-top:1px;}
.hdr .sub{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--dim);margin-left:auto;text-align:right;line-height:1.4;}
.mrow{display:flex;gap:4px;margin-bottom:0.6rem;flex-wrap:wrap;}
.mpill{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:0.4rem 0.7rem;flex:1;min-width:120px;text-align:center;}
.mpill .ml{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.6px;}
.mpill .mv{font-family:'Inter',sans-serif;font-weight:700;font-size:1.15rem;}
.mv.up{color:var(--g);}.mv.dn{color:var(--r);}.mv.nt{color:var(--b);}
.ctx{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.78rem;margin:0.4rem 0;}
.ctx td,.ctx th{padding:0.35rem 0.5rem;text-align:center;border:1px solid var(--border);}
.ctx th{background:#1C2128;color:var(--dim);font-weight:500;font-size:0.65rem;text-transform:uppercase;}
.ctx .pos{color:var(--g);}.ctx .neg{color:var(--r);}
.ctx .hdr-cell{background:var(--card);color:var(--t);font-weight:600;text-align:left;width:120px;}
.dtbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.75rem;margin-top:0.5rem;}
.dtbl th{color:var(--b);font-weight:500;padding:0.4rem 0.6rem;border-bottom:1px solid var(--border);font-size:0.62rem;text-transform:uppercase;letter-spacing:0.5px;text-align:center;}
.dtbl td{padding:0.35rem 0.6rem;text-align:center;color:var(--t);border-bottom:1px solid rgba(255,255,255,0.03);}
.dtbl tr:hover td{background:rgba(88,166,255,0.04);}
.sig-box{border-radius:6px;padding:1rem;text-align:center;border-width:2px;border-style:solid;}
.sig-long{background:var(--gbg);border-color:var(--g);}
.sig-cash{background:var(--rbg);border-color:var(--r);}
.sig-box .sl{font-family:'Inter',sans-serif;font-weight:800;font-size:2rem;}
.sig-box .sd{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-top:2px;}
.chk{display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--t);}
.chk .ok{color:var(--g);font-weight:700;}.chk .no{color:var(--r);font-weight:700;}
.icard{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;}
.icard .ic-title{font-family:'Inter',sans-serif;font-weight:700;font-size:0.85rem;color:var(--w);margin-bottom:0.5rem;border-bottom:1px solid var(--border);padding-bottom:0.3rem;}
.icard .ic-row{display:flex;justify-content:space-between;padding:0.2rem 0;font-family:'JetBrains Mono',monospace;font-size:0.8rem;}
.icard .ic-label{color:var(--dim);}.icard .ic-val{color:var(--t);font-weight:500;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-weight:600;font-size:0.82rem;color:var(--dim);padding:0.5rem 1.5rem;}
.stTabs [aria-selected="true"]{color:#F7931A !important;border-bottom:2px solid #F7931A !important;}
[data-testid="stSidebar"]{background:var(--card);}
</style>
""", unsafe_allow_html=True)

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
@st.cache_data(ttl=55)
def scrape_cboe_futures() -> pd.DataFrame:
    """
    Lanza Chromium, scrapea, cierra — todo en el mismo thread.
    Cache de 55s evita relanzar el browser en cada rerun de Streamlit.
    El check_playwright_installed() ya validó que Chromium existe.
    """
    log = logging.getLogger("vix_controller")

    if not pw_ready:
        log.error("CBOE_SCRAPE: Playwright no disponible")
        st.session_state["scrape_debug"] = "❌ Playwright/Chromium no instalado"
        return pd.DataFrame()

    from playwright.sync_api import sync_playwright

    html = ""
    try:
        log.info("CBOE_SCRAPE: lanzando Chromium...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
                      '--disable-extensions', '--no-first-run'],
            )
            page = browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
            )
            # Bloquear solo trackers — no CSS ni JS de CBOE
            page.route("**/googletagmanager**", lambda r: r.abort())
            page.route("**/google-analytics**", lambda r: r.abort())
            page.route("**/doubleclick**",       lambda r: r.abort())

            log.info("CBOE_SCRAPE: navegando...")
            page.goto(
                'https://www.cboe.com/delayed_quotes/futures/future_quotes',
                wait_until='networkidle', timeout=45000
            )

            # Esperar texto VX/ en página
            try:
                page.wait_for_function(
                    "() => document.body.innerText.includes('VX/')",
                    timeout=25000
                )
                log.info("CBOE_SCRAPE: VX/ detectado ✅")
            except Exception:
                log.warning("CBOE_SCRAPE: VX/ no apareció en 25s — tomando HTML igual")

            html = page.content()
            browser.close()

        vx_n = html.count('VX/')
        log.info(f"CBOE_SCRAPE: HTML {len(html):,} chars · VX/ hits: {vx_n}")
        st.session_state["scrape_debug"] = (
            f"HTML: {len(html):,} chars · 'VX/' en HTML: {vx_n} · "
            f"{now_cdmx().strftime('%H:%M:%S')}"
        )

    except Exception as e:
        log.error(f"CBOE_SCRAPE: error — {e}")
        st.session_state["scrape_debug"] = f"❌ Error: {e}"
        return pd.DataFrame()

    # Parsear tablas HTML
    try:
        all_tables = pd.read_html(StringIO(html))
        log.info(f"CBOE_SCRAPE: {len(all_tables)} tablas en HTML")
    except Exception as e:
        log.error(f"CBOE_SCRAPE: read_html error — {e}")
        st.session_state["scrape_debug"] += f" | read_html error: {e}"
        return pd.DataFrame()

    df_vx = pd.DataFrame()
    table_info = []
    for i, df in enumerate(all_tables):
        cols_upper = [str(c).upper().strip() for c in df.columns]
        table_info.append(f"T{i}:{cols_upper[:4]}")
        if 'SYMBOL' in cols_upper and 'EXPIRATION' in cols_upper:
            sym_col = df.columns[cols_upper.index('SYMBOL')]
            if df[sym_col].astype(str).str.startswith('VX').any():
                df_vx = df.copy()
                log.info(f"CBOE_SCRAPE: tabla VX en índice {i} ✅")
                break

    st.session_state["scrape_debug"] += f" | {len(all_tables)} tables: {' '.join(table_info[:4])}"

    if df_vx.empty:
        log.warning("CBOE_SCRAPE: tabla VX no encontrada")
        st.session_state["scrape_html_sample"] = html[1500:2500]
        return pd.DataFrame()

    df_vx.columns = [str(c).strip().upper() for c in df_vx.columns]
    rename = {
        'SYMBOL': 'Symbol', 'EXPIRATION': 'Expiration',
        'LAST': 'Last', 'CHANGE': 'Change',
        'HIGH': 'High', 'LOW': 'Low',
        'SETTLEMENT': 'Settlement', 'VOLUME': 'Volume',
    }
    df_vx.rename(columns={k: v for k, v in rename.items() if k in df_vx.columns}, inplace=True)

    if 'Symbol' in df_vx.columns:
        mask = df_vx['Symbol'].astype(str).str.match(r'^VX/[A-Z]\d+$')
        df_vx = df_vx[mask].reset_index(drop=True)

    if 'Expiration' in df_vx.columns:
        df_vx['Expiration'] = pd.to_datetime(df_vx['Expiration'], errors='coerce')
        df_vx = df_vx.sort_values('Expiration').reset_index(drop=True)

    for col in ['Last', 'Change', 'High', 'Low', 'Settlement', 'Volume']:
        if col in df_vx.columns:
            df_vx[col] = pd.to_numeric(
                df_vx[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            )

    today = pd.Timestamp(now_cdmx().date()).normalize()
    if 'Expiration' in df_vx.columns:
        df_vx['DTE'] = (df_vx['Expiration'] - today).dt.days

    df_vx['Price'] = df_vx.apply(
        lambda r: r['Last'] if pd.notna(r.get('Last')) and r.get('Last', 0) > 0
                  else r.get('Settlement', 0), axis=1
    )
    df_vx['Scraped_At'] = now_cdmx().strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"CBOE_SCRAPE: {len(df_vx)} contratos mensuales ✅")
    return df_vx


@st.cache_data(ttl=55)
def fetch_vix_spot():
    try:
        h = yf.Ticker("^VIX").history(period="5d")
        if not h.empty:
            c = round(float(h['Close'].iloc[-1]), 2)
            p = round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else c
            return dict(price=c, prev=p, chg=round(c - p, 2))
    except: pass
    return None


@st.cache_data(ttl=55)
def fetch_etps():
    out = {}
    for name, sym in [("VXX","VXX"),("SVXY","SVXY"),("SVIX","SVIX"),("SPY","SPY")]:
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                out[name] = dict(
                    close=round(float(h['Close'].iloc[-1]), 2),
                    open=round(float(h['Open'].iloc[-1]), 2),
                    prev=round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else None,
                )
        except: continue
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE ANALYTICS — DATA LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=300)
def fetch_edge_extra():
    """Fetches VVIX, SKEW, HYG, IEF live from yfinance.
    Index is timezone-normalized to UTC-naive for safe joining."""
    out = {}
    for name, sym in [("VVIX","^VVIX"), ("SKEW","^SKEW"), ("HYG","HYG"), ("IEF","IEF")]:
        try:
            h = yf.download(sym, period="2y", progress=False, auto_adjust=True)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            if not h.empty:
                # Normalize index: remove timezone so join with parquet works
                if hasattr(h.index, "tz") and h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                h.index = pd.DatetimeIndex(h.index).normalize()
                out[name] = h
        except Exception as ex:
            logging.getLogger("vix_controller").warning(f"fetch_edge_extra {sym}: {ex}")
            continue
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOL SKEW & IV SURFACE — BLACK-SCHOLES IV ENGINE
# IV calculada desde cero via Brent's method (igual que functions.py
# del proyecto Volatility Surface de Georgios Drosogiannis).
# No dependemos de la IV de yfinance (ruidosa/incorrecta).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bs_call(S, X, r, T, v, q):
    """Black-Scholes call price con dividendo continuo."""
    if S <= 0 or X <= 0 or T <= 0: return max(S - X, 0.0)
    if v <= 0: return max(S*np.exp(-q*T) - X*np.exp(-r*T), 0.0)
    d1 = (np.log(S/X) + (r - q + 0.5*v**2)*T) / (v*np.sqrt(T))
    d2 = d1 - v*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)

def _bs_put(S, X, r, T, v, q):
    """Black-Scholes put price con dividendo continuo."""
    if S <= 0 or X <= 0 or T <= 0: return max(X - S, 0.0)
    if v <= 0: return max(X*np.exp(-r*T) - S*np.exp(-q*T), 0.0)
    d1 = (np.log(S/X) + (r - q + 0.5*v**2)*T) / (v*np.sqrt(T))
    d2 = d1 - v*np.sqrt(T)
    return X*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def _bs_iv(S, X, r, T, price, option_type, q, tol=1e-6):
    """IV via Brent's method. Retorna NaN si no converge o fuera de bounds."""
    if T <= 0 or S <= 0 or X <= 0 or not np.isfinite(price) or price <= 0:
        return np.nan
    fn = _bs_call if option_type == "C" else _bs_put
    # Bounds de precio
    lo = max(S*np.exp(-q*T) - X*np.exp(-r*T), 0.0) if option_type == "C" \
         else max(X*np.exp(-r*T) - S*np.exp(-q*T), 0.0)
    hi = S*np.exp(-q*T) if option_type == "C" else X*np.exp(-r*T)
    if not (lo <= price <= hi):
        return np.nan
    try:
        iv = brentq(lambda v: price - fn(S, X, r, T, v, q), 1e-6, 5.0, xtol=tol)
        return np.nan if iv <= tol else iv
    except (ValueError, RuntimeError):
        return np.nan

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

def _yahoo_options_session():
    return _get_cffi_session()

@st.cache_data(ttl=1200)   # 20 min — da tiempo para que rate-limit expire
def fetch_options_chains(ticker: str = "SPY", n_exp: int = 4) -> tuple:
    """
    Descarga opciones con triple estrategia anti-rate-limit:
    1. curl_cffi con impersonación Chrome124 (TLS fingerprint real)
       - Rota entre query1 y query2 en cada chain
       - Delay adaptativo: duplica si obtiene 429
    2. Fallback yfinance con backoff exponencial
    3. TTL=20 min para no re-golpear tras rate-limit
    """
    log = logging.getLogger("vix_controller")

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
    log.info(f"Options {ticker}: yfinance fallback")
    try:
        t = yf.Ticker(ticker)
        def _bo(fn, label, n=5):
            for i in range(n):
                try: return fn()
                except Exception as ex:
                    if any(k in str(ex).lower() for k in
                           ["rate limit","too many","429","throttle"]) and i < n-1:
                        w = 2**(i+1)
                        log.warning(f"{label} RL→wait {w}s")
                        time.sleep(w)
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
        return chains, spot
    except Exception as e:
        log.error(f"fetch_options_chains {ticker}: {e}")
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

def _bs_gamma(S: float, X: float, r: float, T: float, v: float, q: float) -> float:
    """Gamma Black-Scholes: ∂²C/∂S² = ∂²P/∂S² (igual para call y put)."""
    if S <= 0 or X <= 0 or T <= 0 or v <= 0:
        return 0.0
    d1 = (np.log(S / X) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * v * np.sqrt(T)))


def compute_gex_profile(chains: dict, spot: float,
                        r: float = 0.043, q: float = 0.013,
                        contract_mult: int = 100) -> pd.DataFrame:
    """
    Calcula el GEX neto de dealers por strike, sumando todos los vencimientos.
    Usa la IV calculada por BS (columna 'iv') si está disponible,
    sino reintenta con la IV de yfinance como fallback.

    Retorna DataFrame con:
      strike, calls_gex, puts_gex, net_gex (todo en USD millones)
    """
    rows = []
    for exp_str, data in chains.items():
        dte = data["dte"]
        T   = dte / 365.0
        if T <= 0:
            continue

        for side, sign in [("calls", -1), ("puts", +1)]:
            df = data[side].copy()
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

            df["gamma"] = df.apply(
                lambda row: _bs_gamma(spot, row["strike"], r, T,
                                      row[iv_col], q),
                axis=1
            )
            # GEX en dólares: OI × Gamma × S² × multiplier × 1% move
            df["gex_usd"] = (
                sign
                * df["openInterest"]
                * df["gamma"]
                * (spot ** 2)
                * contract_mult
                * 0.01   # 1% move
            )
            for _, row in df.iterrows():
                rows.append({
                    "strike":   row["strike"],
                    "side":     side,
                    "dte":      dte,
                    "gex_usd":  row["gex_usd"],
                    "oi":       row["openInterest"],
                })

    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
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

    colors = ["#3FB950" if v >= 0 else "#F85149" for v in df["net_gex"]]

    fig.add_trace(go.Bar(
        x=df["strike"], y=df["net_gex"],
        name="Net GEX (dealers)",
        marker_color=colors,
        opacity=0.8,
        hovertemplate="Strike: $%{x:.0f}<br>Net GEX: $%{y:.2f}M<extra></extra>",
    ))

    # Spot
    fig.add_vline(x=spot, line_dash="solid", line_color="#F0F6FC", line_width=2,
                  annotation_text=f"  Spot ${spot:.1f}",
                  annotation_font=dict(size=10, color="#F0F6FC", family="JetBrains Mono"))

    # Gamma flip
    gf = summary.get("flip_strike")
    if gf:
        fig.add_vline(x=gf, line_dash="dash", line_color="#D29922", line_width=1.5,
                      annotation_text=f"  Flip ${gf:.0f}",
                      annotation_font=dict(size=9, color="#D29922", family="JetBrains Mono"),
                      annotation_position="top right")

    # Call wall
    cw = summary.get("call_wall")
    if cw and lo <= cw <= hi:
        fig.add_vline(x=cw, line_dash="dot", line_color="#BC8CFF", line_width=1.5,
                      annotation_text=f"  Call Wall ${cw:.0f}",
                      annotation_font=dict(size=9, color="#BC8CFF", family="JetBrains Mono"),
                      annotation_position="bottom right")

    # Put wall
    pw = summary.get("put_wall")
    if pw and lo <= pw <= hi:
        fig.add_vline(x=pw, line_dash="dot", line_color="#39D2C0", line_width=1.5,
                      annotation_text=f"  Put Wall ${pw:.0f}",
                      annotation_font=dict(size=9, color="#39D2C0", family="JetBrains Mono"),
                      annotation_position="bottom left")

    regime    = summary.get("regime", "?")
    total_gex = summary.get("total_gex", 0)
    regime_clr = "#3FB950" if regime == "POSITIVE" else "#F85149"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>GEX Profile — {ticker}</b>"
                f"<sup>  Net GEX: <span style='color:{regime_clr}'>${total_gex:+.1f}M · {regime}</span>"
                f"  |  Gamma Flip: ${gf:.0f}" if gf else
                f"<b>GEX Profile — {ticker}</b>"
                f"<sup>  Net GEX: ${total_gex:+.1f}M · {regime}</sup>"
            ),
            font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5,
        ),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=420, margin=dict(l=55, r=30, t=65, b=50),
        xaxis=dict(
            title=dict(text="Strike Price ($)", font=dict(size=10, color="#8B949E")),
            gridcolor="#21262D",
            tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono"),
            tickprefix="$",
        ),
        yaxis=dict(
            title=dict(text="Net GEX ($M / 1% move)", font=dict(size=10, color="#8B949E")),
            gridcolor="#21262D",
            tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono"),
            ticksuffix="M",
            zeroline=True, zerolinecolor="#484F58", zerolinewidth=2,
        ),
        showlegend=False,
        hovermode="x unified",
        bargap=0.1,
    )
    return fig


def build_gex_by_expiry_chart(chains: dict, spot: float,
                               r: float = 0.043, q: float = 0.013) -> go.Figure:
    """
    GEX total por vencimiento — muestra qué expiración concentra más gamma.
    """
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
    colors = ["#3FB950" if v >= 0 else "#F85149" for v in df_e["net_gex_m"]]

    fig.add_trace(go.Bar(
        x=df_e["exp"], y=df_e["net_gex_m"],
        marker_color=colors, opacity=0.8,
        hovertemplate="%{x}<br>GEX: $%{y:.2f}M<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="#484F58", line_width=1.5)
    fig.update_layout(
        title=dict(
            text="<b>GEX por Vencimiento</b><sup>  Qué expiración concentra más gamma</sup>",
            font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5,
        ),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=280, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"),
                   title=dict(text="Vencimiento", font=dict(size=10, color="#8B949E"))),
        yaxis=dict(title=dict(text="Net GEX ($M)", font=dict(size=10, color="#8B949E")),
                   gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"),
                   ticksuffix="M", zeroline=True, zerolinecolor="#484F58"),
        showlegend=False, bargap=0.2,
    )
    return fig


# ─── Métricas de skew (usa columna 'iv' BS) ────────────────────────────────
def build_gex_delta_exposure_chart(chains: dict, spot: float,
                                    r: float = 0.043, q: float = 0.013,
                                    strike_range_pct: float = 0.15) -> go.Figure:
    """
    DEX (Delta Exposure): muestra el delta neto de dealers por strike.
    Donde DEX cruza cero = nivel de máximo dolor (max pain).
    Complementa GEX para entender hacia dónde se mueve el precio.
    """
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
    colors_d = ["#3FB950" if v >= 0 else "#F85149" for v in df_d["delta_usd"]]

    fig.add_trace(go.Bar(
        x=df_d["strike"], y=df_d["delta_usd"],
        marker_color=colors_d, opacity=0.75, name="Delta Exposure",
        hovertemplate="Strike: $%{x:.0f}<br>DEX: $%{y:.2f}M<extra></extra>"))

    fig.add_vline(x=spot, line_dash="solid", line_color="#F0F6FC", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F0F6FC"))
    fig.add_hline(y=0, line_dash="dash", line_color="#484F58", line_width=1.5)

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
            fig.add_vline(x=mp, line_dash="dot", line_color="#D29922", line_width=2,
                          annotation_text=f"  Max Pain ${mp:.0f}",
                          annotation_font=dict(size=9, color="#D29922", family="JetBrains Mono"))

    fig.update_layout(
        title=dict(text="<b>Delta Exposure (DEX)</b><sup>  Presión de hedging por strike · Max Pain marcado</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=320, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="DEX ($M)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"), ticksuffix="M",
                   zeroline=True, zerolinecolor="#484F58", zerolinewidth=2),
        showlegend=False, hovermode="x unified", bargap=0.1)
    return fig


def build_gex_vanna_charm_chart(chains: dict, spot: float,
                                 r: float = 0.043, q: float = 0.013,
                                 strike_range_pct: float = 0.15) -> go.Figure:
    """
    Vanna + Charm por strike.
    Vanna = ∂Delta/∂Vol = ∂Gamma/∂Spot (efecto de cambio de IV sobre delta)
    Charm = ∂Delta/∂t  (decaimiento del delta con el tiempo, importante en expiry weeks)
    Ambos generan flujos de hedging autónomos — críticos para predecir pinning en expiración.
    """
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
        marker_color="#58A6FF", opacity=0.7,
        hovertemplate="Strike: $%{x:.0f}<br>Vanna: %{y:.3f}M<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=df_c["strike"], y=df_c["val"],
        name="Charm (×1000)", yaxis="y2",
        line=dict(color="#BC8CFF", width=2),
        hovertemplate="Strike: $%{x:.0f}<br>Charm: %{y:.3f}<extra></extra>"))

    fig.add_vline(x=spot, line_dash="dash", line_color="#F0F6FC", line_width=1.5,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F0F6FC"))
    fig.add_hline(y=0, line_dash="dot", line_color="#484F58", line_width=1)

    fig.update_layout(
        title=dict(text="<b>Vanna & Charm</b><sup>  Flujos de hedging por cambio de vol (Vanna) y tiempo (Charm)</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=320, margin=dict(l=55, r=60, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="Vanna ($M)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#58A6FF")),
        yaxis2=dict(title="Charm (×1000)", overlaying="y", side="right",
                    tickfont=dict(size=9, color="#BC8CFF"), showgrid=False),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
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
    colors_area = ["rgba(63,185,80,0.15)" if v >= 0 else "rgba(248,81,73,0.15)"
                   for v in df["gex_cumsum"]]

    fig.add_trace(go.Scatter(
        x=df["strike"], y=df["gex_cumsum"],
        name="GEX Acumulado",
        line=dict(color="#39D2C0", width=2.5),
        fill="tozeroy", fillcolor="rgba(57,210,192,0.1)",
        hovertemplate="Strike: $%{x:.0f}<br>GEX Acum: $%{y:.2f}M<extra></extra>"))

    fig.add_vline(x=spot, line_dash="solid", line_color="#F0F6FC", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F0F6FC"))
    fig.add_hline(y=0, line_dash="dash", line_color="#D29922", line_width=2,
                  annotation_text="  Gamma Flip Zone",
                  annotation_font=dict(size=9, color="#D29922"))

    for _, row in zero_crossings.iterrows():
        fig.add_vline(x=row["strike"], line_dash="dot", line_color="#D29922",
                      line_width=1.5)

    flip_pct = (zero_crossings["strike"].iloc[0] / spot - 1)*100 if not zero_crossings.empty else None
    subtitle = f"Flip en ${zero_crossings['strike'].iloc[0]:.0f} ({flip_pct:+.1f}% del spot)" if flip_pct else "Sin flip visible en rango"

    fig.update_layout(
        title=dict(text=f"<b>GEX Acumulado</b><sup>  {subtitle}</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=300, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="GEX Acum ($M)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"), ticksuffix="M",
                   zeroline=True, zerolinecolor="#D29922", zerolinewidth=2),
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
    colors_gex = ["#3FB950" if v >= 0 else "#F85149" for v in df["net_gex"]]
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
            (em1, f"±1σ ({dte_f}d exp)", "#D29922", "dash"),
            (em2, f"±2σ ({dte_f}d exp)", "#F85149", "dot"),
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
            font=dict(size=9, color="#D29922", family="JetBrains Mono"),
            align="right")

    fig.add_vline(x=spot, line_dash="solid", line_color="#F0F6FC", line_width=2,
                  annotation_text=f"  Spot ${spot:.0f}",
                  annotation_font=dict(size=9, color="#F0F6FC"))

    fig.update_layout(
        title=dict(text="<b>Expected Move + GEX Levels</b>"
                        "<sup>  Rango ±1σ/±2σ vs muros de gamma</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=320, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Strike ($)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"), tickprefix="$"),
        yaxis=dict(title="|GEX| ($M)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"), ticksuffix="M"),
        showlegend=False, hovermode="x unified", bargap=0.1)
    return fig


def fit_svi_slice(strikes: np.ndarray, ivs: np.ndarray, F: float,
                  max_iter: int = 500) -> dict | None:
    """
    ═══════════════════════════════════════════════════════════════════
    MODELO: SVI — Stochastic Volatility Inspired Parametrization
    Ref: Gatheral (2004) "A parsimonious arbitrage-free implied volatility
         parameterization with application to the valuation of volatility
         derivatives." Merrill Lynch Global Quantitative Research.
    ═══════════════════════════════════════════════════════════════════

    ESPECIFICACIÓN (Raw SVI):
      w(k) = a + b·[ρ·(k-m) + √((k-m)² + σ²)]

    donde:
      k   = log-moneyness = log(K/F)
      w   = varianza total implícita = IV² × T
      a   = nivel general de varianza (cte)
      b   = pendiente/curvatura total (≥0)
      ρ   = asimetría ∈ [-1, 1]  (ρ < 0 = put skew dominante)
      m   = centro del smile (offset de ATM)
      σ   = suavidad del smile (curvature at-the-money)

    POR QUÉ SVI ES EL MEJOR MODELO PARA ESTE PROPÓSITO:
    ────────────────────────────────────────────────────
    1. ARBITRAGE-FREE: Satisface condiciones de no-arbitraje butterfly
       por construcción (Durrleman 2005) cuando b·(1+|ρ|) ≤ 2.

    2. POCAS PARÁMETROS: 5 parámetros capturan toda la forma del smile
       (nivel, pendiente, curvatura, asimetría, centrado).

    3. EXTRAPOLACIÓN CORRECTA: Alcista/bajista en las colas de forma
       consistente con modelos estocásticos de vol (Heston, SABR).

    4. ESTABILIDAD: Robusto ante sparse data — interpola y extrapola
       de forma coherente donde yfinance tiene huecos de liquidez.

    5. BENCHMARK INDUSTRIA: Estándar de facto en equity vol desks
       (Bergomi 2016, "Stochastic Volatility Modeling").

    RETORNA: dict con parámetros {a, b, rho, m, sigma} + fitted_iv o None.
    """
    if len(strikes) < 5 or len(ivs) < 5:
        return None

    log_mon = np.log(strikes / F)
    T_proxy = 1.0   # varianza total ~ IV² (normalizamos por T en el caller)
    w_obs   = ivs**2 * T_proxy

    def svi_w(k, a, b, rho, m, sigma):
        disc = np.maximum((k - m)**2 + sigma**2, 1e-12)
        return a + b * (rho*(k - m) + np.sqrt(disc))

    # Constraints: a>0, b>0, |rho|<1, sigma>0, b*(1+|rho|)<=2
    from scipy.optimize import minimize

    def loss(params):
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if b * (1 + abs(rho)) > 2:
            return 1e10
        w_fit = svi_w(log_mon, a, b, rho, m, sigma)
        if np.any(w_fit < 0):
            return 1e10
        return float(np.sum((w_obs - w_fit)**2))

    # Initial guess: ATM level, moderate skew
    a0    = float(np.median(w_obs)) * 0.8
    b0    = 0.1
    rho0  = -0.3
    m0    = 0.0
    sig0  = 0.1

    try:
        res = minimize(loss, [a0, b0, rho0, m0, sig0],
                       method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-8})
        if not res.success and res.fun > 0.1:
            return None
        a, b, rho, m, sigma = res.x
        if b <= 0 or sigma <= 0 or abs(rho) >= 0.99:
            return None

        # Fitted IVs
        w_fit = svi_w(log_mon, a, b, rho, m, sigma)
        iv_fit = np.sqrt(np.maximum(w_fit / T_proxy, 0))

        return {
            "a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma),
            "iv_fit": iv_fit,
            "log_mon": log_mon,
            "r2": float(1 - np.sum((w_obs - w_fit)**2) / np.sum((w_obs - w_obs.mean())**2)),
        }
    except Exception:
        return None


def forecast_vol_surface(chains: dict, spot: float,
                          r: float = 0.043, q: float = 0.013,
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
    log = logging.getLogger("vix_controller")
    F = spot * np.exp((r - q) * 30/365)   # forward ~30d

    svi_fits = {}
    for exp_str, data in chains.items():
        dte = data["dte"]; T = dte / 365.0
        if T <= 0: continue
        puts_f  = data["puts"][data["puts"]["moneyness"].between(0.80, 1.02)]
        calls_f = data["calls"][data["calls"]["moneyness"].between(0.98, 1.20)]
        combo   = pd.concat([puts_f, calls_f]).drop_duplicates("strike").sort_values("strike")
        iv_col  = "iv" if "iv" in combo.columns else "impliedVolatility"
        combo   = combo[combo[iv_col].notna() & (combo[iv_col] > 0.01)]
        if len(combo) < 5: continue

        fit = fit_svi_slice(combo["strike"].values, combo[iv_col].values,
                            F * np.exp((r-q)*T))
        if fit:
            svi_fits[exp_str] = {**fit, "dte": dte, "combo": combo}
            log.info(f"SVI {exp_str}: R²={fit['r2']:.3f} ρ={fit['rho']:.3f} b={fit['b']:.3f}")

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
            df = df[df[iv_col].notna() & (df["strike"].between(spot*0.82, spot*1.18))]
            if df.empty: continue

            for _, row in df.iterrows():
                K   = row["strike"]; iv_c = float(row.get(iv_col, 0) or 0)
                oi  = row["openInterest"]; mid = row.get("midPrice", 0)
                if K <= 0 or iv_c <= 0 or mid <= 0: continue
                if not (0.82 <= K/spot <= 1.18): continue  # ±18% max

                k_  = np.log(K / (F * np.exp((r-q)*T)))
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
        "F":             F,
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

    # Puntos observados
    combo = fit.get("combo", pd.DataFrame())
    iv_col = "iv" if "iv" in combo.columns else "impliedVolatility"
    if not combo.empty and iv_col in combo.columns:
        obs_k   = np.log(combo["strike"].values / forecast_result["F"]) * 100
        obs_iv  = combo[iv_col].values * 100
        fig.add_trace(go.Scatter(
            x=obs_k, y=obs_iv, mode="markers",
            name="Observado",
            marker=dict(color="#8B949E", size=6, opacity=0.7),
            hovertemplate="k: %{x:.1f}%<br>IV obs: %{y:.1f}%<extra></extra>"))

    # Fitted SVI actual
    fig.add_trace(go.Scatter(
        x=k_pct, y=fc["iv_cur"]*100, mode="lines",
        name="SVI Actual",
        line=dict(color="#58A6FF", width=2.5),
        hovertemplate="k: %{x:.1f}%<br>IV SVI: %{y:.1f}%<extra></extra>"))

    # SVI Forecasted
    fig.add_trace(go.Scatter(
        x=k_pct, y=fc["iv_fc"]*100, mode="lines",
        name=f"SVI Forecast ({iv_change_pct:+.0%})",
        line=dict(color="#3FB950", width=2.5, dash="dash"),
        hovertemplate="k: %{x:.1f}%<br>IV forecast: %{y:.1f}%<extra></extra>"))

    # Área de oportunidad (IV drop)
    fig.add_trace(go.Scatter(
        x=list(k_pct)+list(k_pct[::-1]),
        y=list(fc["iv_cur"]*100)+list(fc["iv_fc"]*100)[::-1],
        fill="toself", fillcolor="rgba(63,185,80,0.12)",
        line=dict(width=0), name="IV Drop (oportunidad)", hoverinfo="skip"))

    fig.add_vline(x=0, line_dash="dash", line_color="#8B949E", line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=9, color="#8B949E"))

    fig.update_layout(
        title=dict(
            text=f"<b>SVI Smile — {exp_str} ({dte}d)</b>"
                 f"<sup>  Azul=actual · Verde=forecast · zona=oportunidad de venta</sup>",
            font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=350, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title="Log-moneyness k (%)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"), ticksuffix="%",
                   zeroline=True, zerolinecolor="#30363D"),
        yaxis=dict(title="IV (%)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"), ticksuffix="%"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
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

    colorscale = ([[0.0,"#1565C0"],[0.3,"#3FB950"],[0.6,"#D29922"],[1.0,"#F85149"]]
                  if view == "drop" else
                  [[0.0,"#1a237e"],[0.3,"#0288D1"],[0.6,"#3FB950"],[0.8,"#D29922"],[1.0,"#F85149"]])

    fig.add_trace(go.Surface(
        x=k_grid, y=dtes, z=Z,
        colorscale=colorscale,
        colorbar=dict(title=dict(text="∆IV pts" if view=="drop" else "IV%",
                                  font=dict(color="#8B949E", size=10)),
                      tickfont=dict(color="#8B949E", size=9), len=0.6, thickness=12),
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
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        scene=dict(
            xaxis=dict(title="Log-moneyness (%)", gridcolor="#30363D",
                       backgroundcolor="#0D1117", tickfont=dict(size=9, color="#8B949E")),
            yaxis=dict(title="DTE (días)", gridcolor="#30363D",
                       backgroundcolor="#0D1117", tickfont=dict(size=9, color="#8B949E")),
            zaxis=dict(title="∆IV pts" if view=="drop" else "IV %",
                       gridcolor="#30363D", backgroundcolor="#0D1117",
                       tickfont=dict(size=9, color="#8B949E")),
            bgcolor="#0D1117",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.9))),
        paper_bgcolor="#0D1117", height=500, margin=dict(l=0, r=0, t=50, b=0))
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
    "#58A6FF","#F0883E","#3FB950","#BC8CFF",
    "#39D2C0","#D29922","#F85149","#79C0FF",
]

def build_skew_curves(chains: dict, spot: float,
                      moneyness_range=(0.75, 1.25),
                      y_mode: str = "moneyness") -> go.Figure:
    """
    Curvas IV (BS) vs Moneyness o log-moneyness por vencimiento.
    y_mode: 'moneyness' → % vs spot | 'log' → ln(K/F)
    Usa columna 'iv' (BS calculado), no yfinance.
    """
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
            F = spot * np.exp(0 * T)   # r, q are baked into iv already
            x_vals = np.log(combined["strike"].values / spot)  # approx log-moneyness
            x_label = "Log-moneyness  ln(K/S)"
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

    fig.add_vline(x=0, line_dash="dash", line_color="#8B949E", line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=10, color="#8B949E"))
    fig.update_layout(
        title=dict(text="<b>Volatility Skew</b><sup>  IV Black-Scholes por vencimiento</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=420, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title=dict(text=x_label, font=dict(size=10, color="#8B949E")),
                   gridcolor="#21262D", zeroline=True, zerolinecolor="#30363D",
                   tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono")),
        yaxis=dict(title=dict(text="Implied Volatility BS (%)", font=dict(size=10, color="#8B949E")),
                   gridcolor="#21262D",
                   tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono"),
                   ticksuffix="%"),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363D", borderwidth=1,
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
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
        line=dict(color="#39D2C0", width=3, shape="spline"),
        marker=dict(size=10, color="#39D2C0", line=dict(width=2, color="#0D1117")),
        text=[f"{v:.1f}%" for v in df_atm["atm_iv"]],
        textposition="top center",
        textfont=dict(size=9, color="#C9D1D9", family="JetBrains Mono"),
        hovertemplate="DTE: %{x}d<br>ATM IV: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>ATM IV Term Structure</b><sup>  IV en el dinero por vencimiento</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=300, margin=dict(l=55, r=30, t=60, b=50),
        xaxis=dict(title=dict(text="Días al Vencimiento (DTE)", font=dict(size=10, color="#8B949E")),
                   gridcolor="#21262D",
                   tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono")),
        yaxis=dict(title=dict(text="ATM IV (%)", font=dict(size=10, color="#8B949E")),
                   gridcolor="#21262D",
                   tickfont=dict(size=10, color="#8B949E", family="JetBrains Mono"),
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
            [0.0,"#1a237e"],[0.15,"#1565C0"],[0.30,"#0288D1"],
            [0.45,"#00ACC1"],[0.55,"#3FB950"],[0.65,"#D29922"],
            [0.80,"#F0883E"],[1.0,"#F85149"],
        ],
        colorbar=dict(title=dict(text="IV %", font=dict(color="#8B949E",size=10)),
                      tickfont=dict(color="#8B949E",size=9), len=0.6, thickness=12),
        hovertemplate="DTE: %{x:.0f}d<br>Y: %{y:.2f}<br>IV: %{z:.1f}%<extra></extra>",
        opacity=0.92,
    ))
    fig.update_layout(
        title=dict(text="<b>Implied Volatility Surface</b><sup>  IV Black-Scholes · griddata interpolation</sup>",
                   font=dict(size=14, color="#C9D1D9", family="Inter"), x=0.5),
        scene=dict(
            xaxis=dict(title="DTE (días)", gridcolor="#30363D", backgroundcolor="#0D1117",
                       tickfont=dict(size=9, color="#8B949E")),
            yaxis=dict(title=y_label, gridcolor="#30363D", backgroundcolor="#0D1117",
                       tickfont=dict(size=9, color="#8B949E")),
            zaxis=dict(title="IV (%)", gridcolor="#30363D", backgroundcolor="#0D1117",
                       tickfont=dict(size=9, color="#8B949E")),
            bgcolor="#0D1117",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9), up=dict(x=0,y=0,z=1)),
        ),
        paper_bgcolor="#0D1117", height=520, margin=dict(l=0,r=0,t=50,b=0),
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
        colorscale=[[0.0,"#1565C0"],[0.25,"#0288D1"],[0.50,"#3FB950"],
                    [0.70,"#D29922"],[0.85,"#F0883E"],[1.0,"#F85149"]],
        colorbar=dict(title=dict(text="IV %",font=dict(color="#8B949E",size=10)),
                      tickfont=dict(color="#8B949E",size=9), len=0.8, thickness=14),
        hoverongaps=False,
        hovertemplate="Δ Spot: %{x}<br>DTE: %{y}<br>IV(BS): %{z:.1f}%<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.add_vline(x=labels_x[atm_idx], line_dash="dash", line_color="#8B949E",
                  line_width=1.5,
                  annotation_text="ATM", annotation_font=dict(size=9, color="#8B949E"))
    fig.update_layout(
        title=dict(text="<b>IV Surface — Heatmap</b><sup>  Filas=DTE · Columnas=%Spot · Color=IV(BS)%</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=380, margin=dict(l=55, r=20, t=60, b=60),
        xaxis=dict(tickfont=dict(size=8,color="#8B949E",family="JetBrains Mono"),
                   title=dict(text="Distancia al Spot",font=dict(size=10,color="#8B949E")),
                   tickangle=-45),
        yaxis=dict(tickfont=dict(size=9,color="#8B949E",family="JetBrains Mono"),
                   title=dict(text="DTE",font=dict(size=10,color="#8B949E")),
                   autorange="reversed"),
    )
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE EXTENSION — SPY + VIX desde yfinance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=55)
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
        try:
            h = yf.Ticker(sym).history(period="45d")
            if h.empty:
                continue
            s = h["Close"].copy()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = pd.DatetimeIndex(s.index).normalize()
            frames[col] = s
        except Exception as ex:
            log.warning(f"fetch_live_spy_vix {sym}: {ex}")
    if "SPY_Close" in frames and "VIX_Close" in frames:
        df = pd.DataFrame(frames)
        log.info(f"Live extension: {len(df)} rows, last={df.index[-1].date()}")
        return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)
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

    for i in range(window + 22, n):    # sin -h: predecimos hasta hoy inclusive
        t0 = i - window
        X  = np.column_stack([np.ones(window),
                               gv_a[t0:i], bv_a[t0:i],
                               rw_a[t0:i], rm_a[t0:i]])
        y  = rvf_a[t0:i]
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
        valid = (m1 > 0) & (dte > 0) & m1.notna() & dte.notna() & spot.notna()
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
        line=dict(color='#F85149', width=2.5),
        hovertemplate='VIX: %{y:.1f}<extra></extra>'), row=1, col=1)

    if use_har:
        fig.add_trace(go.Scatter(
            x=p.index, y=p['HAR_Forecast'], name='HAR-RV Forecast (E[vol])',
            line=dict(color='#58A6FF', width=2, dash='dash'),
            hovertemplate='HAR Forecast: %{y:.1f}<extra></extra>'), row=1, col=1)
        if 'RV_Fwd_22' in p.columns:
            fig.add_trace(go.Scatter(
                x=p.index, y=p['RV_Fwd_22'], name='RV Realizada 22d (ex-post)',
                line=dict(color='#39D2C0', width=1.5, dash='dot'),
                hovertemplate='RV realizada: %{y:.1f}<extra></extra>'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=p.index, y=p['RV20'], name='RV20 (trailing)',
            line=dict(color='#58A6FF', width=2),
            hovertemplate='RV20: %{y:.1f}<extra></extra>'), row=1, col=1)

    # Panel 2: VRP como área
    if col_vrp in p.columns and p[col_vrp].notna().sum() > 5:
        vrp_vals = p[col_vrp].fillna(0)
        colors_vrp = ['#3FB950' if v >= 0 else '#F85149' for v in vrp_vals]
        fig.add_trace(go.Bar(
            x=p.index, y=vrp_vals,
            name='VRP = VIX − E[RV]' if use_har else 'VRP = VIX − RV20',
            marker_color=colors_vrp, opacity=0.7,
            hovertemplate='VRP: %{y:+.1f} pts<extra></extra>'), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='#484F58',
                      line_width=1.5, row=2, col=1)

        # Líneas de percentil P25/P75 en VRP
        vrp_clean = vrp_vals[vrp_vals.notna()]
        if len(vrp_clean) > 20:
            p25 = float(vrp_clean.quantile(0.25))
            p75 = float(vrp_clean.quantile(0.75))
            fig.add_hline(y=p25, line_dash='dot', line_color='#D29922',
                          line_width=1, row=2, col=1,
                          annotation_text=f' P25: {p25:.1f}',
                          annotation_font=dict(size=8, color='#D29922'))
            fig.add_hline(y=p75, line_dash='dot', line_color='#3FB950',
                          line_width=1, row=2, col=1,
                          annotation_text=f' P75: {p75:.1f}',
                          annotation_font=dict(size=8, color='#3FB950'))

    subtitle = ('VIX vs HAR-RV Forecast · VRP = prima pagada sobre vol esperada'
                if use_har else 'VIX - RV20 trailing (definición simplificada)')
    fig.update_layout(
        title=dict(
            text=f'<b>Volatility Risk Premium</b><sup>  {subtitle}</sup>',
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=480, margin=dict(l=55, r=30, t=60, b=40),
        xaxis2=dict(gridcolor='#21262D',
                    tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Vol %', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis2=dict(title='VRP (pts)', gridcolor='#21262D',
                    tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono'),
                    zeroline=True, zerolinecolor='#30363D'),
        legend=dict(orientation='h', y=1.03, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#C9D1D9', family='JetBrains Mono')),
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
        line=dict(color='#39D2C0', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>RV real: %{y:.1f}%<extra></extra>'))
    fig_ts.add_trace(go.Scatter(
        x=idx, y=fc, name='HAR-A Forecast',
        line=dict(color='#F0883E', width=2, dash='dash'),
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
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=300, margin=dict(l=55, r=30, t=60, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Vol % (anualizada)', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#C9D1D9', family='JetBrains Mono')),
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
        marker=dict(color='#58A6FF', size=4, opacity=0.5),
        name='Observaciones',
        hovertemplate='Forecast: %{x:.1f}%<br>Real: %{y:.1f}%<extra></extra>'))
    # Línea ideal 45°
    fig_mz.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax], mode='lines',
        name='Ideal (a=0, b=1)',
        line=dict(color='#3FB950', width=2, dash='dot')))
    # Línea MZ regresión
    x_line = np.linspace(vmin, vmax, 100)
    y_line = alpha + beta * x_line
    fig_mz.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        name=f'MZ fit: a={alpha:.2f}, b={beta:.2f}',
        line=dict(color='#F0883E', width=2)))
    fig_mz.update_layout(
        title=dict(
            text=f'<b>Mincer-Zarnowitz</b>'
                 f'<sup>  a={alpha:.2f} (↓0) · b={beta:.2f} (↑1) · sin sesgo si a≈0, b≈1</sup>',
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=300, margin=dict(l=55, r=30, t=60, b=55),
        xaxis=dict(title='HAR-A Forecast (%)', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        yaxis=dict(title='RV Realizada (%)', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#C9D1D9', family='JetBrains Mono')))

    return fig_ts, fig_mz


def build_rv_chart(bt, window=252):
    p = bt.tail(window).dropna(subset=['RV20'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['VIX_Close'], name='VIX',
        line=dict(color='#F85149', width=2.5)))
    for col, lbl, clr in [('RV5','RV5 (1w)','#D29922'), ('RV10','RV10 (2w)','#F0883E'),
                           ('RV20','RV20 (1m)','#58A6FF'), ('RV60','RV60 (3m)','#BC8CFF')]:
        if col in p.columns:
            fig.add_trace(go.Scatter(x=p.index, y=p[col], name=lbl, line=dict(color=clr, width=1.2)))
    fig.update_layout(
        title=dict(text='<b>Implied vs Realized Vol</b><sup>  VIX encima = VRP positivo</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=350, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Vol %', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_roll_yield_chart(bt, window=252):
    if 'Roll_Yield' not in bt.columns:
        return go.Figure()
    p = bt.tail(window).dropna(subset=['Roll_Yield'])
    colors = ['#3FB950' if v > 0 else '#F85149' for v in p['Roll_Yield']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p.index, y=p['Roll_Yield'], marker_color=colors,
        name='Roll Yield %', opacity=0.7))
    fig.add_trace(go.Scatter(x=p.index, y=p['Roll_Yield'].rolling(20).mean(),
        name='SMA(20)', line=dict(color='#39D2C0', width=2)))
    fig.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=1)
    fig.update_layout(
        title=dict(text='<b>Roll Yield</b><sup>  Carry anualizado · Verde=cobras</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=300, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Ann. %', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
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
        line=dict(color='#BC8CFF', width=2),
        fill='tozeroy', fillcolor='rgba(188,140,255,0.07)',
        hovertemplate='%{x|%Y-%m-%d}<br>VVIX/VIX: %{y:.2f}<extra></extra>'))

    # Bands contextuales
    fig.add_hrect(y0=6, y1=max(float(y_vals.max(skipna=True)) + 1, 8),
                  fillcolor='rgba(248,81,73,0.07)', line_width=0)
    fig.add_hline(y=6, line_dash='dash', line_color='#F85149', line_width=1.5,
        annotation_text='  ⚠ Danger > 6', annotation_font=dict(color='#F85149', size=10))
    fig.add_hline(y=5, line_dash='dot', line_color='#D29922', line_width=1,
        annotation_text='  Warning > 5', annotation_font=dict(color='#D29922', size=9))
    fig.add_hline(y=4, line_dash='dot', line_color='#3FB950', line_width=0.8,
        annotation_text='  Calm < 4', annotation_font=dict(color='#3FB950', size=8))

    # SMA 20d
    sma = pd.Series(y_vals.values, index=p.index).rolling(20, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=p.index, y=sma, name='SMA(20)',
        line=dict(color='#39D2C0', width=1.2, dash='dot'), showlegend=True,
        hovertemplate='SMA20: %{y:.2f}<extra></extra>'))

    fig.update_layout(
        title=dict(
            text=f'<b>VVIX / VIX Ratio</b><sup>  Fuente: {src_label} · > 6 = dealers anticipan spike</sup>',
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=300, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Ratio', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
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
        fillcolor='rgba(88,166,255,0.06)', line_width=0)
    fig.add_trace(go.Scatter(
        x=p.index, y=p['SKEW'], name='CBOE SKEW (^SKEW · yfinance)',
        line=dict(color='#F0883E', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>SKEW: %{y:.0f}<extra></extra>'))
    # SMA 20
    skew_sma = p['SKEW'].rolling(20, min_periods=5).mean()
    fig.add_trace(go.Scatter(
        x=p.index, y=skew_sma, name='SMA(20)',
        line=dict(color='#8B949E', width=1.2, dash='dot'),
        hovertemplate='SMA: %{y:.0f}<extra></extra>'))
    fig.add_hline(y=mean_skew, line_dash='dot', line_color='#58A6FF', line_width=1,
        annotation_text=f'  μ={mean_skew:.0f}',
        annotation_font=dict(color='#58A6FF', size=9))
    fig.add_hline(y=150, line_dash='dash', line_color='#F85149', line_width=1.5,
        annotation_text='  Extremo > 150 (tail-risk hedging)',
        annotation_font=dict(color='#F85149', size=9))
    fig.add_hline(y=130, line_dash='dot', line_color='#D29922', line_width=1,
        annotation_text='  Elevado > 130',
        annotation_font=dict(color='#D29922', size=8))
    fig.update_layout(
        title=dict(text='<b>CBOE SKEW Index</b><sup>  ^SKEW yfinance · > 150 = demanda extrema de cola</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='SKEW', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
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
    colors_cs = ['#F85149' if v > 0 else '#3FB950' for v in p['Credit_Spread']]
    fig.add_trace(go.Scatter(
        x=p.index, y=p['Credit_Spread'].clip(lower=0),
        name='Spread widening (risk-off)',
        fill='tozeroy', line=dict(color='#F85149', width=0),
        fillcolor='rgba(248,81,73,0.15)',
        hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=p.index, y=p['Credit_Spread'], name='Credit Spread (HYG-IEF · yfinance)',
        line=dict(color='#D29922', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Spread: %{y:.2f}<extra></extra>'))
    # VIX en eje derecho
    fig.add_trace(go.Scatter(
        x=p.index, y=p['VIX_Close'], name='VIX (^VIX)',
        yaxis='y2', line=dict(color='#F85149', width=1.5, dash='dot'),
        hovertemplate='VIX: %{y:.1f}<extra></extra>'))
    # Líneas de percentil
    fig.add_hline(y=0, line_dash='dash', line_color='#484F58', line_width=1)
    fig.add_hline(y=p75, line_dash='dot', line_color='#D29922', line_width=1,
        annotation_text=f'  P75: {p75:.2f}',
        annotation_font=dict(color='#D29922', size=8))
    fig.add_hline(y=p90, line_dash='dash', line_color='#F85149', line_width=1,
        annotation_text=f'  P90: {p90:.2f} (stress)',
        annotation_font=dict(color='#F85149', size=8))
    fig.update_layout(
        title=dict(
            text='<b>Credit Spread vs VIX</b><sup>  HYG/IEF yfinance · Divergencia credit/VIX = warning</sup>',
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=60, t=55, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Credit Spread (20d momentum)',
                   gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        yaxis2=dict(title='VIX', overlaying='y', side='right',
                    tickfont=dict(size=9, color='#F85149'), showgrid=False),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MONITOR OPERATIVO — DATA LAYER (parquet local del repo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARQUET_PATH = "data/master.parquet"

@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=55)
def fetch_today_prices():
    """Precios del día: VXX, SVXY, SVIX, VIX, SPY."""
    out = {}
    for name, sym in [("VXX","VXX"),("SVXY","SVXY"),("SVIX","SVIX"),
                       ("VIX","^VIX"),("SPY","SPY")]:
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                out[name] = dict(
                    close=round(float(h['Close'].iloc[-1]), 2),
                    prev =round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else None,
                    date =h.index[-1].date(),
                )
        except:
            continue
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WALK-FORWARD BACKTEST ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600)
def run_walkforward_backtest(bt: pd.DataFrame,
                              svxy_col: str = "SVXY_Close",
                              spy_col:  str = "SPY_Close",
                              rf_annual: float = 0.043,
                              wf_months: int = 6) -> dict:
    """
    Walk-Forward Analysis de la estrategia BB×Contango.

    METODOLOGÍA:
    ─────────────────────────────────────────────────────────
    1. FULL-SAMPLE backtest: aplica la señal sobre todo el histórico.
       - Retorno diario: sig_final(t-1) × ret_SVXY(t)
       - Benchmark: Buy & Hold SVXY
       - Benchmark2: Buy & Hold SPY

    2. WALK-FORWARD: ventanas deslizantes de `wf_months` meses.
       Por cada ventana calcula Sharpe, Calmar, Win Rate, etc.
       Permite ver si el edge se mantiene o se deteriora en el tiempo.

    3. SHARPE ROLLING 6M: señal de alerta si Sharpe < 0.5 dos meses
       consecutivos → edge en deterioro.

    4. MÉTRICAS COMPLETAS:
       - CAGR, Sharpe, Sortino, Calmar, Max Drawdown
       - Win rate por trade, Avg hold time
       - Alpha vs SPY (Jensen's alpha mensual)
       - Hit rate contango filter (% días en contango con señal activa)

    RETORNA dict con:
      - full:   dict métricas full-sample
      - wf_df:  DataFrame walk-forward por ventana
      - equity: Series equity curve diaria
      - trades: DataFrame trades individuales
      - monthly: DataFrame retornos mensuales para heatmap
    """
    from scipy import stats as _stats
    log = logging.getLogger("vix_controller")

    needed = ["sig_final", "VXX_Close", "In_Contango", "Contango_pct"]
    for c in needed:
        if c not in bt.columns:
            log.warning(f"WF backtest: columna faltante {c}")
            return {}

    # ── 1. Retornos diarios ────────────────────────────────────────────────
    df = bt.copy()

    # Retorno del vehículo: SVXY si disponible, sino aprox -0.5×VXX
    if svxy_col in df.columns and df[svxy_col].notna().sum() > 100:
        df["ret_vehicle"] = df[svxy_col].pct_change()
    else:
        # Aproximación: -0.5 × cambio % VXX (antes de costos)
        df["ret_vehicle"] = -0.5 * df["VXX_Close"].pct_change()
        log.info("WF: usando aproximación -0.5×VXX como retorno")

    if spy_col in df.columns and df[spy_col].notna().sum() > 100:
        df["ret_spy"] = df[spy_col].pct_change()
    else:
        df["ret_spy"] = np.nan

    # Retorno de la estrategia: señal del día anterior × retorno del vehículo
    df["ret_strat"] = df["sig_final"].shift(1).fillna(0) * df["ret_vehicle"]
    df["ret_bh"]    = df["ret_vehicle"]   # buy & hold SVXY

    df = df.dropna(subset=["ret_strat"]).copy()
    if len(df) < 60:
        return {}

    rf_daily = (1 + rf_annual)**(1/252) - 1

    # ── 2. Función métricas de un período ─────────────────────────────────
    def _metrics(ret_series: pd.Series, rf_d: float = rf_daily) -> dict:
        r = ret_series.dropna()
        if len(r) < 20:
            return {}
        n    = len(r)
        days = (r.index[-1] - r.index[0]).days or 1
        years = days / 365.25

        eq   = (1 + r).cumprod()
        cagr = float(eq.iloc[-1]**(1/years) - 1) if years > 0 else np.nan

        excess = r - rf_d
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan

        neg    = r[r < 0]
        sortino= float(excess.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else np.nan

        roll_max = eq.cummax()
        dd       = (eq - roll_max) / roll_max
        max_dd   = float(dd.min())
        calmar   = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

        total_ret  = float(eq.iloc[-1] - 1)
        volatility = float(r.std() * np.sqrt(252))
        win_days   = (r > 0).sum() / n

        return dict(
            n_days=n, years=round(years, 2),
            cagr=round(cagr*100, 2) if not np.isnan(cagr) else None,
            sharpe=round(sharpe, 3) if not np.isnan(sharpe) else None,
            sortino=round(sortino, 3) if not np.isnan(sortino) else None,
            calmar=round(calmar, 3) if not np.isnan(calmar) else None,
            max_dd=round(max_dd*100, 2),
            total_ret=round(total_ret*100, 2),
            volatility=round(volatility*100, 2),
            win_days=round(win_days*100, 1),
        )

    # ── 3. Full-sample metrics ─────────────────────────────────────────────
    full = _metrics(df["ret_strat"])
    full_bh  = _metrics(df["ret_bh"])
    full_spy = _metrics(df["ret_spy"].dropna()) if "ret_spy" in df else {}

    # Alpha vs SPY (monthly OLS)
    if "ret_spy" in df.columns:
        mret_strat = df["ret_strat"].resample("ME").apply(lambda x: (1+x).prod()-1)
        mret_spy   = df["ret_spy"].resample("ME").apply(lambda x: (1+x).prod()-1)
        merged_m   = pd.concat([mret_strat, mret_spy], axis=1).dropna()
        merged_m.columns = ["strat","spy"]
        if len(merged_m) >= 12:
            slope, intercept, r_val, _, _ = _stats.linregress(
                merged_m["spy"].values, merged_m["strat"].values)
            full["alpha_monthly"]   = round(float(intercept)*100, 3)
            full["beta"]            = round(float(slope), 3)
            full["r2_vs_spy"]       = round(float(r_val**2), 3)
        else:
            full["alpha_monthly"] = np.nan

    full["bh_sharpe"]   = full_bh.get("sharpe")
    full["bh_cagr"]     = full_bh.get("cagr")
    full["spy_sharpe"]  = full_spy.get("sharpe")
    full["spy_cagr"]    = full_spy.get("cagr")

    # ── 4. Trades individuales ─────────────────────────────────────────────
    sig  = df["sig_final"].shift(1).fillna(0).astype(int)
    ret  = df["ret_vehicle"]
    equity_vec = [1.0]
    trades_list = []
    in_trade = False; entry_date = None; entry_eq = 1.0; trade_ret = 1.0

    for i in range(len(df)):
        s_prev = sig.iloc[i]
        r_i    = ret.iloc[i] if not np.isnan(ret.iloc[i]) else 0.0

        if s_prev == 1 and not in_trade:
            in_trade = True; entry_date = df.index[i]; entry_eq = equity_vec[-1]

        if s_prev == 1:
            trade_ret = (equity_vec[-1] + equity_vec[-1] * r_i) / entry_eq
            equity_vec.append(equity_vec[-1] * (1 + r_i))
        else:
            equity_vec.append(equity_vec[-1])
            if in_trade:
                trades_list.append({
                    "entry":    entry_date,
                    "exit":     df.index[i],
                    "hold_d":   (df.index[i] - entry_date).days,
                    "ret_pct":  round((trade_ret - 1)*100, 2),
                    "exit_why": "BB" if df["sig_bb"].iloc[i] == 0 else "CT",
                })
                in_trade = False; trade_ret = 1.0

    equity = pd.Series(equity_vec[1:], index=df.index, name="equity")
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

    if not trades_df.empty:
        full["n_trades"]   = len(trades_df)
        full["win_rate"]   = round((trades_df["ret_pct"] > 0).mean()*100, 1)
        full["avg_hold_d"] = round(trades_df["hold_d"].mean(), 1)
        full["avg_win"]    = round(trades_df.loc[trades_df["ret_pct"]>0,"ret_pct"].mean(), 2)
        full["avg_loss"]   = round(trades_df.loc[trades_df["ret_pct"]<0,"ret_pct"].mean(), 2)
        full["profit_factor"] = round(
            abs(trades_df.loc[trades_df["ret_pct"]>0,"ret_pct"].sum() /
                trades_df.loc[trades_df["ret_pct"]<0,"ret_pct"].sum())
            if (trades_df["ret_pct"] < 0).any() else np.inf, 2)

    # ── 5. Walk-Forward: ventanas de wf_months meses ───────────────────────
    wf_rows = []
    window_days = wf_months * 21   # aprox días hábiles

    for start_i in range(0, len(df) - window_days, 21):
        end_i = start_i + window_days
        if end_i > len(df): break
        window = df.iloc[start_i:end_i]
        m = _metrics(window["ret_strat"])
        if not m: continue
        m_bh = _metrics(window["ret_bh"])
        m["period_start"] = window.index[0].date()
        m["period_end"]   = window.index[-1].date()
        m["bh_sharpe"]    = m_bh.get("sharpe")
        m["pct_invested"] = round(window["sig_final"].mean()*100, 1)
        m["ct_ratio"]     = round(window["In_Contango"].mean()*100, 1) if "In_Contango" in window else None
        wf_rows.append(m)

    wf_df = pd.DataFrame(wf_rows)
    if not wf_df.empty:
        wf_df = wf_df.sort_values("period_start").reset_index(drop=True)

    # ── 6. Sharpe rolling 6M ───────────────────────────────────────────────
    roll_window = 126   # ~6 meses hábiles
    df["sharpe_roll"] = (
        df["ret_strat"].rolling(roll_window).apply(
            lambda x: (x - rf_daily).mean() / x.std() * np.sqrt(252)
            if x.std() > 0 else np.nan, raw=True)
    )
    df["dd_roll"] = (equity / equity.rolling(roll_window).max() - 1) * 100

    # ── 7. Retornos mensuales ──────────────────────────────────────────────
    monthly_strat = df["ret_strat"].resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    monthly_bh    = df["ret_bh"].resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    monthly_df    = pd.DataFrame({"Estrategia": monthly_strat, "SVXY B&H": monthly_bh})
    monthly_df.index = monthly_df.index.to_period("M").astype(str)

    log.info(f"WF Backtest: {full.get('n_trades',0)} trades · Sharpe={full.get('sharpe','?')} · CAGR={full.get('cagr','?')}%")

    return {
        "full":    full,
        "wf_df":   wf_df,
        "equity":  equity,
        "trades":  trades_df,
        "monthly": monthly_df,
        "df_with_rolling": df,
        "wf_months": wf_months,
    }


def build_equity_curve_chart(result: dict) -> go.Figure:
    """Curva de equity de la estrategia vs Buy & Hold SVXY y SPY."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.03)

    eq   = result.get("equity", pd.Series(dtype=float))
    df_r = result.get("df_with_rolling", pd.DataFrame())

    if eq.empty: return fig

    # Normalizar a base 100
    eq_norm = eq / eq.iloc[0] * 100

    # Benchmark B&H SVXY
    if "ret_bh" in df_r.columns:
        bh_eq = (1 + df_r["ret_bh"].fillna(0)).cumprod()
        bh_eq = bh_eq / bh_eq.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df_r.index, y=bh_eq, name="SVXY Buy & Hold",
            line=dict(color="#484F58", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>B&H: %{y:.0f}<extra></extra>"), row=1, col=1)

    if "ret_spy" in df_r.columns:
        spy_eq = (1 + df_r["ret_spy"].fillna(0)).cumprod()
        spy_eq = spy_eq / spy_eq.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df_r.index, y=spy_eq, name="SPY Buy & Hold",
            line=dict(color="#8B949E", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>SPY: %{y:.0f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq_norm, name="Estrategia BB×CT",
        line=dict(color="#3FB950", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Estrategia: %{y:.0f}<extra></extra>"), row=1, col=1)

    # Panel 2: Sharpe rolling 6M
    if "sharpe_roll" in df_r.columns:
        sr = df_r["sharpe_roll"].dropna()
        colors_sr = ["#3FB950" if v >= 1.0 else "#D29922" if v >= 0.5 else "#F85149" for v in sr]
        fig.add_trace(go.Scatter(
            x=sr.index, y=sr, name="Sharpe Rolling 6M",
            line=dict(color="#58A6FF", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe 6M: %{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="#3FB950", line_width=1,
                      annotation_text="  1.0 (óptimo)", annotation_font=dict(color="#3FB950", size=8),
                      row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#D29922", line_width=1,
                      annotation_text="  0.5 (alerta)", annotation_font=dict(color="#D29922", size=8),
                      row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="#484F58", line_width=1,
                      row=2, col=1)

    wf_m = result.get("wf_months", 6)
    fig.update_layout(
        title=dict(
            text=f"<b>Equity Curve + Sharpe Rolling {wf_m}M</b>"
                 "<sup>  Base 100 · Verde=estrategia · Gris=benchmarks</sup>",
            font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=520, margin=dict(l=55, r=30, t=65, b=40),
        xaxis=dict(gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"),
                   rangeselector=dict(
                       buttons=[dict(count=1,label="1A",step="year",stepmode="backward"),
                                dict(count=3,label="3A",step="year",stepmode="backward"),
                                dict(count=5,label="5A",step="year",stepmode="backward"),
                                dict(step="all",label="Todo")],
                       bgcolor="#161B22", activecolor="#F7931A",
                       font=dict(size=9, color="#C9D1D9", family="JetBrains Mono"))),
        xaxis2=dict(gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        yaxis=dict(title="Índice (base 100)", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        yaxis2=dict(title="Sharpe 6M", gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"),
                    zeroline=True, zerolinecolor="#484F58"),
        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
        hovermode="x unified")
    return fig


def build_drawdown_chart(result: dict) -> go.Figure:
    """Drawdown diario de la estrategia."""
    eq   = result.get("equity", pd.Series(dtype=float))
    if eq.empty: return go.Figure()
    roll_max = eq.cummax()
    dd       = (eq / roll_max - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Drawdown",
        fill="tozeroy", line=dict(color="#F85149", width=1.5),
        fillcolor="rgba(248,81,73,0.2)",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=-10, line_dash="dot", line_color="#D29922", line_width=1,
                  annotation_text="  -10%", annotation_font=dict(color="#D29922", size=8))
    fig.add_hline(y=-20, line_dash="dash", line_color="#F85149", line_width=1,
                  annotation_text="  -20%", annotation_font=dict(color="#F85149", size=8))
    fig.update_layout(
        title=dict(text="<b>Drawdown Histórico</b>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=280, margin=dict(l=55, r=30, t=55, b=40),
        xaxis=dict(gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        yaxis=dict(title="Drawdown %", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"),
                   ticksuffix="%"),
        showlegend=False, hovermode="x unified")
    return fig


def build_walkforward_chart(wf_df: pd.DataFrame, wf_months: int = 6) -> go.Figure:
    """Sharpe por ventana walk-forward + % tiempo invertido."""
    if wf_df.empty or "sharpe" not in wf_df.columns: return go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)

    # Panel 1: Sharpe por ventana
    xs = wf_df["period_end"].astype(str)
    colors_wf = ["#3FB950" if v >= 1.0 else "#D29922" if v >= 0.5 else "#F85149"
                 for v in wf_df["sharpe"]]
    fig.add_trace(go.Bar(
        x=xs, y=wf_df["sharpe"], name=f"Sharpe {wf_months}M",
        marker_color=colors_wf, opacity=0.8,
        hovertemplate="Fin: %{x}<br>Sharpe: %{y:.2f}<extra></extra>"), row=1, col=1)

    if "bh_sharpe" in wf_df.columns:
        fig.add_trace(go.Scatter(
            x=xs, y=wf_df["bh_sharpe"], name="Sharpe B&H SVXY",
            line=dict(color="#484F58", width=1.5, dash="dot"),
            hovertemplate="B&H Sharpe: %{y:.2f}<extra></extra>"), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="#3FB950", line_width=1.2, row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#D29922", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#484F58", line_width=1, row=1, col=1)

    # Panel 2: % tiempo invertido
    if "pct_invested" in wf_df.columns:
        fig.add_trace(go.Bar(
            x=xs, y=wf_df["pct_invested"], name="% Tiempo invertido",
            marker_color="#58A6FF", opacity=0.6,
            hovertemplate="%{x}<br>Invertido: %{y:.0f}%<extra></extra>"), row=2, col=1)
        if "ct_ratio" in wf_df.columns:
            fig.add_trace(go.Scatter(
                x=xs, y=wf_df["ct_ratio"], name="% Días contango",
                line=dict(color="#39D2C0", width=1.5),
                hovertemplate="Contango: %{y:.0f}%<extra></extra>"), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>Walk-Forward Analysis — Ventanas de {wf_months} Meses</b>"
                 "<sup>  Verde=Sharpe≥1 · Amarillo=0.5-1 · Rojo=<0.5</sup>",
            font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=440, margin=dict(l=55, r=30, t=65, b=40),
        xaxis2=dict(gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono"),
                    tickangle=-30),
        xaxis=dict(gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        yaxis=dict(title="Sharpe", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        yaxis2=dict(title="%", gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E", family="JetBrains Mono")),
        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
        hovermode="x unified", bargap=0.1)
    return fig


def build_monthly_returns_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """Heatmap de retornos mensuales año × mes."""
    if monthly_df.empty: return go.Figure()
    fig = go.Figure()

    strat = monthly_df["Estrategia"]
    periods = strat.index  # "2023-01", "2023-02", …

    # Reshape en matriz año × mes
    years  = sorted(set(p[:4] for p in periods))
    months = [f"{m:02d}" for m in range(1, 13)]
    month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    Z    = np.full((len(years), 12), np.nan)
    text = [[""] * 12 for _ in range(len(years))]

    for p, v in strat.items():
        yr  = p[:4]; mo = int(p[5:7]) - 1
        if yr in years:
            ri = years.index(yr)
            Z[ri, mo]    = round(float(v), 1)
            text[ri][mo] = f"{v:+.1f}%"

    colorscale = [
        [0.0,  "#7B0000"], [0.1,  "#C62828"], [0.25, "#EF5350"],
        [0.40, "#FFCDD2"], [0.50, "#F5F5F5"],
        [0.60, "#C8E6C9"], [0.75, "#4CAF50"], [0.90, "#2E7D32"], [1.0,  "#1B5E20"],
    ]

    fig.add_trace(go.Heatmap(
        z=Z, x=month_names, y=years,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9, color="#F0F6FC", family="JetBrains Mono"),
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(title=dict(text="Ret%", font=dict(color="#8B949E", size=9)),
                      tickfont=dict(color="#8B949E", size=8),
                      len=0.8, thickness=12, ticksuffix="%"),
        hovertemplate="Año: %{y}<br>Mes: %{x}<br>Ret: %{z:.1f}%<extra></extra>",
        xgap=2, ygap=2))

    fig.update_layout(
        title=dict(text="<b>Retornos Mensuales — Estrategia BB×Contango</b>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=max(280, len(years)*38 + 80),
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(tickfont=dict(size=10, color="#C9D1D9", family="JetBrains Mono")),
        yaxis=dict(tickfont=dict(size=10, color="#C9D1D9", family="JetBrains Mono"),
                   autorange="reversed"))
    return fig


def build_trades_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Barras de retorno por trade + curva acumulada."""
    if trades_df.empty: return go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06)

    colors_t = ["#3FB950" if v > 0 else "#F85149" for v in trades_df["ret_pct"]]
    fig.add_trace(go.Bar(
        x=list(range(1, len(trades_df)+1)), y=trades_df["ret_pct"],
        marker_color=colors_t, opacity=0.8, name="Ret % por trade",
        hovertemplate="Trade %{x}<br>%{y:+.1f}%<br>Hold: " +
                      trades_df.get("hold_d", pd.Series()).astype(str).fillna("?") +
                      "d<extra></extra>"), row=1, col=1)

    # Curva acumulada de trades
    cum = (1 + trades_df["ret_pct"]/100).cumprod() * 100
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum)+1)), y=cum, name="Equity por trade (base 100)",
        line=dict(color="#58A6FF", width=2.5),
        hovertemplate="Trade %{x}<br>Eq: %{y:.0f}<extra></extra>"), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#484F58", row=1, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="#484F58", row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>Trades Individuales</b><sup>  Verde=ganador · Rojo=perdedor</sup>",
                   font=dict(size=13, color="#C9D1D9", family="Inter"), x=0.5),
        template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        height=400, margin=dict(l=55, r=30, t=60, b=40),
        xaxis=dict(title="Trade #", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E")),
        xaxis2=dict(title="Trade #", gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E")),
        yaxis=dict(title="Retorno %", gridcolor="#21262D",
                   tickfont=dict(size=9, color="#8B949E"), ticksuffix="%"),
        yaxis2=dict(title="Equity (base 100)", gridcolor="#21262D",
                    tickfont=dict(size=9, color="#8B949E")),
        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9, color="#C9D1D9", family="JetBrains Mono")),
        showlegend=True, hovermode="x unified", bargap=0.15)
    return fig


@st.cache_data(ttl=3600)
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

    # Señal BB pura
    sig = pd.Series(0, index=bt.index)
    pos = 0
    for i in range(len(bt)):
        p = bt['VXX_Close'].iloc[i]
        s = bt['BB_SMA20'].iloc[i]
        u = bt['BB_Upper'].iloc[i]
        if pd.isna(s) or pd.isna(u) or pd.isna(p):
            sig.iloc[i] = pos; continue
        if pos == 0 and p < s:   pos = 1
        elif pos == 1 and p > u: pos = 0
        sig.iloc[i] = pos

    bt['sig_bb']    = sig.shift(1).fillna(0).astype(int)
    bt['ct_filter'] = bt['In_Contango'].fillna(0).astype(int)
    bt['sig_final'] = (bt['sig_bb'] * bt['ct_filter']).astype(int)
    return bt


def build_vxx_operational_chart(bt: pd.DataFrame,
                                 vxx_today: float,
                                 final_sig_today: int,
                                 ct_today: float | None) -> go.Figure:
    """
    Gráfica operativa VXX — versión mejorada v2.

    Tres paneles:
      Panel 1 (55%) : VXX + SMA(20) + BB 2σ + shading por bloques LONG/BKWD
                       + flechas grandes ENTRY/EXIT como scatter markers (visibles)
                       + marker "HOY" grande
      Panel 2 (20%) : Contango % histórico (barras)
      Panel 3 (25%) : Equity SVXY del régimen (la estrategia en sí, no el VXX)

    Mejoras vs v1:
      - Shading por rectángulos (vrect) en lugar de fill tozeroy que tapaba
      - Flechas = markers Scatter (triángulos nativos) con size=14, outline blanco
        → se ven incluso en sparkline, no se pierden como text annotations
      - Tooltip custom que muestra qué regla disparó la salida
      - Panel de equity: muestra cuánto ganó/perdió la estrategia históricamente
      - Título informativo con P&L total, # trades, win rate
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25],
        vertical_spacing=0.035,
        subplot_titles=(
            "<b>VXX · Precio + Bollinger Bands</b>",
            "<b>Contango (M2-M1)/M1 %</b>",
            "<b>Equity Curve — Estrategia BB × Contango sobre VXX inverso</b>",
        ),
    )

    sig    = bt['sig_final'].astype(int)
    sig_bb = bt['sig_bb'].astype(int)
    ct     = bt['ct_filter'].astype(int)
    vxx    = bt['VXX_Close']

    # ══════════════════════════════════════════════════════
    # PANEL 1: VXX + BB + shading + flechas
    # ══════════════════════════════════════════════════════

    # ── Shading por bloques LONG ─────────────────────────
    # Detectar bloques contiguos donde sig==1 y pintar rectangle verde tenue
    in_block = False
    block_start = None
    long_blocks = []
    for dt, s in sig.items():
        if s == 1 and not in_block:
            block_start = dt
            in_block = True
        elif s == 0 and in_block:
            long_blocks.append((block_start, dt))
            in_block = False
    if in_block:
        long_blocks.append((block_start, sig.index[-1]))

    for start, end in long_blocks:
        fig.add_vrect(x0=start, x1=end, row=1, col=1,
                      fillcolor='rgba(63,185,80,0.08)', line_width=0, layer='below')

    # ── Shading por bloques BACKWARDATION (sig_bb=1 pero ct=0) ─
    bkwd_mask = (sig_bb == 1) & (ct == 0)
    in_block = False
    block_start = None
    bkwd_blocks = []
    for dt, m in bkwd_mask.items():
        if m and not in_block:
            block_start = dt; in_block = True
        elif not m and in_block:
            bkwd_blocks.append((block_start, dt)); in_block = False
    if in_block:
        bkwd_blocks.append((block_start, bkwd_mask.index[-1]))
    for start, end in bkwd_blocks:
        fig.add_vrect(x0=start, x1=end, row=1, col=1,
                      fillcolor='rgba(248,81,73,0.07)', line_width=0, layer='below')

    # Proxy traces para aparecer en la leyenda (shading no genera entries)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='rgba(63,185,80,0.35)', symbol='square'),
        name='Zona LONG', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='rgba(248,81,73,0.35)', symbol='square'),
        name='Backwardation', showlegend=True), row=1, col=1)

    # ── BB band (fill entre upper y lower) ────────────────
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_Upper'],
        mode='lines', name='BB 2σ Upper',
        line=dict(color='#F85149', width=1, dash='dot'),
        hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_Lower'],
        mode='lines', name='BB 2σ Lower', showlegend=False,
        line=dict(color='#F85149', width=0.7, dash='dot'),
        fill='tonexty', fillcolor='rgba(248,81,73,0.04)',
        hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_SMA20'],
        mode='lines', name='SMA(20)',
        line=dict(color='#58A6FF', width=1.5, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d} · SMA: $%{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=vxx,
        mode='lines', name='VXX',
        line=dict(color='#F0F6FC', width=2.2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>VXX: <b>$%{y:.2f}</b><extra></extra>'),
        row=1, col=1)

    # ── Flechas como SCATTER MARKERS (visibles) ──────────
    entry_dates, entry_prices = [], []
    exit_bb_dates, exit_bb_prices = [], []
    exit_ct_dates, exit_ct_prices = [], []

    sig_arr    = sig.values
    sig_bb_arr = sig_bb.values
    ct_arr     = ct.values
    idx_arr    = sig.index
    vxx_arr    = vxx.values

    for i in range(1, len(sig_arr)):
        prev_s, cur_s   = sig_arr[i-1], sig_arr[i]
        prev_bb, cur_bb = sig_bb_arr[i-1], sig_bb_arr[i]
        prev_ct, cur_ct = ct_arr[i-1], ct_arr[i]

        if cur_s == 1 and prev_s == 0:
            entry_dates.append(idx_arr[i])
            entry_prices.append(vxx_arr[i])
        elif cur_s == 0 and prev_s == 1:
            # Determinar regla que disparó salida
            if cur_bb == 0 and prev_bb == 1:
                exit_bb_dates.append(idx_arr[i])
                exit_bb_prices.append(vxx_arr[i])
            elif cur_ct == 0 and prev_ct == 1:
                exit_ct_dates.append(idx_arr[i])
                exit_ct_prices.append(vxx_arr[i])
            else:
                exit_bb_dates.append(idx_arr[i])
                exit_bb_prices.append(vxx_arr[i])

    n_entries = len(entry_dates)

    # Entries (triángulo verde hacia arriba, debajo del precio)
    if entry_dates:
        fig.add_trace(go.Scatter(
            x=entry_dates,
            y=[p * 0.94 for p in entry_prices],  # ligeramente debajo
            mode='markers',
            name=f'▲ Entrada ({n_entries})',
            marker=dict(size=14, color='#3FB950', symbol='triangle-up',
                        line=dict(width=1.5, color='#FFFFFF')),
            hovertemplate='<b>ENTRADA</b><br>%{x|%Y-%m-%d}<br>VXX: $%{customdata:.2f}<extra></extra>',
            customdata=entry_prices,
        ), row=1, col=1)

    # Exits por BB (triángulo amarillo hacia abajo, encima del precio)
    if exit_bb_dates:
        fig.add_trace(go.Scatter(
            x=exit_bb_dates,
            y=[p * 1.06 for p in exit_bb_prices],
            mode='markers',
            name=f'▼ Salida BB ({len(exit_bb_dates)})',
            marker=dict(size=14, color='#D29922', symbol='triangle-down',
                        line=dict(width=1.5, color='#FFFFFF')),
            hovertemplate='<b>SALIDA · BB 2σ</b><br>%{x|%Y-%m-%d}<br>VXX: $%{customdata:.2f}<extra></extra>',
            customdata=exit_bb_prices,
        ), row=1, col=1)

    # Exits por Contango Rule
    if exit_ct_dates:
        fig.add_trace(go.Scatter(
            x=exit_ct_dates,
            y=[p * 1.06 for p in exit_ct_prices],
            mode='markers',
            name=f'▼ Salida CT ({len(exit_ct_dates)})',
            marker=dict(size=14, color='#F85149', symbol='triangle-down',
                        line=dict(width=1.5, color='#FFFFFF')),
            hovertemplate='<b>SALIDA · Contango</b><br>%{x|%Y-%m-%d}<br>VXX: $%{customdata:.2f}<extra></extra>',
            customdata=exit_ct_prices,
        ), row=1, col=1)

    # Punto HOY (grande, con halo)
    today_clr = '#3FB950' if final_sig_today else '#F85149'
    today_lbl = 'HOY · LONG' if final_sig_today else 'HOY · CASH'
    # Halo (marker grande semi-transparente)
    fig.add_trace(go.Scatter(
        x=[bt.index[-1]], y=[vxx_today],
        mode='markers', showlegend=False,
        marker=dict(size=26, color=today_clr, opacity=0.25, symbol='circle'),
        hoverinfo='skip',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[bt.index[-1]], y=[vxx_today],
        mode='markers', name=today_lbl,
        marker=dict(size=14, color=today_clr,
                    line=dict(width=2.5, color='white'), symbol='diamond'),
        hovertemplate=f'<b>{today_lbl}</b><br>VXX: ${vxx_today:.2f}<extra></extra>',
    ), row=1, col=1)

    # ══════════════════════════════════════════════════════
    # PANEL 2: Contango histórico
    # ══════════════════════════════════════════════════════
    if 'Contango_pct' in bt.columns:
        ct_hist  = bt['Contango_pct'].fillna(0)
        bar_clrs = ['#3FB950' if v > 0 else '#F85149' for v in ct_hist]
        fig.add_trace(go.Bar(
            x=bt.index, y=ct_hist,
            name='Contango % hist', marker_color=bar_clrs, opacity=0.75,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Contango: %{y:+.2f}%<extra></extra>',
            showlegend=False,
        ), row=2, col=1)
        if ct_today is not None:
            ct_clr = '#3FB950' if ct_today > 0 else '#F85149'
            fig.add_trace(go.Scatter(
                x=[bt.index[-1]], y=[ct_today],
                mode='markers', name=f'CT hoy: {ct_today:+.2f}%',
                marker=dict(size=12, color=ct_clr, symbol='diamond',
                            line=dict(width=2, color='white')),
                showlegend=False,
            ), row=2, col=1)
        fig.add_hline(y=0, line_color='#484F58', line_width=1, row=2, col=1)

    # ══════════════════════════════════════════════════════
    # PANEL 3: Equity Curve del régimen (estrategia inversa sobre VXX)
    # Cuando sig_final==1, capturamos -VXX return (porque estamos LONG SVXY ~ -0.5x VXX)
    # ══════════════════════════════════════════════════════
    vxx_ret    = vxx.pct_change().fillna(0)
    # Posición aplicada con shift=1 (señal de cierre, ejecutamos al open siguiente)
    pos        = sig.shift(1).fillna(0)
    # SVXY ~ -0.5x VXX (intraday), aproximación razonable
    strat_ret  = -0.5 * vxx_ret * pos
    equity     = (1.0 + strat_ret).cumprod()

    # Benchmark buy-and-hold VXX (inverso para comparar "qué tan mejor")
    bh_ret    = -0.5 * vxx_ret  # si siempre hubiéramos estado LONG SVXY
    bh_equity = (1.0 + bh_ret).cumprod()

    fig.add_trace(go.Scatter(
        x=bt.index, y=bh_equity,
        mode='lines', name='Buy & Hold SVXY',
        line=dict(color='#8B949E', width=1.2, dash='dash'),
        hovertemplate='B&H: %{y:.3f}x<extra></extra>',
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=bt.index, y=equity,
        mode='lines', name='Estrategia BB × CT',
        line=dict(color='#39D2C0', width=2.2),
        fill='tozeroy', fillcolor='rgba(57,210,192,0.08)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Equity: %{y:.3f}x<extra></extra>',
    ), row=3, col=1)
    fig.add_hline(y=1.0, line_color='#484F58', line_width=1,
                  line_dash='dot', row=3, col=1)

    # ── Stats para el título ──────────────────────────────
    final_eq  = equity.iloc[-1]
    bh_eq     = bh_equity.iloc[-1]
    total_ret = (final_eq - 1) * 100
    # Días en LONG
    days_long = int(sig.sum())
    pct_long  = days_long / len(sig) * 100 if len(sig) else 0
    # Win rate por trade (por bloque LONG)
    wins = 0
    trades = 0
    for start, end in long_blocks:
        if start in equity.index and end in equity.index:
            r = equity.loc[end] / equity.loc[start] - 1
            trades += 1
            if r > 0:
                wins += 1
    win_rate = (wins / trades * 100) if trades else 0

    # ══════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════
    title_html = (
        f"<b>VXX — Monitor Operativo · BB(20, 2σ) × Contango Rule</b>"
        f"<br><span style='font-size:0.7rem;color:#8B949E;font-family:JetBrains Mono'>"
        f"Trades: <b style='color:#58A6FF'>{trades}</b> · "
        f"Win Rate: <b style='color:{'#3FB950' if win_rate>=50 else '#D29922'}'>{win_rate:.0f}%</b> · "
        f"Días LONG: <b style='color:#3FB950'>{pct_long:.0f}%</b> · "
        f"Retorno estrategia: <b style='color:{'#3FB950' if total_ret>=0 else '#F85149'}'>{total_ret:+.0f}%</b> · "
        f"vs B&H SVXY: <b style='color:#F0F6FC'>{(bh_eq-1)*100:+.0f}%</b>"
        f"</span>"
    )

    fig.update_layout(
        title=dict(text=title_html, font=dict(size=14, color='#F0F6FC',
                    family='Inter'), x=0.5, xanchor='center'),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=720, margin=dict(l=60, r=30, t=95, b=45),
        hovermode='x unified', dragmode='zoom', bargap=0,
        legend=dict(orientation='h', yanchor='bottom', y=1.04, x=0.5, xanchor='center',
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9.5, color='#C9D1D9', family='JetBrains Mono')),
    )

    # Subplot titles (las subtítulos generados por make_subplots)
    for annotation in fig['layout']['annotations'][:3]:
        annotation['font'] = dict(size=11, color='#8B949E', family='Inter')
        annotation['xanchor'] = 'left'
        annotation['x'] = 0.01

    # Axes
    fig.update_xaxes(
        gridcolor='#21262D', showgrid=True,
        tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono'),
    )
    fig.update_xaxes(
        row=1, col=1,
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1A",  step="year",  stepmode="backward"),
                dict(count=3,  label="3A",  step="year",  stepmode="backward"),
                dict(step="all", label="Todo"),
            ],
            bgcolor='#161B22', activecolor='#F7931A', bordercolor='#30363D',
            font=dict(size=9, color='#C9D1D9', family='JetBrains Mono'),
            y=1.12,
        ),
    )
    fig.update_yaxes(gridcolor='#21262D',
                     tickfont=dict(size=9.5, color='#8B949E', family='JetBrains Mono'))
    fig.update_yaxes(title=dict(text="VXX ($)", font=dict(size=10, color='#8B949E')),
                     row=1, col=1)
    fig.update_yaxes(title=dict(text="CT (%)", font=dict(size=10, color='#8B949E')),
                     zeroline=True, zerolinecolor='#30363D', row=2, col=1)
    fig.update_yaxes(title=dict(text="Equity ($1 → x)", font=dict(size=10, color='#8B949E')),
                     row=3, col=1)

    return fig


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

def _rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Percentil rolling del último valor vs la ventana histórica.
    Devuelve 0-100. Usa método 'rank' — 0 = mínimo hist, 100 = máximo hist.
    """
    return s.rolling(window, min_periods=max(30, window // 5)).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) * 100 if len(x) > 0 else np.nan,
        raw=False,
    )


@st.cache_data(ttl=300)
def compute_vts_barometer(
    bt: pd.DataFrame,
    edge_extra: dict,
    gex_summary: dict | None = None,
    skew_metrics: dict | None = None,
    window: int = 252,
) -> dict:
    """
    Calcula el VTS Volatility Barometer — 13 métricas promediadas a score 0-100.

    Parameters
    ----------
    bt : DataFrame con VIX_Close, SPY_Close, M1_Price, M2_Price,
         Contango_pct, VXX_Close, BB_SMA20, VVIX_Live (opcional)
    edge_extra : dict de DataFrames de fetch_edge_extra() — VVIX, SKEW, HYG, IEF
    gex_summary : dict opcional con net_gex (del tab GEX)
    skew_metrics : dict opcional con skew_25d (del tab Vol Skew)
    window : ventana rolling para percentiles (252 = 1 año)

    Returns
    -------
    dict con:
      - score : float 0-100 (el barómetro VTS)
      - regime : str ("LOW", "MID", "ELEVATED", "EXTREME")
      - position : str (recomendación operativa)
      - metrics : list[dict] con cada métrica, su valor y su percentil
      - history : pd.Series del score histórico (últimos 252 días)
    """
    if bt.empty or len(bt) < 60:
        return {}

    df = bt.copy()
    metrics = []

    # ══════════════════════════════════════════════════════
    # MÉTRICA 1 — VIX SPOT LEVEL
    # Percentil alto = vol alta
    # ══════════════════════════════════════════════════════
    if 'VIX_Close' in df.columns:
        vix_pct = _rolling_percentile(df['VIX_Close'], window)
        last_val = df['VIX_Close'].iloc[-1]
        last_pct = vix_pct.iloc[-1] if pd.notna(vix_pct.iloc[-1]) else 50
        metrics.append({
            'name': 'VIX Spot Level',
            'value': f"{last_val:.2f}",
            'percentile': last_pct,
            'weight': 1.2,  # más peso — es la señal primaria
            'interpretation': 'Nivel absoluto de volatilidad implícita',
            'series': vix_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 2 — VIX/VIX3M RATIO (term structure inversion)
    # ratio > 1 = backwardation (stress). Percentil alto = vol alta
    # ══════════════════════════════════════════════════════
    # Proxy: 1 - Contango_pct/100 es ~VIX/M1 spread
    # Mejor: usar VIX / M1_Price directamente
    if 'VIX_Close' in df.columns and 'M1_Price' in df.columns:
        vix_m1 = df['VIX_Close'] / df['M1_Price']
        vm1_pct = _rolling_percentile(vix_m1, window)
        metrics.append({
            'name': 'VIX / VIX-Fut M1',
            'value': f"{vix_m1.iloc[-1]:.3f}",
            'percentile': vm1_pct.iloc[-1] if pd.notna(vm1_pct.iloc[-1]) else 50,
            'weight': 1.0,
            'interpretation': '>1 indica backwardation (stress a corto plazo)',
            'series': vm1_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 3 — VVIX (vol de vol) — percentil alto = stress
    # ══════════════════════════════════════════════════════
    vvix_s = None
    if 'VVIX_Live' in df.columns and df['VVIX_Live'].notna().sum() > 60:
        vvix_s = df['VVIX_Live']
    elif 'VVIX' in edge_extra and not edge_extra['VVIX'].empty:
        vvix_s = edge_extra['VVIX']['Close'].reindex(df.index).ffill()
    if vvix_s is not None and vvix_s.notna().sum() > 60:
        vvix_pct = _rolling_percentile(vvix_s, window)
        metrics.append({
            'name': 'VVIX (vol del VIX)',
            'value': f"{vvix_s.iloc[-1]:.1f}" if pd.notna(vvix_s.iloc[-1]) else '—',
            'percentile': vvix_pct.iloc[-1] if pd.notna(vvix_pct.iloc[-1]) else 50,
            'weight': 1.0,
            'interpretation': 'Demanda de protección via opciones sobre VIX',
            'series': vvix_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 4 — CONTANGO M1-M2 (inverted: alto contango = vol baja)
    # INVERTIDO: vol alta = bajo contango/backwardation = 100-contango_pct
    # ══════════════════════════════════════════════════════
    if 'Contango_pct' in df.columns:
        ct_pct = _rolling_percentile(df['Contango_pct'], window)
        # Invertido: contango alto = vol baja = score bajo
        inv_pct = 100 - ct_pct
        metrics.append({
            'name': 'Contango M1-M2 (inv)',
            'value': f"{df['Contango_pct'].iloc[-1]:+.2f}%",
            'percentile': inv_pct.iloc[-1] if pd.notna(inv_pct.iloc[-1]) else 50,
            'weight': 1.2,  # clave para la estrategia
            'interpretation': 'Contango alto = vol baja (invertido para el score)',
            'series': inv_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 5 — VXX MOMENTUM vs SMA(20)
    # VXX > SMA significa vol subiendo. Percentil del ratio.
    # ══════════════════════════════════════════════════════
    if 'VXX_Close' in df.columns and 'BB_SMA20' in df.columns:
        vxx_mom = df['VXX_Close'] / df['BB_SMA20']
        vxx_mom_pct = _rolling_percentile(vxx_mom, window)
        metrics.append({
            'name': 'VXX / SMA(20)',
            'value': f"{vxx_mom.iloc[-1]:.3f}",
            'percentile': vxx_mom_pct.iloc[-1] if pd.notna(vxx_mom_pct.iloc[-1]) else 50,
            'weight': 0.9,
            'interpretation': '>1 = momentum alcista en vol',
            'series': vxx_mom_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 6 — SPY REALIZED VOL 22d (annualizada)
    # ══════════════════════════════════════════════════════
    if 'SPY_Close' in df.columns:
        log_ret = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        rv22 = log_ret.rolling(22).std() * np.sqrt(252) * 100
        rv_pct = _rolling_percentile(rv22, window)
        metrics.append({
            'name': 'SPY RV 22d (ann.)',
            'value': f"{rv22.iloc[-1]:.2f}%" if pd.notna(rv22.iloc[-1]) else '—',
            'percentile': rv_pct.iloc[-1] if pd.notna(rv_pct.iloc[-1]) else 50,
            'weight': 1.0,
            'interpretation': 'Volatilidad realizada del SPY últimos 22 días',
            'series': rv_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 7 — VRP (VIX - RV) — Risk Premium
    # VRP bajo o negativo = vol siendo exigida fuerte = stress
    # INVERTIDO: vrp bajo = stress alto
    # ══════════════════════════════════════════════════════
    if 'SPY_Close' in df.columns and 'VIX_Close' in df.columns:
        log_ret = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        rv22 = log_ret.rolling(22).std() * np.sqrt(252) * 100
        vrp = df['VIX_Close'] - rv22
        vrp_pct = _rolling_percentile(vrp, window)
        # Invertido: VRP bajo = stress
        inv_vrp = 100 - vrp_pct
        metrics.append({
            'name': 'VRP (VIX-RV) inv',
            'value': f"{vrp.iloc[-1]:+.2f}" if pd.notna(vrp.iloc[-1]) else '—',
            'percentile': inv_vrp.iloc[-1] if pd.notna(inv_vrp.iloc[-1]) else 50,
            'weight': 0.9,
            'interpretation': 'VRP bajo = mercado exige prima alta por vol',
            'series': inv_vrp,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 8 — SKEW INDEX (CBOE)
    # SKEW alto = cola izquierda cara = demanda de puts OTM
    # ══════════════════════════════════════════════════════
    if 'SKEW' in edge_extra and not edge_extra['SKEW'].empty:
        skew_s = edge_extra['SKEW']['Close'].reindex(df.index).ffill()
        if skew_s.notna().sum() > 60:
            skew_pct = _rolling_percentile(skew_s, window)
            metrics.append({
                'name': 'CBOE SKEW Index',
                'value': f"{skew_s.iloc[-1]:.1f}" if pd.notna(skew_s.iloc[-1]) else '—',
                'percentile': skew_pct.iloc[-1] if pd.notna(skew_pct.iloc[-1]) else 50,
                'weight': 0.8,
                'interpretation': 'Demanda de puts OTM (cola izquierda del SPX)',
                'series': skew_pct,
            })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 9 — HYG/IEF ratio (credit spread proxy)
    # HYG cae vs IEF = credit stress. INVERTIDO.
    # ══════════════════════════════════════════════════════
    if ('HYG' in edge_extra and not edge_extra['HYG'].empty and
        'IEF' in edge_extra and not edge_extra['IEF'].empty):
        hyg_s = edge_extra['HYG']['Close'].reindex(df.index).ffill()
        ief_s = edge_extra['IEF']['Close'].reindex(df.index).ffill()
        hyg_ief = hyg_s / ief_s
        if hyg_ief.notna().sum() > 60:
            hyg_pct = _rolling_percentile(hyg_ief, window)
            # Invertido: ratio bajo = credit stress
            inv_hyg = 100 - hyg_pct
            metrics.append({
                'name': 'HYG/IEF (credit, inv)',
                'value': f"{hyg_ief.iloc[-1]:.3f}" if pd.notna(hyg_ief.iloc[-1]) else '—',
                'percentile': inv_hyg.iloc[-1] if pd.notna(inv_hyg.iloc[-1]) else 50,
                'weight': 0.7,
                'interpretation': 'HYG/IEF cae → credit spread se abre → stress',
                'series': inv_hyg,
            })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 10 — SPY 20d Drawdown
    # Drawdown profundo = stress. Usamos -dd para que sea positivo=stress
    # ══════════════════════════════════════════════════════
    if 'SPY_Close' in df.columns:
        roll_max = df['SPY_Close'].rolling(20, min_periods=5).max()
        dd20 = (df['SPY_Close'] / roll_max - 1) * 100  # negativo
        dd_inv = -dd20  # positivo
        dd_pct = _rolling_percentile(dd_inv, window)
        metrics.append({
            'name': 'SPY Drawdown 20d',
            'value': f"{dd20.iloc[-1]:.2f}%" if pd.notna(dd20.iloc[-1]) else '—',
            'percentile': dd_pct.iloc[-1] if pd.notna(dd_pct.iloc[-1]) else 50,
            'weight': 0.8,
            'interpretation': 'Profundidad del drawdown reciente',
            'series': dd_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 11 — VIX 5d momentum (vol subiendo rápido)
    # ══════════════════════════════════════════════════════
    if 'VIX_Close' in df.columns:
        vix_5d = df['VIX_Close'].pct_change(5) * 100
        vix_5d_pct = _rolling_percentile(vix_5d, window)
        metrics.append({
            'name': 'VIX 5d Change %',
            'value': f"{vix_5d.iloc[-1]:+.2f}%" if pd.notna(vix_5d.iloc[-1]) else '—',
            'percentile': vix_5d_pct.iloc[-1] if pd.notna(vix_5d_pct.iloc[-1]) else 50,
            'weight': 0.7,
            'interpretation': 'Velocidad de subida/bajada del VIX',
            'series': vix_5d_pct,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 12 — Contango M4-M7 (parte larga de la curva)
    # Inversión aquí es más grave. INVERTIDO.
    # ══════════════════════════════════════════════════════
    if 'M4_Price' in df.columns and 'M7_Price' in df.columns:
        long_ct = (df['M7_Price'] - df['M4_Price']) / df['M4_Price'] * 100
        long_ct_pct = _rolling_percentile(long_ct, window)
        inv_long = 100 - long_ct_pct
        metrics.append({
            'name': 'Contango M4-M7 (inv)',
            'value': f"{long_ct.iloc[-1]:+.2f}%" if pd.notna(long_ct.iloc[-1]) else '—',
            'percentile': inv_long.iloc[-1] if pd.notna(inv_long.iloc[-1]) else 50,
            'weight': 0.8,
            'interpretation': 'Curva larga — inversión aquí es stress estructural',
            'series': inv_long,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA 13 — SPY/SMA(200) distance (tendencia macro)
    # Por debajo de SMA(200) = mercado bajista = stress
    # INVERTIDO: distancia positiva = vol baja
    # ══════════════════════════════════════════════════════
    if 'SPY_Close' in df.columns:
        sma200 = df['SPY_Close'].rolling(200, min_periods=50).mean()
        spy_dist = (df['SPY_Close'] / sma200 - 1) * 100
        spy_dist_pct = _rolling_percentile(spy_dist, window)
        # Invertido: distancia alta = bull = vol baja
        inv_dist = 100 - spy_dist_pct
        metrics.append({
            'name': 'SPY vs SMA(200) inv',
            'value': f"{spy_dist.iloc[-1]:+.2f}%" if pd.notna(spy_dist.iloc[-1]) else '—',
            'percentile': inv_dist.iloc[-1] if pd.notna(inv_dist.iloc[-1]) else 50,
            'weight': 0.6,
            'interpretation': 'SPY debajo de SMA(200) = bear regime',
            'series': inv_dist,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA OPCIONAL 14 — GEX (si está disponible)
    # GEX negativo = dealers short gamma = movimientos amplificados = stress
    # ══════════════════════════════════════════════════════
    if gex_summary and 'net_gex' in gex_summary:
        ng = gex_summary.get('net_gex', 0)
        # Simple mapping: gex muy negativo (-3B+) = 90pct, gex muy positivo (+3B+) = 10pct
        # Normalización grosera: percentil basado en thresholds típicos del SPX
        if ng < -3e9:     gex_p = 90
        elif ng < -1e9:   gex_p = 75
        elif ng < 0:      gex_p = 60
        elif ng < 1e9:    gex_p = 45
        elif ng < 3e9:    gex_p = 30
        else:             gex_p = 15
        metrics.append({
            'name': 'Net GEX (SPX)',
            'value': f"{ng/1e9:+.2f}B",
            'percentile': gex_p,
            'weight': 0.7,
            'interpretation': 'GEX negativo = dealers amplifican movimientos',
            'series': None,
        })

    # ══════════════════════════════════════════════════════
    # MÉTRICA OPCIONAL 15 — Skew 25Δ (si está disponible)
    # Skew alto = puts caros vs calls = demanda de protección = stress
    # ══════════════════════════════════════════════════════
    if skew_metrics and 'atm_iv' in skew_metrics:
        # Si hay skew data, usar como métrica adicional
        skew_val = skew_metrics.get('skew_25d', None)
        if skew_val is not None:
            # Mapping simple: skew > 3% = alto stress
            if skew_val > 5:      sk_p = 85
            elif skew_val > 3:    sk_p = 70
            elif skew_val > 1:    sk_p = 50
            elif skew_val > 0:    sk_p = 30
            else:                 sk_p = 15
            metrics.append({
                'name': 'Put/Call Skew 25Δ',
                'value': f"{skew_val:+.2f}%",
                'percentile': sk_p,
                'weight': 0.6,
                'interpretation': 'Skew positivo = puts caros vs calls',
                'series': None,
            })

    # ══════════════════════════════════════════════════════
    # SCORE FINAL — promedio ponderado
    # ══════════════════════════════════════════════════════
    total_w    = sum(m['weight'] for m in metrics if pd.notna(m['percentile']))
    weighted_s = sum(m['percentile'] * m['weight']
                     for m in metrics if pd.notna(m['percentile']))
    score = weighted_s / total_w if total_w > 0 else 50.0

    # Régimen
    if   score < 20: regime, position = "VOL BAJA",     "Aggressive short vol (SVXY/SVIX)"
    elif score < 40: regime, position = "MODERADA",     "Short vol estándar (SVXY)"
    elif score < 60: regime, position = "MID",          "Cash / posición parcial"
    elif score < 80: regime, position = "ELEVADA",      "Cash / defensivo"
    else:            regime, position = "EXTREMA",      "Long VIX / hedge / short equities"

    # Histórico del score (para timeline)
    # Calculamos score histórico promediando las series de percentiles
    score_hist = None
    series_list = [(m['series'], m['weight']) for m in metrics
                    if m.get('series') is not None]
    if series_list:
        weighted_df = pd.concat(
            [s * w for s, w in series_list], axis=1
        ).sum(axis=1)
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
            font=dict(size=40, color='#F0F6FC', family='Inter'),
            valueformat='.2f',
        ),
        domain={'x': [0, 1], 'y': [0, 1]},
        title=dict(
            text=f"<span style='font-size:0.85rem;color:#8B949E;font-family:JetBrains Mono'>"
                 f"{date_str}</span><br>"
                 f"<b style='font-size:1.1rem;color:#F0F6FC;font-family:Inter'>"
                 f"VTS Volatility Barometer</b>",
            font=dict(color='#F0F6FC'),
        ),
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': '#8B949E',
                'tickfont': dict(size=11, color='#8B949E', family='JetBrains Mono'),
                'tickmode': 'array',
                'tickvals': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'ticktext': ['0%','10%','20%','30%','40%','50%',
                             '60%','70%','80%','90%','100%'],
            },
            'bar': {'color': 'rgba(0,0,0,0)', 'thickness': 0.0},
            # Sin borde grueso — el fondo del gauge es limpio
            'bgcolor': '#0D1117',
            'borderwidth': 0,
            'bordercolor': '#0D1117',
            'steps': [
                {'range': [0, 20],   'color': '#2EA043'},    # verde fuerte
                {'range': [20, 35],  'color': '#3FB950'},    # verde
                {'range': [35, 50],  'color': '#85E89D'},    # verde claro
                {'range': [50, 65],  'color': '#FFD33D'},    # amarillo
                {'range': [65, 80],  'color': '#FB8500'},    # naranja
                {'range': [80, 90],  'color': '#F85149'},    # rojo
                {'range': [90, 100], 'color': '#B60205'},    # rojo oscuro
            ],
            'threshold': {
                'line': {'color': '#0D1117', 'width': 8},
                'thickness': 0.85,
                'value': score,
            },
        },
    ))

    # Etiqueta de régimen debajo
    if   score < 20: clr = '#2EA043'
    elif score < 40: clr = '#3FB950'
    elif score < 60: clr = '#FFD33D'
    elif score < 80: clr = '#FB8500'
    else:            clr = '#F85149'

    fig.add_annotation(
        x=0.5, y=-0.05, xref='paper', yref='paper',
        text=f"<b style='font-size:1.3rem;color:{clr};font-family:Inter'>{regime}</b>",
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0D1117',
        font=dict(color='#C9D1D9'),
        height=420,
        margin=dict(l=30, r=30, t=80, b=60),
    )
    return fig


def build_vts_metrics_table(metrics: list) -> go.Figure:
    """
    Tabla horizontal de cada métrica con su percentil como barra de progreso.
    Similar al desglose que usa VTS para justificar el score.
    """
    if not metrics:
        return go.Figure()

    n = len(metrics)
    # Barras horizontales ordenadas por percentil descendente
    sorted_m = sorted(metrics, key=lambda m: m['percentile'] if pd.notna(m['percentile']) else -1,
                      reverse=True)

    names  = [m['name'] for m in sorted_m]
    pctls  = [m['percentile'] for m in sorted_m]
    vals   = [m['value']  for m in sorted_m]
    wts    = [m['weight'] for m in sorted_m]

    # Color por bucket
    def bucket_color(p):
        if pd.isna(p): return '#484F58'
        if p < 20:  return '#2EA043'
        if p < 40:  return '#3FB950'
        if p < 60:  return '#FFD33D'
        if p < 80:  return '#FB8500'
        return '#F85149'

    bar_colors = [bucket_color(p) for p in pctls]

    fig = go.Figure()

    # Fondo: barra gris hasta 100
    fig.add_trace(go.Bar(
        x=[100] * n, y=names, orientation='h',
        marker=dict(color='#161B22', line=dict(width=0)),
        showlegend=False, hoverinfo='skip',
        width=0.65,
    ))
    # Barra de percentil real
    fig.add_trace(go.Bar(
        x=pctls, y=names, orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{p:.0f}% · {v} · w={w:.1f}" if pd.notna(p) else '—'
              for p, v, w in zip(pctls, vals, wts)],
        textposition='inside', insidetextanchor='start',
        textfont=dict(size=10, color='#F0F6FC', family='JetBrains Mono'),
        showlegend=False,
        hovertemplate='<b>%{y}</b><br>Percentil: %{x:.1f}%<extra></extra>',
        width=0.65,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Desglose del Barómetro — Percentil rolling por métrica</b>"
                 "<br><span style='font-size:0.7rem;color:#8B949E;font-family:JetBrains Mono'>"
                 "Ordenado por nivel de stress · Verde = vol baja · Rojo = vol alta"
                 "</span>",
            font=dict(size=13, color='#F0F6FC', family='Inter'), x=0.5, xanchor='center',
        ),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#0D1117',
        barmode='overlay',
        height=max(360, n * 34 + 110),
        margin=dict(l=160, r=30, t=80, b=40),
        xaxis=dict(
            range=[0, 100], showgrid=True, gridcolor='#21262D',
            tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono'),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
        ),
        yaxis=dict(
            tickfont=dict(size=10, color='#C9D1D9', family='JetBrains Mono'),
            autorange='reversed',
        ),
        showlegend=False,
    )

    # Línea vertical en 50%
    fig.add_vline(x=50, line_color='#30363D', line_dash='dot', line_width=1)

    return fig


def build_vts_history_chart(history: pd.Series, window: int = 252) -> go.Figure:
    """
    Timeline del score del barómetro con bandas de color para cada régimen.
    """
    fig = go.Figure()

    if history is None or history.empty:
        return fig

    # Mostrar solo el último año para claridad
    h = history.tail(window)

    # Bandas de régimen (horizontales)
    for y0, y1, color, label in [
        (0, 20,   'rgba(46,160,67,0.10)',   'Vol Baja'),
        (20, 40,  'rgba(63,185,80,0.08)',   'Moderada'),
        (40, 60,  'rgba(255,211,61,0.08)',  'Mid'),
        (60, 80,  'rgba(251,133,0,0.08)',   'Elevada'),
        (80, 100, 'rgba(248,81,73,0.10)',   'Extrema'),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer='below')

    # Línea del score
    fig.add_trace(go.Scatter(
        x=h.index, y=h.values,
        mode='lines', name='Barómetro VTS',
        line=dict(color='#58A6FF', width=2.2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.06)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}%<extra></extra>',
    ))

    # Punto actual
    fig.add_trace(go.Scatter(
        x=[h.index[-1]], y=[h.iloc[-1]],
        mode='markers', name='HOY',
        marker=dict(size=14, color='#F7931A', symbol='diamond',
                    line=dict(width=2, color='white')),
        showlegend=False,
    ))

    # Línea de media histórica
    mean_val = h.mean()
    fig.add_hline(y=mean_val, line_color='#F7931A', line_dash='dash',
                  line_width=1, annotation_text=f"Media: {mean_val:.1f}%",
                  annotation_position='top right',
                  annotation_font=dict(size=9, color='#F7931A'))

    fig.update_layout(
        title=dict(
            text=f"<b>Histórico del Barómetro — últimos {len(h)} días de trading</b>",
            font=dict(size=13, color='#F0F6FC', family='Inter'),
            x=0.5, xanchor='center',
        ),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=380,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(
            range=[0, 100], gridcolor='#21262D',
            tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono'),
            title=dict(text="Score %", font=dict(size=10, color='#8B949E')),
            tickvals=[0, 20, 40, 60, 80, 100],
        ),
        hovermode='x unified', showlegend=False,
    )
    return fig


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
            name='Last', line=dict(color='#4A90D9', width=3, shape='spline'),
            marker=dict(size=9, color='#4A90D9', line=dict(width=2, color='#0D1117')),
            text=[f"{v:.3f}" for v in vy],
            textposition='top center',
            textfont=dict(size=10, color='#C9D1D9', family='JetBrains Mono'),
            hovertemplate='%{text}<extra></extra>',
        ))

    # Previous day
    if show_prev:
        pvx = [x for x, y in zip(xpos, prev_prices) if y and y > 0]
        pvy = [y for y in prev_prices if y and y > 0]
        if len(pvy) >= 2:
            fig.add_trace(go.Scatter(
                x=pvx, y=pvy, mode='lines+markers',
                name='Previous Close',
                line=dict(color='#8B949E', width=1.5, dash='dot', shape='spline'),
                marker=dict(size=5, color='#8B949E', symbol='diamond'),
                hovertemplate='Prev: %{y:.3f}<extra></extra>',
            ))

    # VIX Index dashed line
    if vix_spot:
        fig.add_hline(y=vix_spot['price'], line_dash="dash", line_color="#3FB950", line_width=2,
                      annotation_text=f"  {vix_spot['price']:.2f}",
                      annotation_position="right",
                      annotation_font=dict(size=12, color="#3FB950", family="Inter"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='VIX Index',
                                 line=dict(color='#3FB950', width=2, dash='dash'), showlegend=True))

    all_y = vy + ([vix_spot['price']] if vix_spot else [])
    y_min = min(all_y) - 1.5 if all_y else 15
    y_max = max(all_y) + 1.5 if all_y else 30

    fig.update_layout(
        title=dict(
            text="<b>VIX Futures Term Structure</b><br><sup>Source: CBOE Delayed Quotes · vixcontroller</sup>",
            font=dict(size=15, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=420, margin=dict(l=50, r=30, t=65, b=50),
        xaxis=dict(tickvals=xpos, ticktext=labels,
                   tickfont=dict(size=11, color='#8B949E', family='JetBrains Mono'),
                   gridcolor='#21262D', showline=True, linecolor='#30363D',
                   title=dict(text="Future Month", font=dict(size=11, color='#8B949E', family='Inter'))),
        yaxis=dict(range=[y_min, y_max],
                   title=dict(text="Volatility", font=dict(size=11, color='#8B949E', family='Inter')),
                   tickfont=dict(size=11, color='#8B949E', family='JetBrains Mono'),
                   gridcolor='#21262D', showline=True, linecolor='#30363D'),
        legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor='rgba(22,27,34,0.9)', bordercolor='#30363D', borderwidth=1,
                    font=dict(size=10, color='#C9D1D9', family='JetBrains Mono')),
        hoverlabel=dict(bgcolor='#1C2128', bordercolor='#58A6FF',
                        font=dict(size=11, family='JetBrains Mono', color='#C9D1D9')),
        hovermode='x unified',
    )
    return fig





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIN AUTO-REFRESH — solo botón manual en sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
now_str = now_cdmx().strftime("%Y-%m-%d %H:%M:%S") + " CDMX"
_h = now_cdmx().hour + now_cdmx().minute / 60
mkt_status = "MARKET OPEN" if 8.5 <= _h < 15 and now_cdmx().weekday() < 5 else "MARKET CLOSED"
mkt_clr = "#3FB950" if "OPEN" in mkt_status else "#8B949E"
st.markdown(f"""
<div class="hdr">
    <div class="logo-box">
        <div class="logo-icon">Vc</div>
        <div>
            <div class="logo-text">VIX CONTROLLER</div>
            <div class="logo-tag">Volatility Intelligence Platform</div>
        </div>
    </div>
    <div class="sub">
        <span style="color:{mkt_clr};font-weight:600">{mkt_status}</span> · {now_str}<br>
        Source: CBOE Delayed Quotes · Actualiza con botón manual
    </div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    N_MONTHS = st.slider("Max futures months", 4, 12, 8)
    SHOW_PREV = st.checkbox("Show previous day", True)
    SHOW_TABLE = st.checkbox("Show data table", True)
    st.markdown("---")
    st.markdown("**🔄 Actualizar datos**")
    if st.button("📡 Refresh CBOE + yfinance", key="btn_refresh_cboe"):
        scrape_cboe_futures.clear()
        fetch_vix_spot.clear()
        fetch_etps.clear()
        fetch_today_prices.clear()
        st.rerun()
    if st.button("🗄️ Recargar Parquet (repo)", key="btn_reload_parquet"):
        load_master_parquet.clear()
        build_strategy_cached.clear()
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FETCH DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.spinner("🌐 Scraping CBOE delayed quotes…"):
    df_vx = scrape_cboe_futures()

# Mostrar diagnóstico en sidebar siempre
with st.sidebar:
    debug_msg = st.session_state.get("scrape_debug", "")
    if debug_msg:
        if debug_msg.startswith("❌"):
            st.error(debug_msg)
        else:
            st.info(f"🔍 {debug_msg}")
    html_sample = st.session_state.get("scrape_html_sample", "")
    if html_sample:
        st.warning("⚠️ No se encontró tabla VX — fragmento HTML:")
        st.code(html_sample[:600], language="html")

vix_spot = fetch_vix_spot()
etps = fetch_etps()

# Limit to N_MONTHS
if not df_vx.empty and len(df_vx) > N_MONTHS:
    df_vx = df_vx.head(N_MONTHS).reset_index(drop=True)

# Extract M1/M2 prices
m1p = df_vx['Price'].iloc[0] if not df_vx.empty and pd.notna(df_vx['Price'].iloc[0]) else None
m2p = df_vx['Price'].iloc[1] if len(df_vx) > 1 and pd.notna(df_vx['Price'].iloc[1]) else None
front_ct = cpct(m1p, m2p)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fv(v):
    return f"{v:.2f}" if v is not None and pd.notna(v) and v != 0 else "—"
def vc(v):
    if v is None: return "nt"
    return "up" if v >= 0 else "dn"
def fp(v):
    if v is None: return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}%"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab1, tab2, tab_baro, tab_edge, tab_skew, tab_gex, tab3, tab4 = st.tabs([
    "📈  Term Structure",
    "🎯  Monitor Operativo",
    "🌡️  Barómetro VTS",
    "🔬  Edge Analytics",
    "📐  Vol Skew & Surface",
    "⚡  GEX",
    "💡  Recomendaciones",
    "ℹ️  Help",
])

# ━━━━━━━━━━━━━━━━━ TAB 1: TERM STRUCTURE ━━━━━━━━━━━━━━━━━━
with tab1:
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

    # Chart
    fig = build_term_chart(vix_spot, df_vx, show_prev=SHOW_PREV)
    st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=True, displaylogo=False))

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
        st.warning("⚠️ No se pudieron obtener precios de futuros VIX del CBOE.")
        if not pw_ready:
            st.error("❌ Playwright/Chromium no se pudo inicializar. Verifica packages.txt y requirements.txt")
        st.info("💡 La página CBOE carga datos por JavaScript. Se necesita Playwright + Chromium para renderizarla.")

    if not df_vx.empty:
        scraped = df_vx['Scraped_At'].iloc[0] if 'Scraped_At' in df_vx.columns else "?"
        st.caption(f"Contratos: {len(df_vx)} mensuales · Scraped: {scraped} · CBOE Delayed Quotes")


# ━━━━━━━━━━━━━━━━━ TAB 2: MONITOR OPERATIVO ━━━━━━━━━━━━━━━
with tab2:

    # ── Cargar parquet (del repo, instantáneo) ────────────────
    df_master = load_master_parquet()

    if df_master.empty:
        st.error("❌ No se encontró data/master.parquet en el repositorio.")
        st.info("Ejecuta el notebook de actualización y haz push: df.to_parquet('data/master.parquet')")
        st.stop()

    # ── Aplicar estrategia (cacheado 1h) ──────────────────────
    bt = build_strategy_cached(df_master)

    # ── Precios de hoy (yfinance) ─────────────────────────────
    today_px   = fetch_today_prices()
    last_hist  = bt.iloc[-1]
    last_date  = bt.index[-1]

    vxx_today  = float(today_px.get('VXX',  {}).get('close', last_hist['VXX_Close']))
    svxy_today = float(today_px.get('SVXY', {}).get('close', 0))
    svix_today = float(today_px.get('SVIX', {}).get('close', 0))
    vix_val    = float(today_px.get('VIX',  {}).get('close', last_hist.get('VIX_Close', 0)))

    sma20  = float(last_hist['BB_SMA20'])
    bb_up  = float(last_hist['BB_Upper'])

    # BB signal de hoy (posición actual, sin shift)
    bb_pos = int(last_hist['sig_bb'])
    if bb_pos == 0 and vxx_today < sma20:   bb_sig_today = 1
    elif bb_pos == 1 and vxx_today > bb_up: bb_sig_today = 0
    else:                                    bb_sig_today = bb_pos

    # Contango live del CBOE (del Tab 1 — m1p, m2p en scope global)
    if m1p and m2p and m1p > 0:
        ct_today  = cpct(m1p, m2p)
        ct_source = "CBOE live"
        m1_sym    = df_vx['Symbol'].iloc[0] if not df_vx.empty else "M1"
        m2_sym    = df_vx['Symbol'].iloc[1] if len(df_vx) > 1 else "M2"
    else:
        ct_today  = float(last_hist.get('Contango_pct', 0)) if 'Contango_pct' in last_hist else None
        ct_source = "CSV histórico"
        m1_sym    = str(last_hist.get('M1_Symbol', 'M1'))
        m2_sym    = str(last_hist.get('M2_Symbol', 'M2'))

    in_ct_today     = ct_today is not None and ct_today > 0
    final_sig_today = int(bb_sig_today == 1 and in_ct_today)

    exec_date = now_cdmx().date() + timedelta(days=1)
    while exec_date.weekday() >= 5:
        exec_date += timedelta(days=1)

    pct_to_sma = (vxx_today / sma20 - 1) * 100 if sma20 else 0
    pct_to_bb  = (vxx_today / bb_up  - 1) * 100 if bb_up  else 0
    ct_str     = f"{ct_today:+.2f}%" if ct_today is not None else "N/A"

    if vix_val < 15:   regime, r_clr = "BAJO — óptimo",       "var(--g)"
    elif vix_val < 20: regime, r_clr = "NORMAL — bueno",      "var(--g)"
    elif vix_val < 28: regime, r_clr = "ELEVADO — precaución","var(--y)"
    else:              regime, r_clr = "CRISIS — peligro",    "var(--r)"

    def mcard(label, val, clr="nt"):
        return f'<div class="mpill"><div class="ml">{label}</div><div class="mv {clr}">{val}</div></div>'

    # ═══════════════════════════════════════════
    # SECCIÓN 1 — SEÑAL DE HOY
    # ═══════════════════════════════════════════
    sig_cls = "sig-long" if final_sig_today else "sig-cash"
    sig_txt = "LONG SVXY" if final_sig_today else "CASH"
    sig_clr = "var(--g)" if final_sig_today else "var(--r)"
    bb_ok   = "ok" if bb_sig_today else "no"
    ct_ok   = "ok" if in_ct_today  else "no"

    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.5, 1.3])

    with c1:
        st.markdown(f"""<div class="sig-box {sig_cls}">
            <div class="sl" style="color:{sig_clr}">{sig_txt}</div>
            <div class="sd">Ejecutar {exec_date.strftime('%Y-%m-%d')} al OPEN</div>
            <div class="sd">Señal cierre {last_date.strftime('%Y-%m-%d')}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        sma_clr = "var(--g)" if vxx_today < sma20 else "var(--r)"
        bb_clr  = "var(--g)" if vxx_today <= bb_up else "var(--r)"
        st.markdown(f"""<div class="icard">
            <div class="ic-title">📊 BB Timing — VXX</div>
            <div class="ic-row"><span class="ic-label">Señal BB</span>
                <span class="ic-val"><span class="{bb_ok}">{"✓" if bb_sig_today else "✗"}</span>
                {"&nbsp;LONG" if bb_sig_today else "&nbsp;CASH"}</span></div>
            <div class="ic-row"><span class="ic-label">VXX hoy</span>
                <span class="ic-val" style="font-weight:700">${vxx_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">SMA(20)</span>
                <span class="ic-val" style="color:{sma_clr}">${sma20:.2f} ({pct_to_sma:+.1f}%)</span></div>
            <div class="ic-row"><span class="ic-label">BB 2σ</span>
                <span class="ic-val" style="color:{bb_clr}">${bb_up:.2f} ({pct_to_bb:+.1f}%)</span></div>
        </div>""", unsafe_allow_html=True)

    with c3:
        ct_clr    = "var(--g)" if in_ct_today else "var(--r)"
        ct_estado = "CONTANGO" if in_ct_today else "BACKWARDATION"
        m1_disp   = f"${m1p:.2f}" if m1p else "—"
        m2_disp   = f"${m2p:.2f}" if m2p else "—"
        st.markdown(f"""<div class="icard">
            <div class="ic-title">📈 Contango ({ct_source})</div>
            <div class="ic-row"><span class="ic-label">Señal CT</span>
                <span class="ic-val"><span class="{ct_ok}">{"✓" if in_ct_today else "✗"}</span>
                <span style="color:{ct_clr};font-weight:700">&nbsp;{ct_estado}</span></span></div>
            <div class="ic-row"><span class="ic-label">{m1_sym} (M1)</span>
                <span class="ic-val">{m1_disp}</span></div>
            <div class="ic-row"><span class="ic-label">{m2_sym} (M2)</span>
                <span class="ic-val">{m2_disp}</span></div>
            <div class="ic-row"><span class="ic-label">Contango %</span>
                <span class="ic-val" style="color:{ct_clr};font-weight:700">{ct_str}</span></div>
            <div class="ic-row"><span class="ic-label">VIX</span>
                <span class="ic-val" style="color:{r_clr}">{vix_val:.1f} · {regime}</span></div>
        </div>""", unsafe_allow_html=True)

    with c4:
        svxy_chg = ""
        if today_px.get('SVXY', {}).get('prev'):
            d = svxy_today - today_px['SVXY']['prev']
            svxy_chg = f" ({d:+.2f})"
        st.markdown(f"""<div class="icard">
            <div class="ic-title">💼 Vehículos</div>
            <div class="ic-row"><span class="ic-label">SVXY (-0.5x)</span>
                <span class="ic-val" style="color:var(--c);font-weight:700">${svxy_today:.2f}{svxy_chg}</span></div>
            <div class="ic-row"><span class="ic-label">SVIX (-1x)</span>
                <span class="ic-val" style="color:var(--c)">${svix_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">VIX Spot</span>
                <span class="ic-val">{vix_val:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">CSV al</span>
                <span class="ic-val" style="color:var(--dim)">{last_date.strftime('%Y-%m-%d')}</span></div>
        </div>""", unsafe_allow_html=True)

    # Alertas
    if final_sig_today and pct_to_bb > -3:
        st.warning(f"⚠️ VXX a {abs(pct_to_bb):.1f}% de la BB Superior — posible salida pronto")
    if ct_today is not None and 0 < ct_today < 1:
        st.warning(f"⚠️ Contango muy bajo ({ct_today:.2f}%) — monitorear")
    if not final_sig_today and abs(pct_to_sma) < 2 and in_ct_today:
        st.info(f"🔔 Posible entrada pronto — VXX a {abs(pct_to_sma):.1f}% de SMA(20)")
    if not in_ct_today and bb_sig_today == 1:
        st.warning("⚠️ BB dice LONG pero hay backwardation — CASH por Contango Rule")

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECCIÓN 2 — GRÁFICA VXX OPERATIVA
    # ═══════════════════════════════════════════
    fig_mon = build_vxx_operational_chart(
        bt=bt,
        vxx_today=vxx_today,
        final_sig_today=final_sig_today,
        ct_today=ct_today,
    )
    st.plotly_chart(fig_mon, width="stretch",
                    config=dict(displayModeBar=True, displaylogo=False,
                                scrollZoom=False,
                                modeBarButtonsToRemove=['select2d','lasso2d']))

    st.caption(
        f"Histórico: {bt.index[0].strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} "
        f"({len(bt):,} días) · Parquet del repo · "
        f"Contango hoy: {ct_source} · "
        f"▲ verde = Entrada · ▼ amarillo = Salida por BB · ▼ rojo = Salida por Contango · "
        f"💎 = HOY"
    )

    # ═══════════════════════════════════════════════════════════
    # SECCIÓN 3 — BACKTEST WALK-FORWARD
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div style='border-top:2px solid #F7931A;margin:1rem 0 0.5rem'></div>",
                unsafe_allow_html=True)
    st.markdown("## 📊 Backtest Walk-Forward — BB×Contango")

    col_wf1, col_wf2 = st.columns([2, 1])
    with col_wf1:
        wf_window = st.select_slider(
            "Ventana walk-forward",
            options=[3, 6, 9, 12],
            value=6,
            format_func=lambda x: f"{x} meses",
            help="Longitud de cada ventana de evaluación fuera de muestra")
    with col_wf2:
        if st.button("🔄 Recalcular backtest", key="btn_wf"):
            run_walkforward_backtest.clear()

    with st.spinner("⚙️ Calculando walk-forward…"):
        wf_result = run_walkforward_backtest(bt, wf_months=wf_window)

    if not wf_result:
        st.warning("⚠️ Datos insuficientes para el backtest. Verifica que el parquet tenga VXX_Close y sig_final.")
    else:
        full   = wf_result["full"]
        wf_df  = wf_result["wf_df"]
        trades = wf_result["trades"]

        # ── Alerta de deterioro de edge ──────────────────────
        if not wf_df.empty and "sharpe" in wf_df.columns:
            last2 = wf_df["sharpe"].tail(2).values
            if len(last2) == 2 and all(v is not None and v < 0.5 for v in last2):
                st.error("🚨 **ALERTA EDGE:** Sharpe < 0.5 en las últimas 2 ventanas walk-forward — el edge puede estar deteriorándose")
            elif not wf_df.empty and wf_df["sharpe"].iloc[-1] is not None and wf_df["sharpe"].iloc[-1] < 0.5:
                st.warning("⚠️ **Sharpe < 0.5 en la última ventana** — monitorear el edge de cerca")
            else:
                last_sharpe = wf_df["sharpe"].iloc[-1]
                st.success(f"✅ Edge activo — Sharpe última ventana: {last_sharpe:.2f}")

        # ── Métricas full-sample ──────────────────────────────
        def _fmt_m(v, sfx="", prec=2):
            return f"{v:.{prec}f}{sfx}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"

        st.markdown(f"""
        <div class="mrow">
            <div class="mpill">
                <div class="ml">CAGR (full)</div>
                <div class="mv {'up' if (full.get('cagr') or 0) > 0 else 'dn'}">{_fmt_m(full.get('cagr'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Sharpe (full)</div>
                <div class="mv {'up' if (full.get('sharpe') or 0) > 1 else 'nt'}">{_fmt_m(full.get('sharpe'))}</div>
            </div>
            <div class="mpill">
                <div class="ml">Sortino</div>
                <div class="mv nt">{_fmt_m(full.get('sortino'))}</div>
            </div>
            <div class="mpill">
                <div class="ml">Calmar</div>
                <div class="mv nt">{_fmt_m(full.get('calmar'))}</div>
            </div>
            <div class="mpill">
                <div class="ml">Max Drawdown</div>
                <div class="mv dn">{_fmt_m(full.get('max_dd'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Vol anualizada</div>
                <div class="mv nt">{_fmt_m(full.get('volatility'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Alpha mensual</div>
                <div class="mv {'up' if (full.get('alpha_monthly') or 0) > 0 else 'dn'}">{_fmt_m(full.get('alpha_monthly'), '%', 3)}</div>
            </div>
            <div class="mpill">
                <div class="ml">Beta vs SPY</div>
                <div class="mv nt">{_fmt_m(full.get('beta'))}</div>
            </div>
        </div>
        <div class="mrow">
            <div class="mpill">
                <div class="ml">Nº Trades</div>
                <div class="mv nt">{full.get('n_trades', '—')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Win Rate</div>
                <div class="mv nt">{_fmt_m(full.get('win_rate'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Avg Hold</div>
                <div class="mv nt">{_fmt_m(full.get('avg_hold_d'), 'd', 0)}</div>
            </div>
            <div class="mpill">
                <div class="ml">Avg Win</div>
                <div class="mv up">{_fmt_m(full.get('avg_win'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Avg Loss</div>
                <div class="mv dn">{_fmt_m(full.get('avg_loss'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Profit Factor</div>
                <div class="mv nt">{_fmt_m(full.get('profit_factor'))}</div>
            </div>
            <div class="mpill">
                <div class="ml">CAGR B&H SVXY</div>
                <div class="mv nt">{_fmt_m(full.get('bh_cagr'), '%')}</div>
            </div>
            <div class="mpill">
                <div class="ml">Sharpe B&H SVXY</div>
                <div class="mv nt">{_fmt_m(full.get('bh_sharpe'))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='border-top:1px solid #30363D;margin:0.5rem 0'></div>",
                    unsafe_allow_html=True)

        # ── Equity curve + Sharpe rolling ────────────────────
        try:
            fig_eq = build_equity_curve_chart(wf_result)
            if fig_eq.data:
                st.plotly_chart(fig_eq, width="stretch", config=dict(displayModeBar=True))
        except Exception as e:
            st.error(f"Error equity curve: {e}")

        # ── Drawdown + Walk-forward ───────────────────────────
        col_dd, col_wfc = st.columns([1, 1.3])
        with col_dd:
            try:
                fig_dd = build_drawdown_chart(wf_result)
                if fig_dd.data:
                    st.plotly_chart(fig_dd, width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error drawdown: {e}")
        with col_wfc:
            try:
                fig_wf = build_walkforward_chart(wf_df, wf_months=wf_window)
                if fig_wf.data:
                    st.plotly_chart(fig_wf, width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error walk-forward: {e}")

        # ── Monthly returns heatmap ───────────────────────────
        try:
            fig_mh = build_monthly_returns_heatmap(wf_result.get("monthly", pd.DataFrame()))
            if fig_mh.data:
                st.plotly_chart(fig_mh, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error heatmap mensual: {e}")

        # ── Trades individuales ───────────────────────────────
        if not trades.empty:
            with st.expander(f"📋 Trades individuales ({len(trades)} operaciones)", expanded=False):
                try:
                    fig_tr = build_trades_chart(trades)
                    if fig_tr.data:
                        st.plotly_chart(fig_tr, width="stretch", config=dict(displayModeBar=False))
                except Exception as e:
                    st.error(f"Error trades chart: {e}")

                # Tabla de trades — sin estilos para máxima compatibilidad
                trades_display = trades.copy()
                trades_display["ret_pct"] = trades_display["ret_pct"].apply(lambda x: f"{x:+.2f}%")
                if "hold_d" in trades_display.columns:
                    trades_display["hold_d"] = trades_display["hold_d"].apply(lambda x: f"{int(x)}d")
                st.dataframe(trades_display, use_container_width=True, hide_index=True)

                # Resumen de salidas
                if "exit_why" in trades.columns:
                    bb_exits = (trades["exit_why"] == "BB").sum()
                    ct_exits = (trades["exit_why"] == "CT").sum()
                    st.markdown(
                        f"Salidas por **BB**: {bb_exits} ({bb_exits/len(trades)*100:.0f}%) · "
                        f"Salidas por **Contango Rule**: {ct_exits} ({ct_exits/len(trades)*100:.0f}%)"
                    )

        st.caption(
            f"Walk-Forward: ventanas de {wf_window}M · "
            f"Retorno vehículo: {'SVXY_Close' if 'SVXY_Close' in bt.columns else '-0.5×VXX aprox'} · "
            f"RF: {0.043*100:.1f}% anual · Alpha vs SPY (regresión mensual)"
        )


# ━━━━━━━━━━━━━━━━━ TAB EDGE: EDGE ANALYTICS ━━━━━━━━━━━━━━━
with tab_edge:

    if 'df_master' not in dir() or df_master.empty:
        df_master_edge = load_master_parquet()
    else:
        df_master_edge = df_master

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
            ry_val   = last_e.get('Roll_Yield', np.nan)
            skew_val = last_e.get('SKEW', np.nan)
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

            # ── Expander: HAR model diagnostics ─────────────────────
            har_beta = edge.get('har_beta', {})
            if har_beta or use_har:
                with st.expander("🧮 Modelo HAR-RV — Detalles y coeficientes", expanded=False):
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
                st.markdown('<div style="font-family:JetBrains Mono;font-size:0.75rem;color:#3FB950;'
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
                verdict, v_clr, v_bg = "EDGE ACTIVO — CON PRECAUCION", "var(--y)", "#3D2E00"
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

            st.markdown("<div style='border-top:1px solid #30363D;margin:0.6rem 0'></div>",
                        unsafe_allow_html=True)

            # ── Última fecha disponible ──────────────────────────────
            last_date_ebt = ebt.index[-1].date()
            st.markdown(
                f'<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#8B949E;margin-bottom:0.4rem">'
                f'📅 Datos hasta: <b style="color:#C9D1D9">{last_date_ebt}</b>'
                f'{"  ✅ Al día" if last_date_ebt >= (now_cdmx().date() - timedelta(days=3)) else "  ⚠ parquet desactualizado"}'
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
                with st.expander("📈 Backtest HAR-A — ¿Qué tan bien predice la volatilidad futura?", expanded=False):
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
                    st.dataframe(pd.DataFrame(brow1), use_container_width=True, hide_index=True)
                    st.markdown(f"""
<div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#8B949E;margin:0.3rem 0">
<b style="color:#C9D1D9">Mincer-Zarnowitz:</b> α={mz_a} (ideal 0) · β={mz_b} (ideal 1) ·
<b style="color:#C9D1D9">Muestra test:</b> {n_t} días (~2 años)
</div>
<div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8B949E;margin-top:0.3rem">
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


# ━━━━━━━━━━━━━━━━━ TAB: VOL SKEW & SURFACE ━━━━━━━━━━━━━━━━━
with tab_skew:

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
                                   value=0.043, step=0.001, format="%.3f",
                                   help="Tasa libre de riesgo anualizada (ej: 4.3%=0.043)")
    with col_c4:
        skew_div = st.number_input("Dividend Yield (q)", 0.0, 0.10,
                                   value=0.013, step=0.001, format="%.3f",
                                   help="Yield de dividendo continuo (SPY≈1.3%)")

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
        view_mode = st.selectbox("Vista superficie", ["🌐 3D Surface", "🗺️ Heatmap 2D"])

    if st.button("🔄 Actualizar opciones", key="refresh_options"):
        fetch_options_chains.clear()
        st.rerun()

    # ── Fetch raw data ────────────────────────────────────────
    est_secs = n_exps * 1.2 + 2
    with st.spinner(f"📡 Descargando {n_exps} vencimientos de {skew_ticker} (~{est_secs:.0f}s)…"):
        opt_chains_raw, opt_spot = fetch_options_chains(skew_ticker, n_exp=n_exps)

    if not opt_chains_raw or not opt_spot:
        st.error(f"❌ Yahoo Finance rate limit — no se cargaron opciones para **{skew_ticker}**.")
        st.info("💡 Espera 3-5 min · baja a 2 vencimientos · o intenta en horario de mercado (9:30-16:00 ET).")
        st.stop()

    # ── Calcular IV con Black-Scholes (Brent's method) ───────
    with st.spinner("⚙️ Calculando IV Black-Scholes…"):
        opt_chains = compute_bs_iv_for_chains(opt_chains_raw, opt_spot,
                                              r=skew_rfr, q=skew_div)

    if not opt_chains:
        st.warning("⚠️ No se pudo calcular IV BS para ningún vencimiento. "
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
        if rr_raw < -4: interp_parts.append("🔴 **Risk Reversal muy negativo** — put skew extremo, fear elevado")
        elif rr_raw < -2: interp_parts.append("🟡 **Risk Reversal negativo moderado** — demanda de cobertura activa")
        else: interp_parts.append("🟢 **Risk Reversal neutro** — apetito por riesgo presente")
    if pc_raw is not None:
        if pc_raw > 1.5: interp_parts.append("🔴 **P/C Ratio > 1.5** — flujo dominante en puts, hedging institucional")
        elif pc_raw > 1.0: interp_parts.append("🟡 **P/C Ratio > 1.0** — ligero sesgo defensivo")
        else: interp_parts.append("🟢 **P/C Ratio < 1.0** — flujo en calls, risk-on")
    if interp_parts:
        with st.expander("📊 Lectura del Skew", expanded=True):
            for l in interp_parts: st.markdown(l)

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.5rem 0'></div>",
                unsafe_allow_html=True)

    # ── Skew Curves + ATM Term Structure ─────────────────────
    col_sk, col_atm = st.columns([1.6, 1])
    with col_sk:
        try:
            fig_sk = build_skew_curves(opt_chains, opt_spot,
                                       moneyness_range=(mon_lo, mon_hi),
                                       y_mode=y_mode)
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

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.5rem 0'></div>",
                unsafe_allow_html=True)

    # ── IV Surface ────────────────────────────────────────────
    if view_mode == "🌐 3D Surface":
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
    with st.expander("📋 Tabla resumen por vencimiento"):
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
            st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

    st.caption(
        f"IV calculada con Black-Scholes (Brent) · r={skew_rfr:.1%} · q={skew_div:.1%} · "
        f"Spot {skew_ticker}: {spot_disp} · {now_cdmx().strftime('%H:%M:%S')} CDMX"
    )

    # ════════════════════════════════════════════════════════
    # SECCIÓN: VOL SURFACE FORECAST + SELLING RECOMMENDATIONS
    # ════════════════════════════════════════════════════════
    st.markdown("<div style='border-top:2px solid #F7931A;margin:0.8rem 0 0.4rem'></div>",
                unsafe_allow_html=True)
    st.markdown("## 🔮 Forecast de Superficie + Oportunidades de Venta de Vol")
    st.markdown("""
    <div style="font-family:'JetBrains Mono';font-size:0.72rem;color:#8B949E;margin-bottom:0.6rem">
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

    with st.spinner("🔮 Fittando SVI y calculando forecast…"):
        fc_result = forecast_vol_surface(
            opt_chains, opt_spot,
            r=skew_rfr, q=skew_div,
            iv_change_pct=iv_scenario/100,
        )

    if not fc_result:
        st.warning("⚠️ No se pudo fittar el modelo SVI. Se necesitan ≥5 strikes por vencimiento.")
    else:
        svi_fits = fc_result.get("svi_fits", {})
        sell_df  = fc_result.get("sell_df", pd.DataFrame())

        # ── SVI model params ─────────────────────────────────
        with st.expander("📐 Parámetros SVI por vencimiento", expanded=False):
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
                st.dataframe(pd.DataFrame(rows_svi), use_container_width=True, hide_index=True)

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
                    "🔴 Put skew muy pronunciado — alta demanda de protección" if rho_s < -0.4
                    else "🟡 Put skew moderado — skew normal de equity" if rho_s < -0.2
                    else "🟢 Smile casi simétrico — mercado tranquilo"
                )
                st.markdown(f"""
                <div class="icard" style="margin-top:1.5rem">
                    <div class="ic-title">📐 SVI {fc_exp_sel}</div>
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
        st.markdown("### 🎯 Opciones Candidatas para Vender Prima")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono';font-size:0.72rem;color:#8B949E;margin-bottom:0.5rem">
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
                        styles.loc[i,"Tipo"] = "color: #39D2C0"
                    else:
                        styles.loc[i,"Tipo"] = "color: #BC8CFF"
                    if df.loc[i,"P&L Esp. ($)"] > 50:
                        styles.loc[i,"P&L Esp. ($)"] = "color: #3FB950; font-weight:bold"
                    elif df.loc[i,"P&L Esp. ($)"] > 20:
                        styles.loc[i,"P&L Esp. ($)"] = "color: #D29922"
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
                use_container_width=True, hide_index=True
            )

            # ── Best trade summary ────────────────────────────
            best = sell_df.iloc[0]
            st.markdown(f"""
            <div style="background:var(--gbg);border:1px solid var(--g);border-radius:6px;
                        padding:0.7rem 1rem;margin-top:0.5rem">
                <div style="font-family:Inter;font-weight:800;font-size:1rem;color:var(--g)">
                    🎯 MEJOR CANDIDATO</div>
                <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#C9D1D9;margin-top:0.3rem">
                    <b>Vender {best['Tipo']} K=${best['Strike']:.0f} exp {best['Exp']} ({best['DTE']}d)</b>
                    · Dist spot: {best['Dist Spot %']:+.1f}%<br>
                    IV actual: {best['IV Actual %']:.1f}% → IV forecast: {best['IV Forecast %']:.1f}%
                    → Drop esperado: <b>{best['IV Drop pts']:.2f} pts</b><br>
                    Mid: ${best['Mid $']:.2f} · OI: {best['OI']:,} · Vega/ct: ${best['Vega/ct ($)']:.0f}
                    · <b>P&L esperado: ${best['P&L Esp. ($)']:.0f}/contrato</b>
                </div>
                <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8B949E;margin-top:0.3rem">
                ⚠️ Solo análisis educativo. No es recomendación financiera.
                El P&L esperado asume que la IV cae exactamente {iv_scenario:+.0f}% — nada garantizado.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No se encontraron candidatos con P&L positivo bajo el escenario seleccionado.")

    st.caption(
        f"SVI: Gatheral (2004) · No-arbitrage butterfly condition: b(1+|ρ|)≤2 · "
        f"Vega = S·√T·N'(d₁)·100 · Opciones con OI>0 en ±22% del spot"
    )


# ━━━━━━━━━━━━━━━━━ TAB: GEX — GAMMA EXPOSURE ━━━━━━━━━━━━━━━
with tab_gex:

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8B949E;
                padding:0.3rem 0 0.7rem;">
    <b>GEX (Gamma Exposure)</b> mide el gamma neto de los dealers del mercado por strike.
    Cuando GEX es <span style="color:#3FB950">positivo</span>: dealers compran dips y venden rallies → mercado anclado al strike.
    Cuando GEX es <span style="color:#F85149">negativo</span>: dealers venden dips y compran rallies → movimientos se amplifican.
    </div>
    """, unsafe_allow_html=True)

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
        gex_rfr = st.number_input("Risk-Free Rate (r)", 0.0, 0.15, 0.043,
                                   step=0.001, format="%.3f", key="gex_rfr")
    with col_g4:
        gex_div = st.number_input("Dividend Yield (q)", 0.0, 0.10, 0.013,
                                   step=0.001, format="%.3f", key="gex_div")

    col_g5, col_g6 = st.columns([2, 1])
    with col_g5:
        gex_range = st.slider("Rango de strikes mostrado (±% del spot)", 5, 20, 12,
                              help="Porcentaje del spot hacia arriba y abajo del strike central")
    with col_g6:
        if st.button("🔄 Actualizar GEX", key="btn_refresh_gex"):
            fetch_options_chains.clear()
            st.rerun()

    # ── Datos: reusar cache de opciones si el ticker/n_exp coincide ──────
    with st.spinner(f"📡 Cargando opciones {gex_ticker} ({gex_n_exp} vencimientos)…"):
        gex_chains_raw, gex_spot = fetch_options_chains(gex_ticker, n_exp=gex_n_exp)

    if not gex_chains_raw or not gex_spot:
        st.error(f"❌ No se pudieron cargar opciones para **{gex_ticker}**.")
        st.info("Espera 3-5 min · baja a 1-2 vencimientos · o intenta en horario de mercado.")
        st.stop()

    # Calcular IV BS (necesaria para gamma precisa)
    with st.spinner("⚙️ Calculando Gamma (Black-Scholes)…"):
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
        st.warning("⚠️ No hay suficientes datos de opciones para calcular el GEX.")
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
            <div class="mv" style="color:#BC8CFF">${f"{call_w:.0f}" if call_w else "—"} <span style="font-size:0.75rem;color:var(--dim)">({cw_dist})</span></div>
        </div>
        <div class="mpill">
            <div class="ml">Put Wall</div>
            <div class="mv" style="color:#39D2C0">${f"{put_w:.0f}" if put_w else "—"} <span style="font-size:0.75rem;color:var(--dim)">({pw_dist})</span></div>
        </div>
        <div class="mpill">
            <div class="ml">%OTM con GEX+</div>
            <div class="mv nt">{f"{pct_pos:.0f}%" if pct_pos is not None else "—"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interpretación ────────────────────────────────────────
    with st.expander("📊 Interpretación GEX", expanded=True):
        if regime == "POSITIVE":
            st.markdown(
                f"✅ **Régimen POSITIVO** — Los dealers tienen gamma larga neta. "
                f"Cuando el mercado cae, los dealers **compran** para delta-hedgear → actúa como soporte. "
                f"Cuando el mercado sube, los dealers **venden** → actúa como resistencia. "
                f"Resultado: **volatilidad comprimida**, el mercado tiende a mantenerse anclado cerca del gamma flip (${flip_s:.0f} aprox.)."
                if flip_s else
                f"✅ **Régimen POSITIVO** — Los dealers tienen gamma larga neta → mercado estabilizador."
            )
        else:
            st.markdown(
                f"⚠️ **Régimen NEGATIVO** — Los dealers tienen gamma corta neta. "
                f"Cuando el mercado cae, los dealers **también venden** para hedgear → los movimientos se **amplifican**. "
                f"El mercado puede tener gaps y movimientos bruscos. "
                f"{'El Gamma Flip está en $' + str(int(flip_s)) + ' — recuperar ese nivel sería la señal de estabilización.' if flip_s else ''}"
            )
        st.markdown("""
**Niveles clave:**
- 🟣 **Call Wall**: mayor concentración de gamma de calls — los dealers venden agresivamente aquí (techo)
- 🩵 **Put Wall**: mayor concentración de gamma de puts — los dealers compran agresivamente aquí (soporte)
- 🟡 **Gamma Flip**: punto donde el régimen cambia de positivo a negativo
        """)

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.5rem 0'></div>",
                unsafe_allow_html=True)

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
    with st.expander("📋 Top strikes por |GEX|"):
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
                use_container_width=True, hide_index=True)

    st.caption(
        f"GEX/DEX = OI × Greeks_BS × S² × 100 · "
        f"Vanna = ∂Δ/∂σ · Charm = ∂Δ/∂t · "
        f"Max Pain = mínima pérdida agregada de holders · "
        f"r={gex_rfr:.1%} q={gex_div:.1%} · {now_cdmx().strftime('%H:%M:%S')} CDMX"
    )


# ━━━━━━━━━━━━━━━━━ TAB: BARÓMETRO VTS ━━━━━━━━━━━━━━━━━━━━━━
with tab_baro:

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8B949E;
                padding:0.4rem 0 0.8rem;">
    <b>VTS Volatility Barometer</b> · Inspirado en <b>volatilitytradingstrategies.com</b>
    · Combina <b>13+ métricas de volatilidad</b> en un score único 0-100% ·
    Cada métrica convertida a percentil rolling (252 días) ·
    Score = promedio ponderado ·
    <span style="color:#3FB950">Verde</span> = posición agresiva short vol ·
    <span style="color:#D29922">Amarillo</span> = cash / neutral ·
    <span style="color:#F85149">Rojo</span> = hedge / long vol
    </div>
    """, unsafe_allow_html=True)

    # Slider para ventana
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        baro_window = st.select_slider(
            "Ventana rolling para percentiles",
            options=[126, 252, 504, 756],
            value=252,
            format_func=lambda x: f"{x} días (~{x//21}m)",
            help="VTS usa 252d = 1 año. Ventanas más cortas reaccionan más rápido.",
        )
    with col_w2:
        st.write("")
        st.write("")
        if st.button("🔄 Recalcular", key="btn_baro_recalc"):
            compute_vts_barometer.clear()
            fetch_edge_extra.clear()

    # ── Cargar datos necesarios ──────────────────────────
    with st.spinner("🌡️ Calculando barómetro..."):
        df_master_baro = load_master_parquet()
        if df_master_baro.empty:
            st.error("❌ No se encontró data/master.parquet")
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

        # Fetch de datos extra (VVIX, SKEW, HYG, IEF)
        edge_extra_baro = fetch_edge_extra()

        # Calcular el barómetro
        baro = compute_vts_barometer(
            bt=bt_baro,
            edge_extra=edge_extra_baro,
            gex_summary=None,       # opcional — se agrega si está disponible
            skew_metrics=None,      # opcional
            window=baro_window,
        )

    if not baro:
        st.warning("⚠️ No se pudo calcular el barómetro — verifica datos")
        st.stop()

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
        elif score < 80: rc = '#FB8500'
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
            <div class="sig-box" style="background:rgba(247,147,26,0.08);
                 border-color:#F7931A;margin-bottom:0.8rem;">
                <div class="sl" style="color:{rc};">{score:.2f}%</div>
                <div class="sd" style="font-size:0.85rem;color:{rc};font-weight:700;">
                    Régimen: {regime}
                </div>
                <div class="sd" style="margin-top:4px;">{position}</div>
            </div>
            <div class="icard">
                <div class="ic-title">📊 Métricas del barómetro</div>
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

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)

    # ── Histórico del score ──────────────────────────────
    if baro['history'] is not None and not baro['history'].empty:
        hist_fig = build_vts_history_chart(baro['history'], window=baro_window)
        st.plotly_chart(hist_fig, width="stretch", config=dict(displayModeBar=False))

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)

    # ── Desglose de métricas ─────────────────────────────
    metrics_fig = build_vts_metrics_table(baro['metrics'])
    st.plotly_chart(metrics_fig, width="stretch", config=dict(displayModeBar=False))

    # ── Interpretación y guía de lectura ─────────────────
    with st.expander("📖 Metodología y guía de lectura", expanded=False):
        st.markdown(f"""
**Qué mide el barómetro**

El VTS Volatility Barometer combina {n_metrics} métricas de volatilidad en un único score
0-100%. Cada métrica individual solo captura una porción del mercado
(futuros VIX, opciones SPX, credit, etc.), pero combinadas ofrecen una lectura robusta
del régimen de volatilidad actual.

**Métricas incluidas en esta implementación:**

1. **VIX Spot Level** — percentil del nivel absoluto del VIX
2. **VIX / VIX-Fut M1** — inversión de la parte corta de la curva
3. **VVIX** — volatilidad del VIX (demanda de opciones sobre VIX)
4. **Contango M1-M2 (invertido)** — roll yield disponible
5. **VXX / SMA(20)** — momentum direccional del VXX
6. **SPY RV 22d** — volatilidad realizada del subyacente
7. **VRP (VIX-RV) invertido** — risk premium compression
8. **CBOE SKEW Index** — demanda de puts OTM SPX
9. **HYG/IEF (invertido)** — proxy de credit spread
10. **SPY Drawdown 20d** — profundidad del drawdown reciente
11. **VIX 5d change %** — velocidad del cambio de vol
12. **Contango M4-M7 (invertido)** — curva larga, stress estructural
13. **SPY vs SMA(200) (invertido)** — régimen macro bull/bear

**Interpretación del score:**

- **0-20% (Vol BAJA)**: Todos los indicadores apuntan a vol estable.
  Posición agresiva short vol — SVXY/SVIX al 100%.
- **20-40% (MODERADA)**: La mayoría de señales verdes pero algunos indicadores
  elevados. SVXY al 75-100% — el trade funciona pero con alerta.
- **40-60% (MID)**: Señales mixtas. Cash o posición parcial (25-50%).
  Es el rango donde más falsos positivos ocurren.
- **60-80% (ELEVADA)**: Mayoría de señales rojas. Cash. Evitar short vol.
- **80-100% (EXTREMA)**: Entorno de crisis (COVID, Aug 2015, Feb 2018).
  Oportunidad de **long VIX / short equities / long puts**.

**Diferencias vs VTS original:**

VTS usa su propia mezcla propietaria de 13 métricas con pesos afinados durante más
de una década. Esta implementación usa métricas similares pero los pesos pueden
diferir. La utilidad principal es como **filtro de régimen**: confirma cuándo el
entorno es favorable para la estrategia BB × Contango del Monitor Operativo.

**Uso recomendado:**

- Score < 40% + señal LONG del Monitor Operativo = **convicción alta**
- Score 40-60% + señal LONG = **reducir tamaño o esperar confirmación**
- Score > 60% = **NO tomar señal LONG aunque el Monitor la marque**
- Score > 80% = **considerar posiciones long vol (VXX/UVXY)** como hedge
""")

    st.caption(
        f"VTS Volatility Barometer v1.0 · "
        f"Inspirado en volatilitytradingstrategies.com · "
        f"Ventana: {baro_window}d · Métricas: {n_metrics} · "
        f"Última actualización: {baro['date'].strftime('%Y-%m-%d')}"
    )


# ━━━━━━━━━━━━━━━━━ TAB 3: RECOMENDACIONES ━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("""
    ### 💡 Recomendaciones para Mejorar el Análisis

    ---

    **🔧 Mejoras al Monitor Operativo:**

    **1. Alertas por Telegram/Email**
    Configurar un bot que envíe notificación cuando la señal cambie de LONG a CASH o viceversa. Solo 7 alertas al año pero cada una es crítica.

    **2. Dashboard de Régimen de Mercado**
    Panel dedicado que muestre: VIX actual con percentil histórico, ratio VIX/VIX3M (inversión de term structure), VVIX (volatilidad del VIX), y correlación SPX-VIX rolling. Esto da contexto de "qué tan peligroso es el entorno actual".

    **3. Indicador de Calidad de Señal**
    No todas las entradas son iguales. Agregar un "score" que pondere: nivel de contango (más alto = mejor), distancia de VXX a SMA (más lejos debajo = más confianza), VIX absoluto (< 15 = óptimo), y VVIX (< 100 = calma).

    **4. Position Sizing Dinámico**
    En vez de todo-o-nada, escalar la posición según el score de calidad: 100% en VIX < 15 con contango > 5%, 75% en VIX 15-20, 50% en VIX 20-25, 25% o nada en VIX > 25.

    ---

    **📊 Mejoras Analíticas:**

    **5. GEX (Gamma Exposure) Overlay**
    Agregar datos de gamma exposure del SPX para identificar niveles de soporte/resistencia donde los dealers hacen hedging. Esto ayuda a anticipar movimientos explosivos del VIX.

    **6. Skew Monitor**
    Mostrar el skew de opciones del SPX (ratio de puts OTM vs calls OTM). Un skew elevado anticipa demanda de protección y potencial spike de VIX.

    **7. Análisis de Flujos (ETP Flows)**
    Trackear el AUM y flujos netos de VXX, SVXY, UVXY. Flujos masivos hacia VXX = demanda de protección. Flujos hacia SVXY = apetito por riesgo.

    **8. Correlación Rolling SPX-VIX**
    Mostrar la correlación rolling 20d entre SPX y VIX. Cuando se rompe la correlación inversa normal (ambos suben o ambos bajan), es señal de stress estructural.

    ---

    **🔄 Mejoras Operativas:**

    **9. Trade Journal Automático**
    Que el monitor genere automáticamente un registro cada vez que detecta cambio de señal: fecha, precios, condiciones de mercado, y lo append a un Google Sheet via API.

    **10. Backtesting Rolling (Walk-Forward Live)**
    Cada mes, recalcular automáticamente el Sharpe rolling 6m y comparar con el del backtest original. Si cae debajo de 0.5 por 2 meses, flag de alerta.

    **11. Multi-Timeframe Confirmation**
    Agregar un BB(20, 2σ) en timeframe semanal además del diario. Operar solo cuando ambos timeframes coinciden podría reducir whipsaws.

    **12. Slippage Tracker**
    Comparar el precio de ejecución real (que registras en el Sheet) vs el open teórico. Acumular el slippage real por trade para saber cuánto te cuesta la ejecución.

    ---

    **📈 Instrumentos Adicionales:**

    **13. Bull Put Spread como Alternativa**
    En vez de comprar SVXY directamente, vender Bull Put Spreads en SPY cuando la señal está activa. Misma dirección pero con riesgo definido y theta positiva.

    **14. Comparar con SVIX (-1x)**
    Ya tienes SVIX en el monitor. Agregar un panel que compare el retorno acumulado de la misma señal aplicada a SVXY vs SVIX en los últimos 6 meses.

    **15. VIX Futures Roll Yield Monitor**
    Mostrar el roll yield diario implícito: (M1-Spot)/M1 * (365/DTE). Este es el "carry" real que captura la estrategia y es el indicador más directo del edge.

    """)

# ━━━━━━━━━━━━━━━━━ TAB 4: HELP ━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("""
    ### VIX Controller — Guía

    **Tab 1 · Term Structure** — Réplica de VIXCentral.com
    - Datos scrapeados de CBOE Delayed Quotes vía **Playwright + Chromium**
    - Solo contratos mensuales (regex `^VX/[A-Z]\\d+$`)
    - Columnas: **Last, Change, High, Low, Settlement, Volume**
    - Tabla de contango/diferencia entre meses (estilo VIXCentral)
    - Refresh manual desde el botón del sidebar

    **Tab 2 · Monitor Operativo** — Señal BB × Contango (v2)
    - **BB Timing**: VXX < SMA(20) → LONG, VXX > BB Superior → EXIT
    - **Contango Rule**: contango > 0 es requisito para estar LONG
    - **Señal Final** = sig_BB × sig_Contango (AND lógico)
    - Gráfica rediseñada con 3 paneles:
      1. VXX + BB + flechas Entry/Exit diferenciadas
      2. Contango histórico con barras verdes/rojas
      3. **Equity Curve** de la estrategia vs Buy&Hold SVXY
    - Backtest walk-forward con Sharpe rolling

    **Tab 3 · Barómetro VTS** — *NUEVO*
    - Inspirado en `volatilitytradingstrategies.com`
    - **13+ métricas de volatilidad** combinadas en score 0-100%
    - Percentiles rolling (window configurable: 126-756 días)
    - Gauge visual con 5 regímenes coloreados
    - Desglose por métrica + timeline histórico
    - Filtro de régimen para validar señales del Monitor Operativo

    **Tab 4 · Edge Analytics** — Diagnósticos estadísticos
    - VRP (IV-RV), HAR-RV forecast, roll yield, VVIX ratio, skew, credit

    **Tab 5 · Vol Skew & Surface** — IV via Black-Scholes + Brent
    - Smile por vencimiento, term structure ATM, superficie 3D, heatmap
    - SVI fit para smoothing y forecast

    **Tab 6 · GEX** — Gamma Exposure del SPX
    - Perfil GEX por strike, expected move, DEX, cumulative GEX
    - Vanna/Charm, GEX por vencimiento
    - Zero Gamma Level y flip point

    ---

    **Fuentes de datos:**
    - `cboe.com/delayed_quotes/futures/future_quotes` — scrapeado con Playwright
    - Yahoo Finance — VIX, VVIX, SKEW, VXX, SVXY, SVIX, SPY, HYG, IEF, chains de opciones
    - Parquet local `data/master.parquet` — histórico de VXX + futuros VIX

    **Para Streamlit Cloud necesitas:**
    - `packages.txt` con dependencias de Chromium (libnss3, libatk, etc.)
    - `requirements.txt` con playwright + yfinance + plotly + scipy

    **Bugs corregidos en esta versión:**
    - Pestaña COT eliminada (funciones `build_cot_*_chart` nunca se definieron)
    - Decorador `@st.cache_data` huérfano que causaba TTL incorrecto
    - Gráfica Monitor Operativo: flechas ahora son markers scatter (no anotaciones de texto que se perdían)
    """)

st.markdown(f"""
<div style="text-align:center;padding:0.8rem 0 0.3rem;border-top:1px solid #30363D;margin-top:1rem;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#484F58;">
        VIX CONTROLLER · Alberto Alarcón González · Not financial advice
    </span>
</div>""", unsafe_allow_html=True)
