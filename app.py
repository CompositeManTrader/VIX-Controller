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
    out = {}
    for name, sym in [("SKEW", "^SKEW"), ("HYG", "HYG"), ("IEF", "IEF")]:
        try:
            h = yf.download(sym, period="2y", progress=False)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            if not h.empty:
                out[name] = h
        except:
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

def _yahoo_options_session():
    try:
        from curl_cffi import requests as cffi_req
        return cffi_req.Session(impersonate="chrome120")
    except ImportError:
        return None

@st.cache_data(ttl=900)
def fetch_options_chains(ticker: str = "SPY", n_exp: int = 4) -> tuple:
    """
    Descarga opciones y devuelve precios RAW (bid, ask, lastPrice).
    IV se calcula después con BS dado que yfinance's IV es poco fiable.
    Estrategia: curl_cffi → Yahoo v8 API directo → fallback yfinance.
    """
    log = logging.getLogger("vix_controller")

    def _clean(df_raw, spot_px):
        df_c = df_raw.copy()
        for col in ["bid","ask","lastPrice","openInterest","volume","strike"]:
            df_c[col] = pd.to_numeric(df_c.get(col, 0), errors="coerce").fillna(0)
        df_c = df_c[df_c["openInterest"] > 0]
        df_c = df_c[df_c["strike"] > 0]
        # midPrice: (bid+ask)/2 si ambos disponibles, else lastPrice
        df_c["midPrice"] = np.where(
            (df_c["bid"] > 0) & (df_c["ask"] > 0),
            0.5*(df_c["bid"] + df_c["ask"]),
            df_c["lastPrice"]
        )
        df_c = df_c[df_c["midPrice"] > 0]
        df_c = df_c.dropna(subset=["strike","midPrice"])
        df_c["moneyness"] = df_c["strike"] / spot_px
        return df_c.sort_values("strike").reset_index(drop=True)

    sess = _yahoo_options_session()
    if sess is not None:
        try:
            log.info(f"Options {ticker}: curl_cffi Chrome impersonation")
            base = f"https://query1.finance.yahoo.com/v8/finance/options/{ticker}"
            hdrs = {"Accept":"application/json","Referer":"https://finance.yahoo.com/",
                    "Accept-Language":"en-US,en;q=0.9"}
            r0   = sess.get(base, headers=hdrs, timeout=15)
            r0.raise_for_status()
            root = r0.json()["optionChain"]["result"][0]
            spot = float(root["quote"].get("regularMarketPrice", 0))
            if not spot: raise ValueError("spot=0")
            timestamps = root.get("expirationDates", [])
            today  = date.today()
            chains = {}
            sel = sorted(
                [(ts, datetime.fromtimestamp(ts).date().strftime("%Y-%m-%d"),
                  (datetime.fromtimestamp(ts).date() - today).days)
                 for ts in timestamps
                 if (datetime.fromtimestamp(ts).date() - today).days >= 7],
                key=lambda x: x[2])[:n_exp]
            for ts, exp_str, dte in sel:
                time.sleep(0.6)
                try:
                    rx = sess.get(f"{base}?date={ts}", headers=hdrs, timeout=15)
                    rx.raise_for_status()
                    opts = rx.json()["optionChain"]["result"][0]["options"][0]
                    c_df = _clean(pd.DataFrame(opts.get("calls",[])), spot)
                    p_df = _clean(pd.DataFrame(opts.get("puts", [])), spot)
                    if len(c_df) < 3 or len(p_df) < 3: continue
                    chains[exp_str] = {"calls":c_df,"puts":p_df,"dte":dte}
                except Exception as ex:
                    log.warning(f"curl_cffi chain {ticker} {exp_str}: {ex}")
            if chains:
                log.info(f"curl_cffi OK {ticker}: {len(chains)} chains · spot={spot:.2f}")
                return chains, spot
            log.warning(f"curl_cffi: no chains for {ticker} — fallback yfinance")
        except Exception as e:
            log.warning(f"curl_cffi failed {ticker}: {e} — fallback yfinance")

    # ── Fallback: yfinance con backoff ─────────────────────────────────────
    log.info(f"Options {ticker}: yfinance fallback")
    try:
        t = yf.Ticker(ticker)
        def _bo(fn, label, n=4):
            for i in range(n):
                try: return fn()
                except Exception as ex:
                    if any(k in str(ex).lower() for k in
                           ["rate limit","too many","429","throttle"]) and i < n-1:
                        w = 2**(i+1); log.warning(f"{label} RL→{w}s"); time.sleep(w)
                    else: raise
            return None
        exps = _bo(lambda: t.options, f"{ticker}.options")
        if not exps: return {}, None
        time.sleep(0.8)
        hist = _bo(lambda: t.history(period="2d"), f"{ticker}.hist")
        spot = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        if not spot: return {}, None
        today  = date.today()
        valid  = sorted(
            [(e,(datetime.strptime(e,"%Y-%m-%d").date()-today).days)
             for e in exps
             if (datetime.strptime(e,"%Y-%m-%d").date()-today).days >= 7],
            key=lambda x: x[1])[:n_exp]
        time.sleep(1.0)
        chains = {}; streak = 0
        for exp_str, dte in valid:
            if streak >= 2: time.sleep(12); streak = 0
            try:
                ch = _bo(lambda e=exp_str: t.option_chain(e), f"{ticker}.chain.{exp_str}")
                if ch is None: continue
                streak = 0
                chains[exp_str] = {
                    "calls": _clean(ch.calls, spot),
                    "puts":  _clean(ch.puts,  spot),
                    "dte":   dte,
                }
                time.sleep(1.5)
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
    Aplica BS IV (Brent) a cada opción de cada chain.
    Agrega columna 'iv' (float, annualizado) y filtra NaN.
    r = risk-free rate anualizado
    q = dividend yield continuo
    """
    result = {}
    for exp_str, data in chains.items():
        dte = data["dte"]
        T   = dte / 365.0
        if T <= 0:
            continue
        for side, opt_type in [("calls","C"), ("puts","P")]:
            df = data[side].copy()
            df["iv"] = df.apply(
                lambda row: _bs_iv(spot, row["strike"], r, T,
                                   row["midPrice"], opt_type, q),
                axis=1
            )
            df = df[df["iv"].notna() & (df["iv"] > 0.005) & (df["iv"] < 5.0)]
            data[side] = df.reset_index(drop=True)
        if len(data["calls"]) >= 3 and len(data["puts"]) >= 3:
            result[exp_str] = data
    return result

# ─── Métricas de skew (usa columna 'iv' BS) ────────────────────────────────
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
# COT — Commitments of Traders via cot_reports library
# Fuente: CFTC "Traders in Financial Futures" report
# VIX Futures: buscamos "VIX" en Market and Exchange Names
# Leveraged Funds ≈ Managed Money (hedge funds / CTAs)
# Asset Manager ≈ Institucionales pasivos
# Publicación: cada martes ~15:30 ET con datos del martes anterior
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600 * 6)
def fetch_cot_vix(n_weeks: int = 104) -> pd.DataFrame:
    """
    Descarga el COT de Traders in Financial Futures para VIX via cot_reports.
    Columnas clave:
      - mm_long / mm_short   : Leveraged Funds (hedge funds/CTAs)
      - asset_long / short   : Asset Managers (institucionales)
      - dealer_long / short  : Dealer Intermediaries
      - net_mm               : Leveraged Funds net
      - net_mm_pct           : % del Open Interest
    """
    log = logging.getLogger("vix_controller")
    try:
        import cot_reports
    except ImportError:
        log.error("cot_reports no instalado. Agrega 'cot_reports' a requirements.txt")
        return pd.DataFrame()

    try:
        current_year = now_cdmx().year
        years_needed = max(1, (n_weeks // 52) + 2)
        frames = []
        for yr in range(current_year - years_needed + 1, current_year + 1):
            try:
                df_yr = cot_reports.cot_year(
                    year=yr,
                    cot_report_type="traders_in_financial_futures_fut"
                )
                frames.append(df_yr)
                log.info(f"COT: año {yr} OK ({len(df_yr)} filas)")
            except Exception as e:
                log.warning(f"COT año {yr}: {e}")
                continue

        if not frames:
            log.error("COT: no se pudo descargar ningún año")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Filtrar VIX futures
        mask = df["Market and Exchange Names"].str.contains("VIX", case=False, na=False)
        df   = df[mask].copy()
        if df.empty:
            log.error("COT: no se encontraron filas de VIX futures")
            return pd.DataFrame()

        # Parsear fecha
        date_col = "As of Date in Form YYYY-MM-DD"
        if date_col in df.columns:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            # Alternativa: buscar columna con fecha
            date_cols = [c for c in df.columns if "date" in c.lower() or "yyyy" in c.lower()]
            df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce") if date_cols else pd.NaT

        df = df.sort_values("date").reset_index(drop=True)

        # Renombrar columnas (TFF report)
        col_map = {
            "Open Interest (All)":                        "oi",
            "Leveraged Funds-Long (All)":                 "mm_long",
            "Leveraged Funds-Short (All)":                "mm_short",
            "Leveraged Funds-Spreading (All)":            "mm_spread",
            "Asset Manager/Institutional-Long (All)":     "asset_long",
            "Asset Manager/Institutional-Short (All)":    "asset_short",
            "Dealer Intermediary-Long (All)":             "dealer_long",
            "Dealer Intermediary-Short (All)":            "dealer_short",
            "Other Reportables-Long (All)":               "other_long",
            "Other Reportables-Short (All)":              "other_short",
        }
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

        # Convertir a numérico
        for c in ["oi","mm_long","mm_short","mm_spread","asset_long","asset_short",
                  "dealer_long","dealer_short","other_long","other_short"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Métricas derivadas
        if "mm_long" in df.columns and "mm_short" in df.columns:
            df["net_mm"]     = df["mm_long"] - df["mm_short"]
            df["net_mm_pct"] = (df["net_mm"] / df["oi"] * 100).where(df["oi"] > 0)
            df["net_mm_pct_pctile"] = df["net_mm_pct"].rank(pct=True) * 100

        if "dealer_long" in df.columns and "dealer_short" in df.columns:
            df["net_dealer"] = df["dealer_long"] - df["dealer_short"]

        if "asset_long" in df.columns and "asset_short" in df.columns:
            df["net_commercial"] = df["asset_long"] - df["asset_short"]

        last_ok = df["date"].dropna().iloc[-1].strftime("%Y-%m-%d") if not df["date"].dropna().empty else "?"
        log.info(f"COT VIX (TFF): {len(df)} semanas · última: {last_ok}")
        return df.tail(n_weeks).reset_index(drop=True)

    except Exception as e:
        log.error(f"fetch_cot_vix: {e}")
        return pd.DataFrame()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COT — Commitments of Traders (CFTC Public API)
# Futuros VIX: código CFTC 1170E1 · Disaggregated Report
# API: https://publicreporting.cftc.gov (Socrata, sin auth)
# Publicación: martes ~15:30 ET con datos del martes anterior
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COT_VIX_CODE   = "1170E1"                        # CBOE VIX VOLATILITY INDEX
COT_API_BASE   = "https://publicreporting.cftc.gov/resource"
COT_DISAGG_ID  = "72hh-3qpy"                     # Disaggregated Futures & Options Combined
COT_LEGACY_ID  = "6dca-aqww"                     # Legacy (si disagg falla)


def build_cot_positioning_chart(cot_df, window=104):
    """Net Managed Money positioning + percentile bands."""
    p = cot_df.tail(window).copy()
    if 'net_mm' not in p.columns or 'date' not in p.columns:
        return go.Figure()
    p = p.dropna(subset=['net_mm', 'date'])
    if len(p) < 5:
        return go.Figure()
    colors = ['#3FB950' if v >= 0 else '#F85149' for v in p['net_mm']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p['date'], y=p['net_mm'], marker_color=colors,
        name='Net MM', opacity=0.7))
    if 'net_mm_pct' in p.columns:
        fig.add_trace(go.Scatter(x=p['date'], y=p['net_mm'].rolling(8).mean(),
            name='SMA(8w)', line=dict(color='#39D2C0', width=2)))
    fig.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=1)
    fig.update_layout(
        title=dict(text='<b>Managed Money Net Positioning</b><sup>  VIX Futures · CFTC COT</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=350, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Contratos (net)', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_cot_oi_chart(cot_df, window=104):
    """Open Interest total."""
    p = cot_df.tail(window).copy()
    if 'oi' not in p.columns or 'date' not in p.columns:
        return go.Figure()
    p = p.dropna(subset=['oi', 'date'])
    if len(p) < 5:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p['date'], y=p['oi'], name='Open Interest',
        line=dict(color='#58A6FF', width=2), fill='tozeroy',
        fillcolor='rgba(88,166,255,0.1)'))
    fig.update_layout(
        title=dict(text='<b>Open Interest</b><sup>  VIX Futures</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Contratos', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        hovermode='x unified', showlegend=False)
    return fig


def build_cot_breakdown_chart(cot_df, window=104):
    """Breakdown: MM, Dealers, Asset Managers."""
    p = cot_df.tail(window).copy()
    if 'date' not in p.columns:
        return go.Figure()
    p = p.dropna(subset=['date'])
    fig = go.Figure()
    traces = [
        ('net_mm', 'Managed Money', '#3FB950'),
        ('net_dealer', 'Dealers', '#F0883E'),
        ('net_commercial', 'Asset Managers', '#58A6FF'),
    ]
    for col, name, color in traces:
        if col in p.columns:
            fig.add_trace(go.Scatter(x=p['date'], y=p[col], name=name,
                line=dict(color=color, width=2)))
    fig.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=1)
    fig.update_layout(
        title=dict(text='<b>Net Positioning by Category</b>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Contratos (net)', gridcolor='#21262D',
                   tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


# Sin cache — recibe DataFrame (unhashable)
def compute_edge_analytics(df, edge_extra):
    out = {}
    bt = df[df['VIX_Close'].notna() & df['SPY_Close'].notna()].copy()
    if len(bt) < 60:
        return out

    log_ret = np.log(bt['SPY_Close'] / bt['SPY_Close'].shift(1))
    bt['RV5']  = log_ret.rolling(5).std()  * np.sqrt(252) * 100
    bt['RV10'] = log_ret.rolling(10).std() * np.sqrt(252) * 100
    bt['RV20'] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    bt['RV60'] = log_ret.rolling(60).std() * np.sqrt(252) * 100
    bt['VRP']  = bt['VIX_Close'] - bt['RV20']

    vrp_2y = bt['VRP'].tail(504).dropna()
    if len(vrp_2y) > 20:
        out['vrp_percentile'] = round((vrp_2y < vrp_2y.iloc[-1]).mean() * 100, 0)

    if 'M1_Price' in bt.columns and 'M1_DTE' in bt.columns:
        m1 = bt['M1_Price']; dte = bt['M1_DTE']; spot = bt['VIX_Close']
        valid = (m1 > 0) & (dte > 0) & m1.notna() & dte.notna() & spot.notna()
        bt['Roll_Yield'] = np.where(valid, (m1 - spot) / m1 * (365 / dte) * 100, np.nan)

    if 'VVIX_Close' in bt.columns:
        bt['VVIX_VIX'] = np.where(bt['VIX_Close'] > 0, bt['VVIX_Close'] / bt['VIX_Close'], np.nan)

    if 'SKEW' in edge_extra and not edge_extra['SKEW'].empty:
        skew_df = edge_extra['SKEW'][['Close']].rename(columns={'Close': 'SKEW'})
        bt = bt.join(skew_df, how='left')

    if 'HYG' in edge_extra and 'IEF' in edge_extra:
        hyg = edge_extra['HYG'][['Close']].rename(columns={'Close': 'HYG'})
        ief = edge_extra['IEF'][['Close']].rename(columns={'Close': 'IEF'})
        bt = bt.join(hyg, how='left').join(ief, how='left')
        if 'HYG' in bt.columns and 'IEF' in bt.columns:
            bt['Credit_Spread'] = -(bt['HYG'].pct_change().rolling(20).sum() -
                                    bt['IEF'].pct_change().rolling(20).sum()) * 100

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
    p = bt.tail(window).dropna(subset=['VRP'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['VIX_Close'], name='VIX (Implied)',
        line=dict(color='#F85149', width=2)))
    fig.add_trace(go.Scatter(x=p.index, y=p['RV20'], name='RV20 (Realized)',
        line=dict(color='#58A6FF', width=2)))
    fig.add_trace(go.Scatter(x=p.index, y=p['VRP'], name='VRP (IV - RV)',
        fill='tozeroy', line=dict(color='#3FB950', width=1),
        fillcolor='rgba(63,185,80,0.15)'))
    fig.add_hline(y=0, line_dash='dash', line_color='#8B949E', line_width=1)
    fig.update_layout(
        title=dict(text='<b>Volatility Risk Premium</b><sup>  VIX - RV20 · Tu edge en puntos</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=350, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Vol Points', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        legend=dict(orientation='h', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


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
    if 'VVIX_VIX' not in bt.columns:
        return go.Figure()
    p = bt.tail(window).dropna(subset=['VVIX_VIX'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['VVIX_VIX'], name='VVIX/VIX',
        line=dict(color='#BC8CFF', width=2)))
    fig.add_hline(y=6, line_dash='dash', line_color='#F85149', line_width=1.5,
        annotation_text='  Danger > 6', annotation_font=dict(color='#F85149', size=10))
    fig.add_hline(y=5, line_dash='dot', line_color='#D29922', line_width=1,
        annotation_text='  Warning > 5', annotation_font=dict(color='#D29922', size=9))
    fig.update_layout(
        title=dict(text='<b>VVIX / VIX Ratio</b><sup>  > 6 = dealers anticipan spike</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=300, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Ratio', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        hovermode='x unified')
    return fig


def build_skew_chart(bt, window=252):
    if 'SKEW' not in bt.columns:
        return go.Figure()
    p = bt.tail(window).dropna(subset=['SKEW'])
    if len(p) < 10:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['SKEW'], name='CBOE SKEW',
        line=dict(color='#F0883E', width=2)))
    fig.add_hline(y=p['SKEW'].mean(), line_dash='dot', line_color='#8B949E', line_width=1,
        annotation_text=f'  Media: {p["SKEW"].mean():.0f}',
        annotation_font=dict(color='#8B949E', size=9))
    fig.add_hline(y=150, line_dash='dash', line_color='#F85149', line_width=1,
        annotation_text='  Extremo > 150', annotation_font=dict(color='#F85149', size=9))
    fig.update_layout(
        title=dict(text='<b>CBOE SKEW</b><sup>  Demanda de proteccion · > 150 = extremo</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='SKEW', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
        hovermode='x unified', showlegend=False)
    return fig


def build_credit_chart(bt, window=252):
    if 'Credit_Spread' not in bt.columns:
        return go.Figure()
    p = bt.tail(window).dropna(subset=['Credit_Spread'])
    if len(p) < 10:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p['Credit_Spread'], name='Credit Spread (HYG-IEF)',
        line=dict(color='#D29922', width=2)))
    fig.add_trace(go.Scatter(x=p.index, y=p['VIX_Close'], name='VIX', yaxis='y2',
        line=dict(color='#F85149', width=1.5, dash='dot')))
    fig.update_layout(
        title=dict(text='<b>Credit Spread vs VIX</b><sup>  Divergencia = warning</sup>',
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=50, r=60, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Credit Spread', gridcolor='#21262D', tickfont=dict(size=9, color='#8B949E')),
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
    Gráfica operativa VXX con dos subpaneles:

    Panel 1 — VXX + SMA(20) + BB 2σ:
      · Zona verde      : LONG activo (sig_final==1)
      · Zona roja tenue : Backwardation (sig_bb==1 pero ct==0)
      · ▲ verde         : Entrada (sig_final 0→1)
      · ▼ naranja       : Salida por BB (VXX cruzó BB_Upper)
      · ▼ rojo          : Salida por Contango Rule (CT se apagó)
      · 💎 hoy          : precio actual (verde=LONG, rojo=CASH)

    Panel 2 — Contango % histórico (barras verdes/rojas del CSV)
              + punto de hoy en CBOE live
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03,
    )

    sig    = bt['sig_final']
    sig_bb = bt['sig_bb']
    ct     = bt['ct_filter']
    vxx    = bt['VXX_Close']
    y_top  = vxx.max() * 1.25

    # ── Zona LONG (verde) ─────────────────────────────────────
    long_y = np.where(sig == 1, y_top, np.nan)
    fig.add_trace(go.Scatter(
        x=bt.index, y=long_y, mode='none',
        fill='tozeroy', fillcolor='rgba(63,185,80,0.09)',
        showlegend=True, name='LONG activo', hoverinfo='skip',
    ), row=1, col=1)

    # ── Zona Backwardation (rojo tenue) ───────────────────────
    bkwd_y = np.where((sig_bb == 1) & (ct == 0), y_top, np.nan)
    fig.add_trace(go.Scatter(
        x=bt.index, y=bkwd_y, mode='none',
        fill='tozeroy', fillcolor='rgba(248,81,73,0.07)',
        showlegend=True, name='Backwardation', hoverinfo='skip',
    ), row=1, col=1)

    # ── BB + SMA + VXX ────────────────────────────────────────
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_Upper'],
        mode='lines', name='BB 2σ',
        line=dict(color='#F85149', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_Lower'],
        mode='lines', showlegend=False,
        line=dict(color='#F85149', width=0.5, dash='dot'),
        fill='tonexty', fillcolor='rgba(248,81,73,0.03)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt['BB_SMA20'],
        mode='lines', name='SMA(20)',
        line=dict(color='#58A6FF', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=vxx,
        mode='lines', name='VXX',
        line=dict(color='#F0F6FC', width=2),
        hovertemplate='%{x|%Y-%m-%d}  VXX: $%{y:.2f}<extra></extra>'), row=1, col=1)

    # ── Flechas ───────────────────────────────────────────────
    for i in range(1, len(sig)):
        date     = sig.index[i]
        y_val    = vxx.iloc[i]
        prev_sig = sig.iloc[i-1];   cur_sig  = sig.iloc[i]
        prev_bb  = sig_bb.iloc[i-1]; cur_bb  = sig_bb.iloc[i]
        prev_ct  = ct.iloc[i-1];    cur_ct   = ct.iloc[i]

        if cur_sig == 1 and prev_sig == 0:
            # Entrada
            fig.add_annotation(x=date, y=y_val, yshift=-22,
                text="▲", showarrow=False,
                font=dict(size=16, color='#3FB950', family='JetBrains Mono'),
                row=1, col=1)
        elif cur_sig == 0 and prev_sig == 1:
            if cur_bb == 0 and prev_bb == 1:
                # Salida por BB (naranja)
                fig.add_annotation(x=date, y=y_val, yshift=22,
                    text="▼", showarrow=False,
                    font=dict(size=16, color='#D29922', family='JetBrains Mono'),
                    row=1, col=1)
            elif cur_ct == 0 and prev_ct == 1:
                # Salida por Contango Rule (rojo)
                fig.add_annotation(x=date, y=y_val, yshift=22,
                    text="▼", showarrow=False,
                    font=dict(size=16, color='#F85149', family='JetBrains Mono'),
                    row=1, col=1)
            else:
                # Ambas (naranja — BB dominó)
                fig.add_annotation(x=date, y=y_val, yshift=22,
                    text="▼", showarrow=False,
                    font=dict(size=16, color='#D29922', family='JetBrains Mono'),
                    row=1, col=1)

    # Punto de hoy
    today_clr = '#3FB950' if final_sig_today else '#F85149'
    fig.add_trace(go.Scatter(
        x=[bt.index[-1]], y=[vxx_today],
        mode='markers', name='HOY — LONG' if final_sig_today else 'HOY — CASH',
        marker=dict(size=14, color=today_clr,
                    line=dict(width=2, color='white'), symbol='diamond'),
        hovertemplate=f'HOY: ${vxx_today:.2f}<extra></extra>',
    ), row=1, col=1)

    # ── Panel 2: Contango histórico ───────────────────────────
    if 'Contango_pct' in bt.columns:
        ct_hist  = bt['Contango_pct'].fillna(0)
        bar_clrs = ['#3FB950' if v > 0 else '#F85149' for v in ct_hist]
        fig.add_trace(go.Bar(
            x=bt.index, y=ct_hist,
            name='Contango %', marker_color=bar_clrs, opacity=0.7,
            hovertemplate='%{x|%Y-%m-%d}  CT: %{y:+.2f}%<extra></extra>',
        ), row=2, col=1)
        if ct_today is not None:
            ct_clr = '#3FB950' if ct_today > 0 else '#F85149'
            fig.add_trace(go.Scatter(
                x=[bt.index[-1]], y=[ct_today],
                mode='markers', name=f'CT hoy: {ct_today:+.2f}%',
                marker=dict(size=10, color=ct_clr, symbol='diamond',
                            line=dict(width=2, color='white')),
            ), row=2, col=1)
        fig.add_hline(y=0, line_color='#484F58', line_width=1, row=2, col=1)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>VXX — Monitor Operativo BB(20, 2σ) + Contango Rule</b>"
                 "<sup>  ▲=Entrada  ▼🟡=Salida BB  ▼🔴=Salida CT  💎=Hoy</sup>",
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5,
        ),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=560, margin=dict(l=55, r=30, t=65, b=40),
        xaxis=dict(
            gridcolor='#21262D',
            tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono'),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1A",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3A",  step="year",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ],
                bgcolor='#161B22', activecolor='#F7931A',
                font=dict(size=9, color='#C9D1D9', family='JetBrains Mono'),
            ),
        ),
        xaxis2=dict(gridcolor='#21262D',
                    tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title=dict(text="VXX ($)", font=dict(size=11, color='#8B949E')),
                   gridcolor='#21262D',
                   tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis2=dict(title=dict(text="Contango %", font=dict(size=10, color='#8B949E')),
                    gridcolor='#21262D',
                    tickfont=dict(size=9, color='#8B949E', family='JetBrains Mono'),
                    zeroline=True, zerolinecolor='#30363D'),
        legend=dict(orientation='h', yanchor='bottom', y=1.05,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified', dragmode=False, bargap=0,
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
tab1, tab2, tab_edge, tab_skew, tab_cot, tab3, tab4 = st.tabs([
    "📈  Term Structure",
    "🎯  Monitor Operativo",
    "🔬  Edge Analytics",
    "📐  Vol Skew & Surface",
    "📋  COT · Futuros VIX",
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
        f"▲=Entrada  ▼🟡=Salida BB  ▼🔴=Salida CT"
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

            vrp_val = last_e.get('VRP', np.nan)
            vrp_pct = edge.get('vrp_percentile', '?')
            rv20_val = last_e.get('RV20', np.nan)
            ry_val = last_e.get('Roll_Yield', np.nan)
            vvix_r = last_e.get('VVIX_VIX', np.nan)
            skew_val = last_e.get('SKEW', np.nan)

            vrp_str = f"{vrp_val:+.1f}" if pd.notna(vrp_val) else "N/A"
            vrp_clr = "up" if pd.notna(vrp_val) and vrp_val > 2 else "dn" if pd.notna(vrp_val) and vrp_val < 0 else "nt"
            ry_str = f"{ry_val:+.1f}%" if pd.notna(ry_val) else "N/A"
            ry_clr = "up" if pd.notna(ry_val) and ry_val > 0 else "dn" if pd.notna(ry_val) and ry_val < 0 else "nt"
            vvix_str = f"{vvix_r:.2f}" if pd.notna(vvix_r) else "N/A"
            vvix_clr = "dn" if pd.notna(vvix_r) and vvix_r > 6 else "up" if pd.notna(vvix_r) and vvix_r < 5 else "nt"
            skew_str = f"{skew_val:.0f}" if pd.notna(skew_val) else "N/A"
            skew_clr = "dn" if pd.notna(skew_val) and skew_val > 150 else "up" if pd.notna(skew_val) and skew_val < 130 else "nt"

            st.markdown(f"""<div class="mrow">
                {ecard("VRP (IV-RV)", vrp_str, f"P{vrp_pct} hist" if vrp_pct != '?' else "", vrp_clr)}
                {ecard("RV20 (SPX)", f"{rv20_val:.1f}" if pd.notna(rv20_val) else "N/A", "Realized Vol 20d", "nt")}
                {ecard("Roll Yield", ry_str, "Carry anualizado", ry_clr)}
                {ecard("VVIX/VIX", vvix_str, "> 6 = peligro", vvix_clr)}
                {ecard("SKEW", skew_str, "> 150 = extremo", skew_clr)}
                {ecard("VIX", f"{last_e['VIX_Close']:.1f}", f"RV20: {rv20_val:.1f}" if pd.notna(rv20_val) else "", "nt")}
            </div>""", unsafe_allow_html=True)

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
                warnings_e.append("VRP negativo — estas pagando por estar posicionado")
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

            # Charts
            try:
                st.plotly_chart(build_vrp_chart(ebt), width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error VRP: {e}")

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

            st.caption(f"Edge Analytics · Ventana: 1 ano · "
                       f"Fuentes: Master Parquet + Yahoo Finance (SKEW, HYG, IEF)")


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
        mon_lo = st.slider("Strike mín (%spot)", 70, 90, 80) / 100
    with col_c6:
        mon_hi = st.slider("Strike máx (%spot)", 110, 140, 125) / 100
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
            st.dataframe(pd.DataFrame(rows_tbl), width="stretch", hide_index=True)

    st.caption(
        f"IV calculada con Black-Scholes (Brent) · r={skew_rfr:.1%} · q={skew_div:.1%} · "
        f"Spot {skew_ticker}: {spot_disp} · {now_cdmx().strftime('%H:%M:%S')} CDMX"
    )


# ━━━━━━━━━━━━━━━━━ TAB: COT — COMMITMENTS OF TRADERS ━━━━━━━━
with tab_cot:

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8B949E;
                padding:0.4rem 0 0.8rem;">
    Fuente: <b>CFTC Disaggregated COT Report</b> · Código VIX Futures: <b>1170E1</b> ·
    Publicación: martes ~15:30 ET con datos del martes anterior ·
    API: publicreporting.cftc.gov (gratuita, sin autenticación)
    </div>
    """, unsafe_allow_html=True)

    col_cot1, col_cot2 = st.columns([3, 1])
    with col_cot1:
        cot_weeks = st.slider("Semanas de historia", 26, 156, 104,
                              help="1 año = 52 semanas · 3 años = 156")
    with col_cot2:
        if st.button("🔄 Actualizar COT", key="btn_refresh_cot"):
            fetch_cot_vix.clear()
            st.rerun()

    with st.spinner("📋 Descargando COT de CFTC…"):
        cot_df = fetch_cot_vix(n_weeks=max(cot_weeks + 10, 156))

    if cot_df.empty:
        st.error("❌ No se pudieron obtener datos COT del CFTC. Verifica conexión.")
        st.info(
            "La API CFTC (publicreporting.cftc.gov) es pública y gratuita. "
            "Si falla, intenta de nuevo en unos minutos — el caché es de 6 horas."
        )
    else:
        last_cot = cot_df.iloc[-1]
        last_date_cot = last_cot["date"].strftime("%Y-%m-%d") if pd.notna(last_cot["date"]) else "?"

        # ── Métricas de resumen ──────────────────────────────────
        mm_net    = int(last_cot.get("net_mm", 0) or 0)
        mm_pct    = last_cot.get("net_mm_pct", None)
        mm_pctile = last_cot.get("net_mm_pct_pctile", None)
        oi_val    = int(last_cot.get("oi", 0) or 0)
        dealer_net = int(last_cot.get("net_dealer", 0) or 0)
        comm_net   = int(last_cot.get("net_commercial", 0) or 0)

        # Señal interpretativa
        if mm_pctile is not None:
            if mm_net > 0 and mm_pctile > 70:
                cot_signal = "⚡ MM NET LONG extremo — alta demanda de vol"
                cot_sig_clr = "var(--r)"
                cot_interp = ("Managed Money está net long VIX futures por encima del percentil 70 histórico. "
                              "El mercado está pagando prima de volatilidad elevada — favorable para estrategias de venta de vol "
                              "pero indica precaución: el mercado anticipa movimiento.")
            elif mm_net > 0:
                cot_signal = "📈 MM NET LONG moderado"
                cot_sig_clr = "var(--y)"
                cot_interp = ("Managed Money tiene posición neta long en VIX futures — expectativa moderada de vol. "
                              "El contango puede estar bajo presión.")
            elif mm_net < 0 and mm_pctile is not None and mm_pctile < 30:
                cot_signal = "✅ MM NET SHORT — complacencia elevada"
                cot_sig_clr = "var(--g)"
                cot_interp = ("Managed Money está net short VIX futures — los especuladores apuestan a que la vol baja. "
                              "Históricamente favorable para estrategias de inverse vol como SVXY/SVIX. "
                              "Señal de complacencia: el mercado no anticipa volatilidad.")
            else:
                cot_signal = "➡️ Posicionamiento neutral"
                cot_sig_clr = "var(--b)"
                cot_interp = "Managed Money está cerca del equilibrio en futuros VIX."
        else:
            cot_signal = "—"
            cot_sig_clr = "var(--dim)"
            cot_interp  = ""

        mm_pct_s   = f"{mm_pct:+.1f}% del OI" if mm_pct is not None else "—"
        mm_pctile_s = f"Pct {mm_pctile:.0f}°" if mm_pctile is not None else "—"

        st.markdown(f"""
        <div class="mrow">
            <div class="mpill" style="min-width:180px">
                <div class="ml">Señal COT</div>
                <div style="font-family:'Inter',sans-serif;font-weight:700;font-size:0.9rem;
                            color:{cot_sig_clr}">{cot_signal}</div>
            </div>
            <div class="mpill">
                <div class="ml">Net MM · {last_date_cot}</div>
                <div class="mv {'up' if mm_net>=0 else 'dn'}">{mm_net:+,}</div>
            </div>
            <div class="mpill">
                <div class="ml">Net MM % OI</div>
                <div class="mv nt">{mm_pct_s}</div>
            </div>
            <div class="mpill">
                <div class="ml">Percentil histórico</div>
                <div class="mv nt">{mm_pctile_s}</div>
            </div>
            <div class="mpill">
                <div class="ml">Open Interest</div>
                <div class="mv nt">{oi_val:,}</div>
            </div>
            <div class="mpill">
                <div class="ml">Net Dealers</div>
                <div class="mv {'up' if dealer_net>=0 else 'dn'}">{dealer_net:+,}</div>
            </div>
            <div class="mpill">
                <div class="ml">Net Commercial</div>
                <div class="mv {'up' if comm_net>=0 else 'dn'}">{comm_net:+,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if cot_interp:
            with st.expander("📊 Interpretación COT", expanded=True):
                st.markdown(cot_interp)
                st.markdown("""
**Guía de lectura rápida:**
- **Managed Money net LONG VIX** → especuladores apuestan a subida de vol → mercado defensivo
- **Managed Money net SHORT VIX** → especuladores apuestan a vol baja → contango favorable
- **OI creciente + MM net short** → el trade de inverse vol tiene viento de cola
- **OI cayendo** → posiciones se están cerrando, reducir convicción
- El COT se publica **cada martes** con datos de la semana anterior
                """)

        st.markdown("<div style='border-top:1px solid #30363D;margin:0.5rem 0'></div>",
                    unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────────
        try:
            fig_pos = build_cot_positioning_chart(cot_df, window=cot_weeks)
            if fig_pos.data:
                st.plotly_chart(fig_pos, width="stretch", config=dict(displayModeBar=False))
        except Exception as e:
            st.error(f"Error chart posicionamiento: {e}")

        col_oi, col_bd = st.columns(2)
        with col_oi:
            try:
                fig_oi = build_cot_oi_chart(cot_df, window=cot_weeks)
                if fig_oi.data:
                    st.plotly_chart(fig_oi, width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error chart OI: {e}")

        with col_bd:
            try:
                fig_bd = build_cot_breakdown_chart(cot_df, window=min(cot_weeks, 104))
                if fig_bd.data:
                    st.plotly_chart(fig_bd, width="stretch", config=dict(displayModeBar=False))
            except Exception as e:
                st.error(f"Error chart breakdown: {e}")

        # ── Tabla histórica ─────────────────────────────────────
        with st.expander("📋 Datos semanales COT"):
            show_cols = [c for c in
                ["date","oi","mm_long","mm_short","net_mm","net_mm_pct","net_mm_pct_pctile",
                 "dealer_long","dealer_short","net_dealer","prod_long","prod_short","net_commercial"]
                if c in cot_df.columns]
            st.dataframe(
                cot_df[show_cols].tail(cot_weeks).sort_values("date", ascending=False),
                width="stretch", hide_index=True,
            )

        st.caption(
            f"CFTC Disaggregated COT · VIX Futures (1170E1) · "
            f"Última semana: {last_date_cot} · "
            f"Cache: 6h · publicreporting.cftc.gov"
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

    **Tab 1: Term Structure** — Réplica de VIXCentral.com
    - Datos scrapeados directamente de la tabla CBOE Delayed Quotes via **Playwright + Chromium**
    - Solo contratos mensuales (regex `^VX/[A-Z]\\d+$` — filtra weeklys como VX12, VX13, etc.)
    - Muestra columnas: **Last, Change, High, Low, Settlement, Volume** (como la tabla CBOE)
    - Tabla de contango/diferencia entre meses (estilo VIXCentral)
    - Month 7 to 4 contango
    - Auto-refresh cada 60 segundos

    **Tab 2: Monitor Operativo** — Señal BB × Contango
    - **BB Timing**: VXX < SMA(20) = LONG, VXX > BB Superior = EXIT
    - **Contango**: se alimenta automáticamente del term structure scrapeado
    - **Señal Final** = BB × Contango
    - Gráfico VXX + BB con zonas y flechas ENTRY/EXIT

    ---

    **Fuentes:**
    - `cboe.com/delayed_quotes/futures/future_quotes` — scrapeado con Playwright
    - Yahoo Finance — VIX spot, VXX, SVXY, SVIX, SPY

    **Para Streamlit Cloud necesitas:**
    - `packages.txt` con dependencias de Chromium
    - `requirements.txt` con playwright
    """)

st.markdown(f"""
<div style="text-align:center;padding:0.8rem 0 0.3rem;border-top:1px solid #30363D;margin-top:1rem;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#484F58;">
        VIX CONTROLLER · Alberto Alarcón González · Not financial advice
    </span>
</div>""", unsafe_allow_html=True)
