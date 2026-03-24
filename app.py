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
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

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
            f"{datetime.now().strftime('%H:%M:%S')}"
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

    today = pd.Timestamp('today').normalize()
    if 'Expiration' in df_vx.columns:
        df_vx['DTE'] = (df_vx['Expiration'] - today).dt.days

    df_vx['Price'] = df_vx.apply(
        lambda r: r['Last'] if pd.notna(r.get('Last')) and r.get('Last', 0) > 0
                  else r.get('Settlement', 0), axis=1
    )
    df_vx['Scraped_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
mkt_status = "MARKET OPEN" if datetime.now().hour >= 9 and datetime.now().hour < 16 and datetime.now().weekday() < 5 else "MARKET CLOSED"
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
    if st.button("📡 Refresh CBOE + yfinance", use_container_width=True):
        scrape_cboe_futures.clear()
        fetch_vix_spot.clear()
        fetch_etps.clear()
        fetch_today_prices.clear()
        st.rerun()
    if st.button("🗄️ Recargar Parquet (repo)", use_container_width=True):
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
tab1, tab2, tab3, tab4 = st.tabs(["📈  Term Structure", "🎯  Monitor Operativo", "💡  Recomendaciones", "ℹ️  Help"])

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

    exec_date = datetime.now().date() + timedelta(days=1)
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
    st.plotly_chart(fig_mon, use_container_width=True,
                    config=dict(displayModeBar=True, displaylogo=False,
                                scrollZoom=False,
                                modeBarButtonsToRemove=['select2d','lasso2d']))

    st.caption(
        f"Histórico: {bt.index[0].strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} "
        f"({len(bt):,} días) · Parquet del repo · "
        f"Contango hoy: {ct_source} · "
        f"▲=Entrada  ▼🟡=Salida BB  ▼🔴=Salida CT"
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
