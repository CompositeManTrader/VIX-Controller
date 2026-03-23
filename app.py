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
    Verifica que Playwright y Chromium estén disponibles.
    Se ejecuta UNA sola vez por deployment (cache_resource).
    NO lanza el browser aquí — solo importa y verifica el ejecutable.
    """
    log = logging.getLogger("vix_controller")
    try:
        from playwright.sync_api import sync_playwright
        # Verificación mínima: solo lanzar y cerrar inmediatamente
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
            )
            browser.close()
        log.info("Playwright check: Chromium OK ✅")
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
.hdr{display:flex;align-items:center;padding:0.5rem 0;border-bottom:2px solid #F7931A;margin-bottom:0.8rem;}
.hdr .logo{font-family:'Inter',sans-serif;font-weight:800;font-size:1.3rem;color:#F7931A;letter-spacing:1px;}
.hdr .sub{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-left:auto;}
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
/* Evitar que Plotly capture el scroll de la página */
.js-plotly-plot .plotly, .js-plotly-plot .plotly div{pointer-events:auto;}
iframe[title="streamlit_component"]{pointer-events:none;}
.element-container:has(.stPlotlyChart) .stPlotlyChart{overflow:visible;}
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
# MONITOR OPERATIVO — DATA LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DRIVE_FILE_ID = "12fzSq4BgkppRjoupeMjM67jCB8Qwo8Yz"
DRIVE_URL     = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

@st.cache_data(ttl=300)  # Refrescar cada 5 min (el CSV no cambia tan rápido)
def load_master_csv() -> pd.DataFrame:
    """Descarga el CSV maestro desde Google Drive vía gdown."""
    log = logging.getLogger("vix_controller")
    tmp = "/tmp/master_historico.csv"
    try:
        import gdown
        gdown.download(DRIVE_URL, tmp, quiet=True, fuzzy=True)
        df = pd.read_csv(tmp, index_col=0, parse_dates=True).sort_index()
        log.info(f"CSV cargado: {len(df):,} filas · {df.index[-1].strftime('%Y-%m-%d')}")
        return df
    except Exception as e:
        log.error(f"Error cargando CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=55)
def fetch_today_prices():
    """Precios del día de yfinance: VXX, SVXY, SVIX, VIX, SPY."""
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


def build_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la estrategia BB(20, 2σ) + Contango Rule sobre el DataFrame histórico.
    Retorna el DataFrame con columnas de señal, indicadores y trades.
    Lógica exacta del notebook:
      Entrada: VXX < SMA(20) → pos=1
      Salida:  VXX > BB_Upper(2σ) → pos=0
      Filtro:  In_Contango == 1
      Ejecución: shift(1) — señal de hoy se ejecuta mañana
    """
    bt = df[df['VXX_Close'].notna() & df['SVXY_Close'].notna() &
            df['M1_Price'].notna()].copy()

    vxx = bt['VXX_Close']
    bt['BB_SMA20']   = vxx.rolling(20).mean()
    bt['BB_STD20']   = vxx.rolling(20).std()
    bt['BB_Upper']   = bt['BB_SMA20'] + 2.0 * bt['BB_STD20']
    bt['BB_Lower']   = bt['BB_SMA20'] - 2.0 * bt['BB_STD20']

    # Señal BB pura
    sig = pd.Series(0, index=bt.index)
    pos = 0
    for i in range(len(bt)):
        p = bt['VXX_Close'].iloc[i]
        s = bt['BB_SMA20'].iloc[i]
        u = bt['BB_Upper'].iloc[i]
        if pd.isna(s) or pd.isna(u) or pd.isna(p):
            sig.iloc[i] = pos; continue
        if pos == 0 and p < s: pos = 1
        elif pos == 1 and p > u: pos = 0
        sig.iloc[i] = pos

    bt['sig_bb']      = sig.shift(1).fillna(0)
    bt['ct_filter']   = bt['In_Contango'].fillna(0).astype(int)
    bt['sig_final']   = (bt['sig_bb'] * bt['ct_filter']).astype(int)

    # Retorno estrategia
    bt['strat_ret'] = bt['SVXY_ret'] * bt['sig_final']
    bt['equity']    = (1 + bt['strat_ret'].fillna(0)).cumprod()

    return bt


def extract_trades(bt: pd.DataFrame) -> pd.DataFrame:
    """Extrae todos los trades (entrada/salida) del histórico."""
    sig = bt['sig_final']
    trades = []
    entry_date = None

    for i in range(1, len(sig)):
        # Entrada: 0 → 1
        if sig.iloc[i] == 1 and sig.iloc[i-1] == 0:
            entry_date = sig.index[i]
        # Salida: 1 → 0
        elif sig.iloc[i] == 0 and sig.iloc[i-1] == 1 and entry_date is not None:
            exit_date = sig.index[i]
            rets = bt['SVXY_ret'].loc[entry_date:exit_date].dropna()
            if len(rets) == 0:
                entry_date = None; continue

            trade_ret = (1 + rets).prod() - 1
            duration  = len(rets)

            # Razón de salida
            prev = sig.index[i-1]
            bb_exit = bt.loc[prev, 'VXX_Close'] > bt.loc[prev, 'BB_Upper'] if prev in bt.index else False
            ct_exit = bt.loc[prev, 'In_Contango'] == 0 if prev in bt.index else False
            if ct_exit and bb_exit: reason = 'Ambas'
            elif ct_exit:           reason = 'Contango Rule'
            else:                   reason = 'BB Superior'

            trades.append({
                'Entrada'        : entry_date.strftime('%Y-%m-%d'),
                'Salida'         : exit_date.strftime('%Y-%m-%d'),
                'Días'           : duration,
                'Retorno'        : round(trade_ret * 100, 2),
                'Razón salida'   : reason,
                'VIX entrada'    : round(bt.loc[entry_date, 'VIX_Close'], 1) if entry_date in bt.index else None,
                'Contango entr.' : round(bt.loc[entry_date, 'Contango_pct'], 2) if entry_date in bt.index else None,
            })
            entry_date = None

    # Trade abierto actualmente
    if entry_date is not None:
        rets = bt['SVXY_ret'].loc[entry_date:].dropna()
        trade_ret = (1 + rets).prod() - 1 if len(rets) > 0 else 0
        trades.append({
            'Entrada'        : entry_date.strftime('%Y-%m-%d'),
            'Salida'         : '🔴 ABIERTO',
            'Días'           : len(rets),
            'Retorno'        : round(trade_ret * 100, 2),
            'Razón salida'   : '—',
            'VIX entrada'    : round(bt.loc[entry_date, 'VIX_Close'], 1) if entry_date in bt.index else None,
            'Contango entr.' : round(bt.loc[entry_date, 'Contango_pct'], 2) if entry_date in bt.index else None,
        })

    return pd.DataFrame(trades)


def calc_metrics(bt: pd.DataFrame) -> dict:
    """Calcula métricas clave del backtest."""
    sr = bt['strat_ret'].dropna()
    if len(sr) < 50: return {}
    eq     = (1 + sr).cumprod()
    years  = len(eq) / 252
    cagr   = (eq.iloc[-1] ** (1/years) - 1) * 100
    peak   = eq.cummax()
    mdd    = ((eq - peak) / peak).min() * 100
    sharpe = sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else 0
    calmar = abs(cagr / mdd) if mdd != 0 else 0
    trades_in = (sr != 0)
    wr     = (sr[trades_in] > 0).mean() * 100 if trades_in.sum() > 0 else 0
    exp    = bt['sig_final'].mean() * 100
    # Retornos anuales
    yearly = {}
    for yr in sorted(set(eq.index.year)):
        yr_r = sr[sr.index.year == yr]
        if len(yr_r) > 20:
            yearly[yr] = round(((1 + yr_r).cumprod().iloc[-1] - 1) * 100, 1)
    return dict(cagr=round(cagr,1), mdd=round(mdd,1), sharpe=round(sharpe,2),
                calmar=round(calmar,2), wr=round(wr,1), exp=round(exp,1),
                yearly=yearly, equity=eq)


def build_bb_chart(bt: pd.DataFrame, window: int = 120) -> go.Figure:
    """Gráfico VXX + Bollinger Bands con zonas y flechas ENTRY/EXIT."""
    p = bt.tail(window).copy()

    # Normalizar nombres de columna — compatibilidad con ambos esquemas
    if 'SMA20' in p.columns and 'BB_SMA20' not in p.columns:
        p['BB_SMA20'] = p['SMA20']
    if 'BB_Upper' not in p.columns and 'BB_Upper_20' in p.columns:
        p['BB_Upper'] = p['BB_Upper_20']
    if 'BB_Lower' not in p.columns and 'BB_Lower_20' in p.columns:
        p['BB_Lower'] = p['BB_Lower_20']
    if 'sig_final' not in p.columns and 'bb_sig' in p.columns:
        p['sig_final'] = p['bb_sig']
    if 'VXX_Close' not in p.columns and 'VXX' in p.columns:
        p['VXX_Close'] = p['VXX']

    sig = p['sig_final']
    fig = go.Figure()

    # Zonas colored background
    for i in range(1, len(p)):
        clr = 'rgba(63,185,80,0.07)' if sig.iloc[i] == 1 else 'rgba(248,81,73,0.03)'
        fig.add_vrect(x0=p.index[i-1], x1=p.index[i],
                      fillcolor=clr, layer="below", line_width=0)

    # BB band fill
    fig.add_trace(go.Scatter(x=p.index, y=p['BB_Upper'], mode='lines',
        name='BB 2σ', line=dict(color='#F85149', width=1.2)))
    fig.add_trace(go.Scatter(x=p.index, y=p['BB_Lower'], mode='lines',
        name='BB Lower', line=dict(color='#F85149', width=0.5),
        fill='tonexty', fillcolor='rgba(88,166,255,0.03)', showlegend=False))
    fig.add_trace(go.Scatter(x=p.index, y=p['BB_SMA20'], mode='lines',
        name='SMA(20)', line=dict(color='#58A6FF', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=p.index, y=p['VXX_Close'], mode='lines',
        name='VXX', line=dict(color='#F0F6FC', width=2)))

    # ENTRY / EXIT arrows
    for i in range(1, len(p)):
        if sig.iloc[i] == 1 and sig.iloc[i-1] == 0:
            fig.add_annotation(x=p.index[i], y=p['VXX_Close'].iloc[i],
                text="▲ ENTRY", showarrow=True, arrowhead=2, arrowcolor="#3FB950",
                font=dict(size=9, color="#3FB950", family="JetBrains Mono"), ax=0, ay=25)
        elif sig.iloc[i] == 0 and sig.iloc[i-1] == 1:
            fig.add_annotation(x=p.index[i], y=p['VXX_Close'].iloc[i],
                text="▼ EXIT", showarrow=True, arrowhead=2, arrowcolor="#F85149",
                font=dict(size=9, color="#F85149", family="JetBrains Mono"), ax=0, ay=-25)

    # Today marker
    fig.add_trace(go.Scatter(x=[p.index[-1]], y=[p['VXX_Close'].iloc[-1]],
        mode='markers', name='Hoy',
        marker=dict(size=12, color='#D29922', line=dict(width=2, color='white')),
        showlegend=False))

    fig.update_layout(
        title=dict(text="<b>VXX + Bollinger Bands</b><sup>  BB(20, 2σ) · Verde=LONG SVXY · Rojo=CASH</sup>",
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=380, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title=dict(text="VXX", font=dict(size=11, color='#8B949E')),
                   gridcolor='#21262D',
                   tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
    return fig


def build_equity_chart(equity: pd.Series) -> go.Figure:
    """Equity curve con drawdown."""
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100

    fig = go.Figure()
    # Drawdown fill
    fig.add_trace(go.Scatter(x=equity.index, y=dd, mode='lines',
        name='Drawdown %', line=dict(color='#F85149', width=1),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.12)',
        yaxis='y2'))
    # Equity
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines',
        name='Equity ($1)', line=dict(color='#3FB950', width=2.5)))

    fig.update_layout(
        title=dict(text="<b>Equity Curve — SVXY BB(2σ) + Contango</b><sup>  Base $1 · 2018–hoy</sup>",
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=340, margin=dict(l=50, r=60, t=55, b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title='Equity ($)', gridcolor='#21262D',
                   tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis2=dict(title='Drawdown %', overlaying='y', side='right',
                    tickfont=dict(size=9, color='#F85149', family='JetBrains Mono'),
                    showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified',
        dragmode=False)
    return fig


def build_yearly_heatmap(yearly: dict) -> go.Figure:
    """Heatmap simple de retornos anuales."""
    years = sorted(yearly.keys())
    vals  = [yearly[y] for y in years]
    colors = ['#3FB950' if v >= 0 else '#F85149' for v in vals]

    fig = go.Figure(go.Bar(
        x=[str(y) for y in years], y=vals,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals],
        textposition='outside',
        textfont=dict(size=10, family='JetBrains Mono', color='#C9D1D9'),
    ))
    fig.update_layout(
        title=dict(text="<b>Retorno Anual</b>",
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=280, margin=dict(l=40, r=20, t=50, b=30),
        xaxis=dict(tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono'),
                   gridcolor='#21262D', zeroline=True, zerolinecolor='#30363D'),
        showlegend=False)
    return fig


def build_operational_chart(bt: pd.DataFrame, col_price: str,
                            label: str, color: str,
                            trades_df: pd.DataFrame,
                            today_price: float = None,
                            today_sig: int = 0) -> go.Figure:
    """
    Gráfica operativa genérica para VXX, SVXY o SVIX.
    Muestra precio histórico + zonas LONG/CASH + flechas entrada/salida
    + punto de hoy si hay precio live.
    col_price : columna del DataFrame (e.g. 'VXX_Close', 'SVXY_Close')
    """
    # Usar solo filas donde existe el precio
    p = bt[bt[col_price].notna()].copy()
    sig = p['sig_final']
    price_s = p[col_price]

    fig = go.Figure()

    # ── Zonas LONG / CASH ──────────────────────────────────────
    for i in range(1, len(p)):
        clr = 'rgba(63,185,80,0.07)' if sig.iloc[i] == 1 else 'rgba(248,81,73,0.025)'
        fig.add_vrect(x0=p.index[i-1], x1=p.index[i],
                      fillcolor=clr, layer="below", line_width=0)

    # ── Precio ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=price_s.index, y=price_s.values,
        mode='lines', name=label,
        line=dict(color=color, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>' + label + ': $%{y:.2f}<extra></extra>',
    ))

    # ── Flechas ENTRY / EXIT desde trades_df ──────────────────
    closed = trades_df[trades_df['Salida'] != '🔴 ABIERTO']
    open_t = trades_df[trades_df['Salida'] == '🔴 ABIERTO']

    for _, t in closed.iterrows():
        entry_d = pd.Timestamp(t['Entrada'])
        exit_d  = pd.Timestamp(t['Salida'])
        # Buscar precio más cercano a esa fecha
        if entry_d in price_s.index:
            ep = price_s.loc[entry_d]
        elif len(price_s.loc[price_s.index >= entry_d]) > 0:
            ep = price_s.loc[price_s.index >= entry_d].iloc[0]
        else:
            continue
        if exit_d in price_s.index:
            xp = price_s.loc[exit_d]
        elif len(price_s.loc[price_s.index >= exit_d]) > 0:
            xp = price_s.loc[price_s.index >= exit_d].iloc[0]
        else:
            continue

        fig.add_annotation(x=entry_d, y=ep,
            text="▲", showarrow=False,
            font=dict(size=13, color="#3FB950", family="JetBrains Mono"),
            yshift=-16)
        fig.add_annotation(x=exit_d, y=xp,
            text="▼", showarrow=False,
            font=dict(size=13, color="#F85149", family="JetBrains Mono"),
            yshift=16)

    # Trade abierto actualmente
    for _, t in open_t.iterrows():
        entry_d = pd.Timestamp(t['Entrada'])
        if entry_d in price_s.index:
            ep = price_s.loc[entry_d]
        elif len(price_s.loc[price_s.index >= entry_d]) > 0:
            ep = price_s.loc[price_s.index >= entry_d].iloc[0]
        else:
            continue
        fig.add_annotation(x=entry_d, y=ep,
            text="▲ OPEN", showarrow=True,
            arrowhead=2, arrowcolor="#3FB950",
            font=dict(size=9, color="#3FB950", family="JetBrains Mono"),
            ax=0, ay=30)

    # ── Punto de hoy ───────────────────────────────────────────
    if today_price and today_price > 0:
        today_clr = '#3FB950' if today_sig == 1 else '#F85149'
        today_lbl = '🟢 HOY — LONG' if today_sig == 1 else '🔴 HOY — CASH'
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp.now().normalize()],
            y=[today_price],
            mode='markers+text',
            name=today_lbl,
            text=[f"${today_price:.2f}"],
            textposition='top center',
            textfont=dict(size=10, color=today_clr, family='JetBrains Mono'),
            marker=dict(size=14, color=today_clr,
                        line=dict(width=2, color='white'), symbol='diamond'),
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — Operativa</b>"
                 f"<sup>  Verde=LONG SVXY · Rojo=CASH · ▲Entrada · ▼Salida</sup>",
            font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=400, margin=dict(l=55, r=30, t=55, b=40),
        xaxis=dict(
            gridcolor='#21262D', rangeslider=dict(visible=False),
            tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono'),
            rangeselector=dict(
                buttons=[
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
        yaxis=dict(
            title=dict(text=f"{label} ($)", font=dict(size=11, color='#8B949E')),
            gridcolor='#21262D',
            tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono'),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified',
    )
    return fig


@st.cache_data(ttl=300)  # mismo TTL que el CSV — solo recalcula si el CSV cambia
def get_strategy_data(df_master: pd.DataFrame):
    """
    Procesa todo el histórico de una vez y cachea el resultado.
    Se recalcula solo cuando el CSV cambia (TTL 5 min).
    Retorna bt, trades_df, metrics como tupla.
    """
    bt        = build_strategy(df_master)
    trades_df = extract_trades(bt)
    metrics   = calc_metrics(bt)
    return bt, trades_df, metrics


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
# AUTO-REFRESH — JS real timer (no requiere interacción del usuario)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFRESH_INTERVAL = 60  # segundos

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
if elapsed > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

# JS countdown + page reload automático cada 60s sin interacción
st.components.v1.html(f"""
<script>
(function() {{
    var remaining = {REFRESH_INTERVAL};
    var timer = setInterval(function() {{
        remaining--;
        var el = window.parent.document.getElementById('refresh-countdown');
        if (el) el.textContent = remaining + 's';
        if (remaining <= 0) {{
            clearInterval(timer);
            window.parent.location.reload();
        }}
    }}, 1000);
}})();
</script>
""", height=0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="hdr">
    <div class="logo">VIX CONTROLLER</div>
    <div class="sub">{now_str} · Auto-refresh in <span id="refresh-countdown">{REFRESH_INTERVAL}s</span> · Source: CBOE Delayed Quotes</div>
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
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
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
tab1, tab2, tab3 = st.tabs(["📈  Term Structure", "🎯  Monitor Operativo", "ℹ️  Help"])

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
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=True, displaylogo=False))

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

    # ── Cargar datos ──────────────────────────────────────────
    with st.spinner("📂 Cargando histórico desde Google Drive…"):
        df_master = load_master_csv()

    today_px = fetch_today_prices()

    if df_master.empty:
        st.error("❌ No se pudo cargar el CSV desde Google Drive. Verifica que el archivo sea público.")
        st.stop()

    # ── Aplicar estrategia sobre histórico (cacheado — no recalcula en cada rerun) ──
    bt, trades_df, metrics = get_strategy_data(df_master)

    # ── Señal de HOY ─────────────────────────────────────────
    # BB: basado en histórico CSV + precio VXX de hoy (yfinance)
    last_hist = bt.iloc[-1]
    last_date = bt.index[-1]

    # VXX de hoy desde yfinance
    vxx_today   = today_px.get('VXX', {}).get('close', last_hist['VXX_Close'])
    vxx_date    = today_px.get('VXX', {}).get('date', last_date.date())
    sma20_today = last_hist['BB_SMA20']
    bb_up_today = last_hist['BB_Upper']

    # BB signal de hoy (sin shift — es la señal que se ejecuta mañana)
    bb_pos_hist = int(last_hist['sig_bb'])  # posición BB al cierre de ayer
    if bb_pos_hist == 0 and vxx_today < sma20_today:
        bb_sig_today = 1
    elif bb_pos_hist == 1 and vxx_today > bb_up_today:
        bb_sig_today = 0
    else:
        bb_sig_today = bb_pos_hist

    # Contango de hoy: CBOE live (Tab 1) tiene prioridad
    # m1p y m2p vienen del scope global (scrapeado del CBOE)
    if m1p and m2p and m1p > 0:
        ct_today     = cpct(m1p, m2p)
        ct_source    = "CBOE live"
        m1_sym_today = df_vx['Symbol'].iloc[0] if not df_vx.empty else "M1"
        m2_sym_today = df_vx['Symbol'].iloc[1] if len(df_vx) > 1 else "M2"
    else:
        ct_today     = float(last_hist.get('Contango_pct', 0))
        ct_source    = "CSV histórico"
        m1_sym_today = str(last_hist.get('M1_Symbol', 'M1'))
        m2_sym_today = str(last_hist.get('M2_Symbol', 'M2'))

    in_ct_today   = ct_today is not None and ct_today > 0
    final_sig_today = int(bb_sig_today == 1 and in_ct_today)

    # Día de ejecución = mañana (siguiente día hábil)
    exec_date = datetime.now().date() + timedelta(days=1)
    while exec_date.weekday() >= 5:
        exec_date += timedelta(days=1)

    pct_to_sma = (vxx_today / sma20_today - 1) * 100 if sma20_today else 0
    pct_to_bb  = (vxx_today / bb_up_today  - 1) * 100 if bb_up_today else 0
    ct_str     = f"{ct_today:+.2f}%" if ct_today is not None else "N/A"
    vix_today  = today_px.get('VIX', {}).get('close', float(last_hist.get('VIX_Close', 0)))
    svxy_today = today_px.get('SVXY', {}).get('close', 0)
    svix_today = today_px.get('SVIX', {}).get('close', 0)
    spy_today  = today_px.get('SPY',  {}).get('close', 0)

    # Régimen VIX
    if vix_today < 15:   regime, r_clr = "BAJO — óptimo",      "var(--g)"
    elif vix_today < 20: regime, r_clr = "NORMAL — bueno",     "var(--g)"
    elif vix_today < 28: regime, r_clr = "ELEVADO — precaución","var(--y)"
    else:                regime, r_clr = "CRISIS — peligro",   "var(--r)"

    # ═══════════════════════════════════════════
    # SECCIÓN 1 — SEÑAL DE HOY
    # ═══════════════════════════════════════════
    sig_cls = "sig-long" if final_sig_today else "sig-cash"
    sig_txt = "LONG SVXY" if final_sig_today else "CASH"
    sig_clr = "var(--g)" if final_sig_today else "var(--r)"
    bb_ok   = "ok" if bb_sig_today == 1 else "no"
    bb_mark = "✓" if bb_sig_today == 1 else "✗"
    ct_ok   = "ok" if in_ct_today else "no"
    ct_mark = "✓" if in_ct_today else "✗"

    col_sig, col_bb, col_ct, col_veh = st.columns([1.3, 1.5, 1.5, 1.3])

    with col_sig:
        st.markdown(f"""
        <div class="sig-box {sig_cls}" style="height:100%">
            <div class="sl" style="color:{sig_clr};font-size:2.2rem">{sig_txt}</div>
            <div class="sd" style="margin-top:0.4rem">Ejecución mañana al OPEN</div>
            <div class="sd">{exec_date.strftime('%Y-%m-%d')} · {vxx_date}</div>
        </div>""", unsafe_allow_html=True)

    with col_bb:
        sma_clr = "var(--g)" if vxx_today < sma20_today else "var(--r)"
        bb_clr2 = "var(--g)" if vxx_today <= bb_up_today else "var(--r)"
        st.markdown(f"""
        <div class="icard">
            <div class="ic-title">📊 BB Timing — VXX</div>
            <div class="ic-row"><span class="ic-label">Señal BB</span>
                <span class="ic-val"><span class="{bb_ok}" style="font-size:1rem">{bb_mark}</span>
                {'LONG' if bb_sig_today else 'CASH'}</span></div>
            <div class="ic-row"><span class="ic-label">VXX</span>
                <span class="ic-val" style="font-weight:700">${vxx_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">SMA(20)</span>
                <span class="ic-val" style="color:{sma_clr}">${sma20_today:.2f}
                ({pct_to_sma:+.1f}%)</span></div>
            <div class="ic-row"><span class="ic-label">BB 2σ</span>
                <span class="ic-val" style="color:{bb_clr2}">${bb_up_today:.2f}
                ({pct_to_bb:+.1f}%)</span></div>
        </div>""", unsafe_allow_html=True)

    with col_ct:
        ct_clr   = "var(--g)" if in_ct_today else "var(--r)"
        ct_estado = "CONTANGO" if in_ct_today else "BACKWARDATION"
        m1_disp  = f"${m1p:.2f}" if m1p else "—"
        m2_disp  = f"${m2p:.2f}" if m2p else "—"
        st.markdown(f"""
        <div class="icard">
            <div class="ic-title">📈 Contango ({ct_source})</div>
            <div class="ic-row"><span class="ic-label">Señal CT</span>
                <span class="ic-val"><span class="{ct_ok}" style="font-size:1rem">{ct_mark}</span>
                <span style="color:{ct_clr};font-weight:700"> {ct_estado}</span></span></div>
            <div class="ic-row"><span class="ic-label">{m1_sym_today} (M1)</span>
                <span class="ic-val">{m1_disp}</span></div>
            <div class="ic-row"><span class="ic-label">{m2_sym_today} (M2)</span>
                <span class="ic-val">{m2_disp}</span></div>
            <div class="ic-row"><span class="ic-label">Contango %</span>
                <span class="ic-val" style="color:{ct_clr};font-weight:700">{ct_str}</span></div>
            <div class="ic-row"><span class="ic-label">VIX</span>
                <span class="ic-val" style="color:{r_clr}">{vix_today:.1f} · {regime}</span></div>
        </div>""", unsafe_allow_html=True)

    with col_veh:
        svxy_chg = ""
        if today_px.get('SVXY', {}).get('prev'):
            d = svxy_today - today_px['SVXY']['prev']
            svxy_chg = f" ({d:+.2f})"
        st.markdown(f"""
        <div class="icard">
            <div class="ic-title">💼 Vehículos</div>
            <div class="ic-row"><span class="ic-label">SVXY (-0.5x)</span>
                <span class="ic-val" style="color:var(--c);font-weight:700">${svxy_today:.2f}{svxy_chg}</span></div>
            <div class="ic-row"><span class="ic-label">SVIX (-1x)</span>
                <span class="ic-val" style="color:var(--c)">${svix_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">SPY</span>
                <span class="ic-val">${spy_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">VIX Spot</span>
                <span class="ic-val">{vix_today:.2f}</span></div>
        </div>""", unsafe_allow_html=True)

    # Alertas
    alerts = []
    if final_sig_today and pct_to_bb > -3:
        alerts.append(f"⚠️ VXX a {abs(pct_to_bb):.1f}% de la BB Superior — posible salida pronto")
    if ct_today is not None and 0 < ct_today < 1:
        alerts.append(f"⚠️ Contango muy bajo ({ct_today:.2f}%) — monitorear de cerca")
    if not final_sig_today and abs(pct_to_sma) < 2 and in_ct_today:
        alerts.append(f"🔔 Posible entrada pronto — VXX a {abs(pct_to_sma):.1f}% de SMA(20)")
    if not in_ct_today and bb_sig_today == 1:
        alerts.append("⚠️ BB dice LONG pero hay backwardation — CASH por Contango Rule")
    for a in alerts:
        st.warning(a)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # SECCIÓN 2 — MÉTRICAS RESUMEN + EQUITY CURVE
    # ═══════════════════════════════════════════
    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0 0.6rem'></div>",
                unsafe_allow_html=True)

    if metrics:
        m = metrics
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        def mcard(label, val, clr="nt"):
            return f'<div class="mpill"><div class="ml">{label}</div><div class="mv {clr}">{val}</div></div>'

        st.markdown(f"""<div class="mrow">
            {mcard("CAGR", f"{m['cagr']:+.1f}%", "up" if m['cagr']>0 else "dn")}
            {mcard("Max DD", f"{m['mdd']:.1f}%", "dn")}
            {mcard("Sharpe", f"{m['sharpe']:.2f}", "up" if m['sharpe']>1 else "nt")}
            {mcard("Calmar", f"{m['calmar']:.2f}", "up" if m['calmar']>1 else "nt")}
            {mcard("Win Rate", f"{m['wr']:.1f}%", "nt")}
            {mcard("Exposición", f"{m['exp']:.1f}%", "nt")}
        </div>""", unsafe_allow_html=True)

        col_eq, col_yr = st.columns([2, 1])
        with col_eq:
            fig_eq = build_equity_chart(m['equity'])
            st.plotly_chart(fig_eq, use_container_width=True,
                            config=dict(displayModeBar=False, displaylogo=False,
                                        scrollZoom=False))
        with col_yr:
            fig_yr = build_yearly_heatmap(m['yearly'])
            st.plotly_chart(fig_yr, use_container_width=True,
                            config=dict(displayModeBar=False, displaylogo=False,
                                        scrollZoom=False))

    # ═══════════════════════════════════════════
    # SECCIÓN 3 — GRÁFICAS OPERATIVAS
    # ═══════════════════════════════════════════
    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0 0.5rem'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-family:Inter;font-weight:700;font-size:0.9rem;"
                "color:#F0F6FC;margin-bottom:0.6rem'>📊 Gráficas Operativas</div>",
                unsafe_allow_html=True)

    # ── Gráfica 1: VXX + BB ────────────────────────────────────
    st.markdown("<div style='font-family:JetBrains Mono;font-size:0.72rem;"
                "color:#8B949E;margin-bottom:0.2rem'>SEÑAL DE TIMING · VXX vs BB(20, 2σ)</div>",
                unsafe_allow_html=True)
    fig_vxx = build_bb_chart(bt, window=len(bt))
    st.plotly_chart(fig_vxx, use_container_width=True,
                    config=dict(displayModeBar=True, displaylogo=False,
                                scrollZoom=False,
                                modeBarButtonsToRemove=['select2d','lasso2d']))

    # ── Gráfica 2: SVIX operativa ──────────────────────────────
    if 'SVIX_Close' in bt.columns and bt['SVIX_Close'].notna().sum() > 10:
        st.markdown("<div style='font-family:JetBrains Mono;font-size:0.72rem;"
                    "color:#8B949E;margin:0.6rem 0 0.2rem'>VEHÍCULO AGRESIVO · SVIX (-1x)</div>",
                    unsafe_allow_html=True)
        fig_svix = build_operational_chart(
            bt, col_price='SVIX_Close', label='SVIX', color='#E91E63',
            trades_df=trades_df,
            today_price=None, today_sig=0,   # solo histórico CSV
        )
        st.plotly_chart(fig_svix, use_container_width=True,
                        config=dict(displayModeBar=True, displaylogo=False,
                                    scrollZoom=False,
                                    modeBarButtonsToRemove=['select2d','lasso2d']))

    # ── Gráfica 3: SVXY operativa ──────────────────────────────
    st.markdown("<div style='font-family:JetBrains Mono;font-size:0.72rem;"
                "color:#8B949E;margin:0.6rem 0 0.2rem'>VEHÍCULO PRINCIPAL · SVXY (-0.5x)</div>",
                unsafe_allow_html=True)
    fig_svxy = build_operational_chart(
        bt, col_price='SVXY_Close', label='SVXY', color='#39D2C0',
        trades_df=trades_df,
        today_price=None, today_sig=0,   # solo histórico CSV
    )
    st.plotly_chart(fig_svxy, use_container_width=True,
                    config=dict(displayModeBar=True, displaylogo=False,
                                scrollZoom=False,
                                modeBarButtonsToRemove=['select2d','lasso2d']))

    # ═══════════════════════════════════════════
    # SECCIÓN 4 — HISTORIAL DE OPERACIONES
    # ═══════════════════════════════════════════
    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0 0.5rem'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-family:Inter;font-weight:700;font-size:0.9rem;"
                "color:#F0F6FC;margin-bottom:0.4rem'>📋 Historial de Operaciones</div>",
                unsafe_allow_html=True)

    if not trades_df.empty:
        n_total  = len(trades_df)
        n_closed = len(trades_df[trades_df['Salida'] != '🔴 ABIERTO'])
        n_win    = (trades_df[trades_df['Salida'] != '🔴 ABIERTO']['Retorno'] > 0).sum()
        avg_ret  = trades_df[trades_df['Salida'] != '🔴 ABIERTO']['Retorno'].mean()
        avg_dur  = trades_df[trades_df['Salida'] != '🔴 ABIERTO']['Días'].mean()

        st.markdown(f"""<div class="mrow">
            {mcard("Trades totales", str(n_total), "nt")}
            {mcard("Ganadores", f"{n_win}/{n_closed}", "up")}
            {mcard("Win Rate", f"{n_win/n_closed*100:.1f}%" if n_closed>0 else "—", "nt")}
            {mcard("Ret. promedio", f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",
                   "up" if not pd.isna(avg_ret) and avg_ret>0 else "dn")}
            {mcard("Duración media", f"{avg_dur:.0f}d" if not pd.isna(avg_dur) else "—", "nt")}
        </div>""", unsafe_allow_html=True)

        # Tabla HTML scrolleable
        rows_html = ""
        for _, t in trades_df.iloc[::-1].iterrows():
            ret     = t['Retorno']
            is_open = t['Salida'] == '🔴 ABIERTO'
            ret_clr = "color:var(--g)" if ret > 0 else "color:var(--r)"
            sal_clr = "color:var(--y);font-weight:700" if is_open else ""
            vix_e   = f"{t['VIX entrada']:.1f}" if pd.notna(t.get('VIX entrada')) else "—"
            ct_e    = f"{t['Contango entr.']:+.2f}%" if pd.notna(t.get('Contango entr.')) else "—"
            rows_html += f"""<tr>
                <td style="color:var(--b);font-weight:600">{t['Entrada']}</td>
                <td style="{sal_clr}">{t['Salida']}</td>
                <td>{t['Días']}</td>
                <td style="{ret_clr};font-weight:700">{ret:+.2f}%</td>
                <td>{t['Razón salida']}</td>
                <td>{vix_e}</td>
                <td>{ct_e}</td>
            </tr>"""

        st.markdown(f"""
        <div style="max-height:400px;overflow-y:auto;margin-top:0.4rem;
                    border:1px solid #30363D;border-radius:4px">
        <table class="dtbl">
            <thead style="position:sticky;top:0;background:#1C2128;z-index:1"><tr>
                <th>Entrada</th><th>Salida</th><th>Días</th>
                <th>Retorno</th><th>Razón salida</th>
                <th>VIX entrada</th><th>Contango entr.</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>""", unsafe_allow_html=True)

        st.caption(
            f"Histórico desde {bt.index[0].strftime('%Y-%m-%d')} · "
            f"CSV actualizado: {last_date.strftime('%Y-%m-%d')} · "
            f"Contango live: {ct_source}"
        )
    else:
        st.info("No se encontraron trades en el histórico.")


# ━━━━━━━━━━━━━━━━━ TAB 3: HELP ━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
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
