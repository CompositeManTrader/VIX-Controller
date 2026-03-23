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
import re, time, warnings, logging
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


@st.cache_data(ttl=55)
def fetch_bb_data():
    end = datetime.now()
    start = end - timedelta(days=300)
    syms = {"VXX":"VXX","SVXY":"SVXY","SVIX":"SVIX","VIX":"^VIX","SPY":"SPY"}
    data = pd.DataFrame()
    for name, sym in syms.items():
        try:
            df_t = yf.download(sym, start=start, end=end, progress=False)
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.get_level_values(0)
            if len(df_t) > 0:
                data[f"{name}_Close"] = df_t["Close"]
                data[f"{name}_Open"] = df_t["Open"]
        except: continue
    if data.empty: return None
    data = data.sort_index()
    vxx = data["VXX_Close"]
    data["SMA20"] = vxx.rolling(20).mean()
    data["STD20"] = vxx.rolling(20).std()
    data["BB_Upper"] = data["SMA20"] + 2.0 * data["STD20"]
    data["BB_Lower"] = data["SMA20"] - 2.0 * data["STD20"]
    clean = data.dropna(subset=["SMA20"]).copy()
    pos = 0
    bb_list = []
    for i in range(len(clean)):
        cp = clean["VXX_Close"].iloc[i]
        s = clean["SMA20"].iloc[i]
        u = clean["BB_Upper"].iloc[i]
        if pd.isna(s) or pd.isna(u) or pd.isna(cp):
            bb_list.append(pos); continue
        if pos == 0 and cp < s: pos = 1
        elif pos == 1 and cp > u: pos = 0
        bb_list.append(pos)
    clean["bb_sig"] = bb_list
    return clean


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


def build_bb_chart(clean, window=120):
    p = clean.tail(window).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Upper"], mode='lines', name='BB Upper',
                             line=dict(color='#F85149', width=1.2)))
    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Lower"], mode='lines', name='BB Lower',
                             line=dict(color='#F85149', width=0.5),
                             fill='tonexty', fillcolor='rgba(88,166,255,0.03)', showlegend=False))
    fig.add_trace(go.Scatter(x=p.index, y=p["SMA20"], mode='lines', name='SMA(20)',
                             line=dict(color='#58A6FF', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=p.index, y=p["VXX_Close"], mode='lines', name='VXX Close',
                             line=dict(color='#F0F6FC', width=2)))
    for i in range(1, len(p)):
        clr = 'rgba(63,185,80,0.06)' if p["bb_sig"].iloc[i] == 1 else 'rgba(248,81,73,0.03)'
        fig.add_vrect(x0=p.index[i-1], x1=p.index[i], fillcolor=clr, layer="below", line_width=0)
    for i in range(1, len(p)):
        if p["bb_sig"].iloc[i] == 1 and p["bb_sig"].iloc[i-1] == 0:
            fig.add_annotation(x=p.index[i], y=p["VXX_Close"].iloc[i],
                text="▲ ENTRY", showarrow=True, arrowhead=2, arrowcolor="#3FB950",
                font=dict(size=9, color="#3FB950", family="JetBrains Mono"), ax=0, ay=25)
        elif p["bb_sig"].iloc[i] == 0 and p["bb_sig"].iloc[i-1] == 1:
            fig.add_annotation(x=p.index[i], y=p["VXX_Close"].iloc[i],
                text="▼ EXIT", showarrow=True, arrowhead=2, arrowcolor="#F85149",
                font=dict(size=9, color="#F85149", family="JetBrains Mono"), ax=0, ay=-25)
    fig.add_trace(go.Scatter(x=[p.index[-1]], y=[p["VXX_Close"].iloc[-1]],
        mode='markers', name='Today',
        marker=dict(size=12, color='#D29922', line=dict(width=2, color='white')), showlegend=False))
    fig.update_layout(
        title=dict(text="<b>VXX + Bollinger Bands</b><sup>  (BB timing — contango from term structure)</sup>",
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=380, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title=dict(text="VXX", font=dict(size=11, color='#8B949E')),
                   gridcolor='#21262D', tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified')
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
    bb_data = fetch_bb_data()

    if bb_data is not None and len(bb_data) > 0:
        last = bb_data.iloc[-1]
        last_date = bb_data.index[-1]
        vxx_close = last["VXX_Close"]
        sma20 = last["SMA20"]
        bb_upper = last["BB_Upper"]
        vix_close = last.get("VIX_Close", vix_spot['price'] if vix_spot else 0)
        svxy_close = last.get("SVXY_Close", 0)
        svix_close = last.get("SVIX_Close", 0)
        spy_close = last.get("SPY_Close", 0)

        # Auto contango from term structure
        auto_m1 = m1p
        auto_m2 = m2p
        auto_m1_sym = df_vx['Symbol'].iloc[0] if not df_vx.empty else "?"
        auto_m2_sym = df_vx['Symbol'].iloc[1] if len(df_vx) > 1 else "?"
        contango_pct_val = cpct(auto_m1, auto_m2) if auto_m1 and auto_m2 else None
        in_contango = contango_pct_val is not None and contango_pct_val > 0

        bb_sig = int(bb_data["bb_sig"].iloc[-1])
        vxx_below_sma = vxx_close < sma20
        final_signal = bb_sig * int(in_contango) if contango_pct_val is not None else 0
        pct_to_sma = (vxx_close / sma20 - 1) * 100
        pct_to_bb = (vxx_close / bb_upper - 1) * 100
        bb_sig_prev = int(bb_data["bb_sig"].iloc[-2]) if len(bb_data) >= 2 else bb_sig
        bb_changed = bb_sig != bb_sig_prev
        ct_val = f"{contango_pct_val:+.2f}%" if contango_pct_val is not None else "N/A"

        c1, c2, c3 = st.columns([1.2, 1.5, 1.3])

        with c1:
            sig_cls = "sig-long" if final_signal else "sig-cash"
            sig_txt = "LONG" if final_signal else "CASH"
            sig_clr = "var(--g)" if final_signal else "var(--r)"
            st.markdown(f"""
            <div class="sig-box {sig_cls}">
                <div class="sl" style="color:{sig_clr}">{sig_txt}</div>
                <div class="sd">{last_date.strftime('%Y-%m-%d')}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            bb_ok = "ok" if bb_sig == 1 else "no"
            bb_mark = "✓ OK" if bb_sig == 1 else "✗ NO"
            ct_ok = "ok" if in_contango else "no"
            ct_mark = "✓ OK" if in_contango else "✗ NO"
            st.markdown(f"""
            <div class="chk"><span class="{bb_ok}">{bb_mark}</span> BB Timing — VXX &lt; SMA(20)</div>
            <div class="chk"><span class="{ct_ok}">{ct_mark}</span> Contango — M2 &gt; M1 ({ct_val})</div>
            """, unsafe_allow_html=True)
            if bb_changed:
                st.markdown('<div class="chk"><span style="color:var(--y)">⚠</span> BB cambió hoy</div>', unsafe_allow_html=True)

        with c2:
            sma_clr = "var(--g)" if vxx_below_sma else "var(--r)"
            sma_lbl = "DEBAJO" if vxx_below_sma else "ENCIMA"
            bb_clr = "var(--g)" if vxx_close <= bb_upper else "var(--r)"
            bb_lbl = "DEBAJO" if vxx_close <= bb_upper else "ENCIMA"
            bb_state = "LONG" if bb_sig else "CASH"
            bb_st_clr = "var(--g)" if bb_sig else "var(--r)"
            st.markdown(f"""
            <div class="icard">
                <div class="ic-title">VXX — Timing (Bollinger Band)</div>
                <div class="ic-row"><span class="ic-label">VXX Close</span><span class="ic-val" style="font-weight:700">${vxx_close:.2f}</span></div>
                <div class="ic-row"><span class="ic-label">SMA(20)</span><span class="ic-val" style="color:{sma_clr}">${sma20:.2f} ({pct_to_sma:+.1f}% {sma_lbl})</span></div>
                <div class="ic-row"><span class="ic-label">BB Superior</span><span class="ic-val" style="color:{bb_clr}">${bb_upper:.2f} ({pct_to_bb:+.1f}% {bb_lbl})</span></div>
                <div class="ic-row"><span class="ic-label">Distancia a BB</span><span class="ic-val">${bb_upper - vxx_close:.2f} ({abs(pct_to_bb):.1f}%)</span></div>
                <div class="ic-row"><span class="ic-label">BB Estado</span><span class="ic-val" style="color:{bb_st_clr};font-weight:700;font-size:1rem">{bb_state}</span></div>
            </div>""", unsafe_allow_html=True)

        with c3:
            ct_clr = "var(--g)" if in_contango else "var(--r)"
            ct_estado = "CONTANGO" if in_contango else "BACKWARDATION"
            if vix_close < 15: regime, r_clr = "BAJO (óptimo)", "var(--g)"
            elif vix_close < 20: regime, r_clr = "NORMAL (bueno)", "var(--g)"
            elif vix_close < 28: regime, r_clr = "ELEVADO (precaución)", "var(--y)"
            else: regime, r_clr = "CRISIS (peligro)", "var(--r)"
            m1_s = f"${auto_m1:.2f}" if auto_m1 else "N/A"
            m2_s = f"${auto_m2:.2f}" if auto_m2 else "N/A"
            st.markdown(f"""
            <div class="icard">
                <div class="ic-title">Contango + VIX</div>
                <div class="ic-row"><span class="ic-label">M1 ({auto_m1_sym})</span><span class="ic-val">{m1_s}</span></div>
                <div class="ic-row"><span class="ic-label">M2 ({auto_m2_sym})</span><span class="ic-val">{m2_s}</span></div>
                <div class="ic-row"><span class="ic-label">Contango</span><span class="ic-val" style="color:{ct_clr};font-weight:700">{ct_val} {ct_estado}</span></div>
                <div class="ic-row"><span class="ic-label">VIX</span><span class="ic-val" style="color:{r_clr}">{vix_close:.1f} {regime}</span></div>
            </div>
            <div class="icard">
                <div class="ic-title">Vehículos</div>
                <div class="ic-row"><span class="ic-label">SVXY (-0.5x)</span><span class="ic-val" style="color:var(--c);font-weight:700">${svxy_close:.2f}</span></div>
                <div class="ic-row"><span class="ic-label">SVIX (-1x agresivo)</span><span class="ic-val" style="color:var(--c);font-weight:700">${svix_close:.2f}</span></div>
                <div class="ic-row"><span class="ic-label">SPY</span><span class="ic-val">${spy_close:.2f}</span></div>
            </div>""", unsafe_allow_html=True)

        # Alerts
        alerts = []
        if final_signal and abs(pct_to_bb) < 3:
            alerts.append(f"⚠️ VXX cerca de BB Superior ({abs(pct_to_bb):.1f}%) — posible salida pronto")
        if contango_pct_val is not None and 0 < contango_pct_val < 1:
            alerts.append(f"⚠️ Contango muy bajo ({contango_pct_val:.1f}%) — monitorear")
        if not final_signal and abs(pct_to_sma) < 2 and in_contango:
            alerts.append(f"🔔 Posible entrada pronto — VXX a {abs(pct_to_sma):.1f}% de SMA")
        for a in alerts:
            st.warning(a)

        # BB Chart
        fig_bb = build_bb_chart(bb_data)
        st.plotly_chart(fig_bb, use_container_width=True, config=dict(displayModeBar=True, displaylogo=False))

        # Execution line
        exec_date = last_date + timedelta(days=1)
        while exec_date.weekday() >= 5:
            exec_date += timedelta(days=1)
        st.markdown(f"""
        <div class="icard" style="text-align:center">
            <span style="font-family:'JetBrains Mono';font-size:0.8rem;color:var(--dim)">
                SEÑAL: <b style="color:{'var(--g)' if final_signal else 'var(--r)'}">{'LONG' if final_signal else 'CASH'}</b>
                · BB: <b>{'LONG' if bb_sig else 'CASH'}</b> × Contango: <b>{'SÍ' if in_contango else 'NO'}</b>
                · Ejecución: <b>{exec_date.strftime('%Y-%m-%d')}</b> al OPEN
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.error("No se pudieron descargar datos de Yahoo Finance para el monitor operativo.")


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
