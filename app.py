"""
VIX Controller — Bloomberg-Style Term Structure + Operational Monitor
Data: CBOE CDN Settlement CSVs (monthly contracts only) + Yahoo Finance
Auto-refresh: every 60 seconds
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, date
import requests, io, time, warnings, json

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="VIX Controller", page_icon="🔴", layout="wide",
                   initial_sidebar_state="collapsed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOOMBERG CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0D1117;--card:#161B22;--border:#30363D;
  --g:#3FB950;--r:#F85149;--y:#D29922;--b:#58A6FF;--c:#39D2C0;
  --t:#C9D1D9;--dim:#8B949E;--w:#F0F6FC;
  --gbg:#0B2E13;--rbg:#3B1218;
}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 1.5rem;max-width:1400px;}

/* Header bar */
.hdr{display:flex;align-items:center;padding:0.5rem 0;border-bottom:2px solid #F7931A;margin-bottom:0.8rem;}
.hdr .logo{font-family:'Inter',sans-serif;font-weight:800;font-size:1.3rem;color:#F7931A;letter-spacing:1px;}
.hdr .sub{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-left:auto;}

/* Metric row */
.mrow{display:flex;gap:4px;margin-bottom:0.6rem;flex-wrap:wrap;}
.mpill{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:0.4rem 0.7rem;flex:1;min-width:120px;text-align:center;}
.mpill .ml{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.6px;}
.mpill .mv{font-family:'Inter',sans-serif;font-weight:700;font-size:1.15rem;}
.mv.up{color:var(--g);}.mv.dn{color:var(--r);}.mv.nt{color:var(--b);}

/* Contango table — VIXCentral style */
.ctx{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.78rem;margin:0.4rem 0;}
.ctx td,.ctx th{padding:0.35rem 0.5rem;text-align:center;border:1px solid var(--border);}
.ctx th{background:#1C2128;color:var(--dim);font-weight:500;font-size:0.65rem;text-transform:uppercase;}
.ctx .pos{color:var(--g);}.ctx .neg{color:var(--r);}
.ctx .hdr-cell{background:var(--card);color:var(--t);font-weight:600;text-align:left;width:120px;}

/* Data table */
.dtbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.75rem;margin-top:0.5rem;}
.dtbl th{color:var(--b);font-weight:500;padding:0.4rem 0.6rem;border-bottom:1px solid var(--border);font-size:0.62rem;text-transform:uppercase;letter-spacing:0.5px;text-align:center;}
.dtbl td{padding:0.35rem 0.6rem;text-align:center;color:var(--t);border-bottom:1px solid rgba(255,255,255,0.03);}
.dtbl tr:hover td{background:rgba(88,166,255,0.04);}

/* Signal box */
.sig-box{border-radius:6px;padding:1rem;text-align:center;border-width:2px;border-style:solid;}
.sig-long{background:var(--gbg);border-color:var(--g);}
.sig-cash{background:var(--rbg);border-color:var(--r);}
.sig-box .sl{font-family:'Inter',sans-serif;font-weight:800;font-size:2rem;}
.sig-box .sd{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-top:2px;}

/* Check items */
.chk{display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--t);}
.chk .ok{color:var(--g);font-weight:700;}.chk .no{color:var(--r);font-weight:700;}

/* Info card */
.icard{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;}
.icard .ic-title{font-family:'Inter',sans-serif;font-weight:700;font-size:0.85rem;color:var(--w);margin-bottom:0.5rem;border-bottom:1px solid var(--border);padding-bottom:0.3rem;}
.icard .ic-row{display:flex;justify-content:space-between;padding:0.2rem 0;font-family:'JetBrains Mono',monospace;font-size:0.8rem;}
.icard .ic-label{color:var(--dim);}.icard .ic-val{color:var(--t);font-weight:500;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-weight:600;font-size:0.82rem;color:var(--dim);padding:0.5rem 1.5rem;}
.stTabs [aria-selected="true"]{color:#F7931A !important;border-bottom:2px solid #F7931A !important;}
[data-testid="stSidebar"]{background:var(--card);}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MC = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
MN = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

def vix_exp(year, month):
    nm, ny = (month+1, year) if month < 12 else (1, year+1)
    d1 = date(ny, nm, 1)
    fri3 = d1 + timedelta(days=(4 - d1.weekday()) % 7 + 14)
    return fri3 - timedelta(days=30)

def monthly_contracts(ref=None, n=9):
    ref = ref or date.today()
    out = []
    m, y = ref.month, ref.year
    for i in range(n + 4):
        cm = ((m - 1 + i) % 12) + 1
        cy = y + ((m - 1 + i) // 12)
        exp = vix_exp(cy, cm)
        if exp >= ref:
            code = MC[cm]
            yr2 = str(cy)[-2:]
            out.append(dict(
                month=cm, year=cy, exp=exp, dte=(exp - ref).days,
                label=f"{MN[cm]}", code=f"M{len(out)+1}",
                symbol=f"VX/{code}{yr2}",
                cboe_sym=f"VX_{exp.strftime('%Y-%m-%d')}",
            ))
        if len(out) >= n:
            break
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
def fetch_cboe_contracts(contracts_json):
    """Fetch settlement prices from CBOE CDN individual contract CSVs."""
    contracts = json.loads(contracts_json)
    results = {}
    for c in contracts:
        url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/{c['cboe_sym']}.csv"
        try:
            r = requests.get(url, timeout=12,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.text) > 100:
                df = pd.read_csv(io.StringIO(r.text))
                if df.empty:
                    continue
                last = df.iloc[-1]
                price = None
                for col in ['Settle', 'Close', 'Last']:
                    if col in df.columns and pd.notna(last.get(col)) and float(last[col]) > 0:
                        price = round(float(last[col]), 2)
                        break
                if price:
                    prev = None
                    if len(df) > 1:
                        prev_row = df.iloc[-2]
                        for col in ['Settle', 'Close', 'Last']:
                            if col in df.columns and pd.notna(prev_row.get(col)) and float(prev_row[col]) > 0:
                                prev = round(float(prev_row[col]), 2)
                                break
                    # Get volume and OI from last row
                    vol = int(last.get('Total Volume', last.get('Volume', 0))) if 'Total Volume' in df.columns or 'Volume' in df.columns else 0
                    oi = int(last.get('Open Interest', 0)) if 'Open Interest' in df.columns else 0
                    hi = round(float(last.get('High', 0)), 2) if 'High' in df.columns and pd.notna(last.get('High')) else None
                    lo = round(float(last.get('Low', 0)), 2) if 'Low' in df.columns and pd.notna(last.get('Low')) else None
                    trade_date = str(last.get('Trade Date', ''))

                    results[c['symbol']] = dict(
                        price=price, prev=prev, vol=vol, oi=oi,
                        high=hi, low=lo, trade_date=trade_date, src='CBOE'
                    )
        except:
            continue
    return results

@st.cache_data(ttl=55)
def fetch_etps():
    """Fetch VXX, SVXY, SVIX, SPY from Yahoo."""
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
    """Fetch VXX data + compute Bollinger Bands for operational monitor."""
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
    if data.empty:
        return None
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
        p = clean["VXX_Close"].iloc[i]
        s = clean["SMA20"].iloc[i]
        u = clean["BB_Upper"].iloc[i]
        if pd.isna(s) or pd.isna(u) or pd.isna(p):
            bb_list.append(pos); continue
        if pos == 0 and p < s: pos = 1
        elif pos == 1 and p > u: pos = 0
        bb_list.append(pos)
    clean["bb_sig"] = bb_list
    return clean

def cpct(p1, p2):
    if p1 and p2 and p1 > 0:
        return round((p2 - p1) / p1 * 100, 2)
    return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART BUILDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_term_chart(vix_spot, contracts, fdata, show_prev=True):
    """VIXCentral-faithful term structure chart."""
    fig = go.Figure()

    xlbl, xpos, yt, yp = [], [], [], []
    for i, c in enumerate(contracts):
        xlbl.append(c['label'])
        xpos.append(i)
        s = c['symbol']
        yt.append(fdata[s]['price'] if s in fdata else None)
        yp.append(fdata[s].get('prev') if s in fdata else None)

    vx = [x for x, y in zip(xpos, yt) if y]; vy = [y for y in yt if y]

    # Today's curve — blue like VIXCentral
    if vy:
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode='lines+markers+text',
            name='Last', line=dict(color='#4A90D9', width=3, shape='spline'),
            marker=dict(size=9, color='#4A90D9', line=dict(width=2, color='#0D1117')),
            text=[f"{v:.3f}" if v else "" for v in vy],
            textposition='top center',
            textfont=dict(size=10, color='#C9D1D9', family='JetBrains Mono'),
            hovertemplate='%{text}<extra></extra>',
        ))

    # Previous day — gray dotted
    if show_prev:
        pvx = [x for x, y in zip(xpos, yp) if y]
        pvy = [y for y in yp if y]
        if len(pvy) >= 2:
            fig.add_trace(go.Scatter(
                x=pvx, y=pvy, mode='lines+markers',
                name='Previous Close', line=dict(color='#8B949E', width=1.5, dash='dot', shape='spline'),
                marker=dict(size=5, color='#8B949E', symbol='diamond'),
                hovertemplate='Prev: %{y:.3f}<extra></extra>',
            ))

    # VIX Index — green dashed line across full chart
    if vix_spot:
        fig.add_hline(
            y=vix_spot['price'], line_dash="dash", line_color="#3FB950", line_width=2,
            annotation_text=f"  {vix_spot['price']:.2f}",
            annotation_position="right",
            annotation_font=dict(size=12, color="#3FB950", family="Inter"),
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            name='VIX Index', line=dict(color='#3FB950', width=2, dash='dash'),
            showlegend=True,
        ))

    all_y = vy + ([vix_spot['price']] if vix_spot else [])
    y_min = min(all_y) - 1.5 if all_y else 15
    y_max = max(all_y) + 1.5 if all_y else 30

    fig.update_layout(
        title=dict(
            text="<b>VIX Futures Term Structure</b><br><sup>Source: CBOE Delayed Quotes · vixcontroller</sup>",
            font=dict(size=15, color='#C9D1D9', family='Inter'), x=0.5,
        ),
        template='plotly_dark',
        paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=420, margin=dict(l=50, r=30, t=65, b=50),
        xaxis=dict(
            tickvals=xpos, ticktext=xlbl,
            tickfont=dict(size=11, color='#8B949E', family='JetBrains Mono'),
            gridcolor='#21262D', showline=True, linecolor='#30363D',
            title=dict(text="Future Month", font=dict(size=11, color='#8B949E', family='Inter')),
        ),
        yaxis=dict(
            range=[y_min, y_max],
            title=dict(text="Volatility", font=dict(size=11, color='#8B949E', family='Inter')),
            tickfont=dict(size=11, color='#8B949E', family='JetBrains Mono'),
            gridcolor='#21262D', showline=True, linecolor='#30363D',
        ),
        legend=dict(
            orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor='rgba(22,27,34,0.9)', bordercolor='#30363D', borderwidth=1,
            font=dict(size=10, color='#C9D1D9', family='JetBrains Mono'),
        ),
        hoverlabel=dict(bgcolor='#1C2128', bordercolor='#58A6FF',
                        font=dict(size=11, family='JetBrains Mono', color='#C9D1D9')),
        hovermode='x unified',
    )
    return fig


def build_bb_chart(clean, window=120):
    """VXX + Bollinger Bands chart for operational monitor."""
    p = clean.tail(window).copy()
    fig = go.Figure()

    # BB bands fill
    fig.add_trace(go.Scatter(
        x=p.index, y=p["BB_Upper"], mode='lines', name='BB Upper',
        line=dict(color='#F85149', width=1.2),
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=p.index, y=p["BB_Lower"], mode='lines', name='BB Lower',
        line=dict(color='#F85149', width=0.5),
        fill='tonexty', fillcolor='rgba(88,166,255,0.03)',
        showlegend=False,
    ))
    # SMA
    fig.add_trace(go.Scatter(
        x=p.index, y=p["SMA20"], mode='lines', name='SMA(20)',
        line=dict(color='#58A6FF', width=1.5, dash='dash'),
    ))
    # VXX
    fig.add_trace(go.Scatter(
        x=p.index, y=p["VXX_Close"], mode='lines', name='VXX Close',
        line=dict(color='#F0F6FC', width=2),
    ))

    # Color background by BB signal
    for i in range(1, len(p)):
        clr = 'rgba(63,185,80,0.06)' if p["bb_sig"].iloc[i] == 1 else 'rgba(248,81,73,0.03)'
        fig.add_vrect(x0=p.index[i-1], x1=p.index[i], fillcolor=clr, layer="below", line_width=0)

    # Entry/Exit arrows
    for i in range(1, len(p)):
        if p["bb_sig"].iloc[i] == 1 and p["bb_sig"].iloc[i-1] == 0:
            fig.add_annotation(x=p.index[i], y=p["VXX_Close"].iloc[i],
                text="▲ ENTRY", showarrow=True, arrowhead=2, arrowcolor="#3FB950",
                font=dict(size=9, color="#3FB950", family="JetBrains Mono"),
                ax=0, ay=25)
        elif p["bb_sig"].iloc[i] == 0 and p["bb_sig"].iloc[i-1] == 1:
            fig.add_annotation(x=p.index[i], y=p["VXX_Close"].iloc[i],
                text="▼ EXIT", showarrow=True, arrowhead=2, arrowcolor="#F85149",
                font=dict(size=9, color="#F85149", family="JetBrains Mono"),
                ax=0, ay=-25)

    # Today marker
    fig.add_trace(go.Scatter(
        x=[p.index[-1]], y=[p["VXX_Close"].iloc[-1]],
        mode='markers', name='Today',
        marker=dict(size=12, color='#D29922', line=dict(width=2, color='white')),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="<b>VXX + Bollinger Bands</b><sup>  (BB timing — contango from term structure)</sup>",
                   font=dict(size=13, color='#C9D1D9', family='Inter'), x=0.5),
        template='plotly_dark',
        paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=380, margin=dict(l=50, r=30, t=55, b=40),
        xaxis=dict(gridcolor='#21262D', tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        yaxis=dict(title=dict(text="VXX", font=dict(size=11, color='#8B949E')),
                   gridcolor='#21262D', tickfont=dict(size=10, color='#8B949E', family='JetBrains Mono')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9, color='#8B949E', family='JetBrains Mono')),
        hovermode='x unified',
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-REFRESH (every 60s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
if elapsed > 60:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
next_refresh = 60 - int(elapsed)
st.markdown(f"""
<div class="hdr">
    <div class="logo">VIX CONTROLLER</div>
    <div class="sub">
        {now_str} &nbsp;·&nbsp; Auto-refresh in {next_refresh}s &nbsp;·&nbsp; Source: CBOE Settlement + Yahoo Finance
    </div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    N_MONTHS = st.slider("Futures months", 4, 12, 8)
    SHOW_PREV = st.checkbox("Show previous day", True)
    SHOW_TABLE = st.checkbox("Show data table", True)
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FETCH DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
contracts = monthly_contracts(n=N_MONTHS)
vix_spot = fetch_vix_spot()
fdata = fetch_cboe_contracts(json.dumps(contracts, default=str))
etps = fetch_etps()

# Collect prices in order
prices = []  # list of (label, price, prev_price)
for c in contracts:
    s = c['symbol']
    if s in fdata:
        prices.append((c['code'], c['label'], fdata[s]['price'], fdata[s].get('prev')))

m1p = prices[0][2] if prices else None
m2p = prices[1][2] if len(prices) > 1 else None
front_ct = cpct(m1p, m2p)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab1, tab2, tab3 = st.tabs(["📈  Term Structure", "🎯  Monitor Operativo", "ℹ️  Help"])

# ━━━━━━━━━━━━━━━━━ TAB 1: TERM STRUCTURE ━━━━━━━━━━━━━━━━━━
with tab1:
    # Metric strip
    def fv(v):
        return f"{v:.2f}" if v is not None else "—"
    def vc(v):
        if v is None: return "nt"
        return "up" if v >= 0 else "dn"
    def fp(v):
        if v is None: return "—"
        return f"{'+' if v >= 0 else ''}{v:.2f}%"

    vix_p = vix_spot['price'] if vix_spot else None
    total_ct = cpct(vix_p, prices[-1][2]) if vix_p and prices else None
    spot_m1 = cpct(vix_p, m1p)

    m1_lbl = prices[0][1] if prices else "—"
    m2_lbl = prices[1][1] if len(prices) > 1 else "—"
    m1_dte = contracts[0]['dte'] if contracts else "?"

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
    fig = build_term_chart(vix_spot, contracts, fdata, show_prev=SHOW_PREV)
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=True, displaylogo=False))

    # ── Contango & Difference table (VIXCentral style) ──
    if len(prices) >= 2:
        ct_cells = ""
        diff_cells = ""
        for i in range(len(prices) - 1):
            n = i + 1
            p1, p2 = prices[i][2], prices[i+1][2]
            ct = cpct(p1, p2)
            diff = round(p2 - p1, 2) if p1 and p2 else None
            ct_cls = "pos" if ct and ct >= 0 else "neg"
            diff_cls = "pos" if diff and diff >= 0 else "neg"
            ct_cells += f'<td>{n}</td><td class="{ct_cls}">{fp(ct)}</td>'
            diff_cells += f'<td>{n}</td><td class="{diff_cls}">{fv(diff)}</td>'

        # Month 7 to 4 contango
        m74_ct, m74_diff = None, None
        if len(prices) >= 7:
            p4, p7 = prices[3][2], prices[6][2]
            m74_ct = cpct(p4, p7)
            m74_diff = round(p7 - p4, 2) if p4 and p7 else None

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
            <td class="{m74_cls}">{fp(m74_ct)}</td>
            <td class="{m74_cls}">{fv(m74_diff)}</td></tr>
            </table>
            """, unsafe_allow_html=True)

    # ── Data table ──
    if SHOW_TABLE and prices:
        rows = ""
        prev_p = vix_p
        for c in contracts:
            s = c['symbol']
            if s in fdata:
                d = fdata[s]
                p = d['price']
                chg = round(p - d['prev'], 2) if d.get('prev') else None
                ct = cpct(prev_p, p)
                chg_c = "color:var(--g)" if chg and chg >= 0 else "color:var(--r)" if chg else ""
                ct_c = "color:var(--g)" if ct and ct >= 0 else "color:var(--r)" if ct else ""
                chg_s = f"{chg:+.3f}" if chg is not None else "—"
                ct_s = fp(ct) if ct is not None else "—"
                hi_s = f"{d['high']:.2f}" if d.get('high') and d['high'] > 0 else "—"
                lo_s = f"{d['low']:.2f}" if d.get('low') and d['low'] > 0 else "—"
                rows += f"""<tr>
                    <td style="color:var(--b);font-weight:600">{c['symbol']}</td>
                    <td>{c['exp'].strftime('%m/%d/%Y')}</td>
                    <td style="font-weight:600">{p:.4f}</td>
                    <td style="{chg_c}">{chg_s}</td>
                    <td>{hi_s}</td><td>{lo_s}</td>
                    <td style="{ct_c}">{ct_s}</td>
                    <td>{c['dte']}</td>
                    <td>{d.get('vol',0):,}</td>
                </tr>"""
                prev_p = p

        st.markdown(f"""
        <table class="dtbl">
            <thead><tr><th>Symbol</th><th>Expiration</th><th>Settlement</th><th>Change</th>
            <th>High</th><th>Low</th><th>Contango</th><th>DTE</th><th>Volume</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    if not prices:
        st.warning("⚠️ No se pudieron obtener precios de futuros VIX del CBOE CDN. Esto puede ocurrir fuera de horario de mercado o si el CDN está temporalmente inaccesible.")
        st.info("💡 Verifica que tengas acceso a `cdn.cboe.com` desde tu red. Los datos se actualizan después de cada sesión de trading.")

    found = len(prices)
    st.caption(f"Contratos cargados: {found}/{N_MONTHS} · Solo mensuales (sin weeklys) · Fuente: CBOE CDN Settlement CSVs")


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
        svxy_open = last.get("SVXY_Open", 0)
        svix_close = last.get("SVIX_Close", 0)
        svix_open = last.get("SVIX_Open", 0)
        spy_close = last.get("SPY_Close", 0)

        # Auto M1/M2 from term structure
        auto_m1 = m1p
        auto_m2 = m2p
        auto_m1_sym = contracts[0]['symbol'] if contracts else "?"
        auto_m2_sym = contracts[1]['symbol'] if len(contracts) > 1 else "?"
        contango_pct = cpct(auto_m1, auto_m2) if auto_m1 and auto_m2 else None
        in_contango = contango_pct is not None and contango_pct > 0

        # BB signal
        bb_sig = int(bb_data["bb_sig"].iloc[-1])
        vxx_below_sma = vxx_close < sma20
        vxx_above_bb = vxx_close > bb_upper

        # FINAL SIGNAL
        final_signal = bb_sig * int(in_contango) if contango_pct is not None else 0

        pct_to_sma = (vxx_close / sma20 - 1) * 100
        pct_to_bb = (vxx_close / bb_upper - 1) * 100

        # Previous BB
        bb_sig_prev = int(bb_data["bb_sig"].iloc[-2]) if len(bb_data) >= 2 else bb_sig
        bb_changed = bb_sig != bb_sig_prev

        # ── Layout ──
        c1, c2, c3 = st.columns([1.2, 1.5, 1.3])

        # Signal box
        with c1:
            sig_cls = "sig-long" if final_signal else "sig-cash"
            sig_txt = "LONG" if final_signal else "CASH"
            sig_clr = "var(--g)" if final_signal else "var(--r)"
            st.markdown(f"""
            <div class="sig-box {sig_cls}">
                <div class="sl" style="color:{sig_clr}">{sig_txt}</div>
                <div class="sd">{last_date.strftime('%Y-%m-%d')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # Checks
            bb_ok = "ok" if bb_sig == 1 else "no"
            bb_mark = "✓ OK" if bb_sig == 1 else "✗ NO"
            ct_ok = "ok" if in_contango else "no"
            ct_mark = "✓ OK" if in_contango else "✗ NO"
            ct_val = f"{contango_pct:+.2f}%" if contango_pct is not None else "N/A"

            st.markdown(f"""
            <div class="chk"><span class="{bb_ok}">{bb_mark}</span> BB Timing — VXX &lt; SMA(20)</div>
            <div class="chk"><span class="{ct_ok}">{ct_mark}</span> Contango — M2 &gt; M1 ({ct_val})</div>
            """, unsafe_allow_html=True)

            if bb_changed:
                st.markdown('<div class="chk"><span style="color:var(--y)">⚠</span> BB cambió hoy</div>', unsafe_allow_html=True)

        # VXX Detail
        with c2:
            sma_clr = "var(--g)" if vxx_below_sma else "var(--r)"
            sma_lbl = "DEBAJO" if vxx_below_sma else "ENCIMA"
            bb_clr = "var(--g)" if not vxx_above_bb else "var(--r)"
            bb_lbl = "DEBAJO" if not vxx_above_bb else "ENCIMA"
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
            </div>
            """, unsafe_allow_html=True)

        # Contango + VIX + Vehicles
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
            </div>
            """, unsafe_allow_html=True)

        # ── Alerts ──
        alerts = []
        if final_signal and abs(pct_to_bb) < 3:
            alerts.append(f"⚠️ VXX cerca de BB Superior ({abs(pct_to_bb):.1f}%) — posible salida pronto")
        if contango_pct is not None and 0 < contango_pct < 1:
            alerts.append(f"⚠️ Contango muy bajo ({contango_pct:.1f}%) — monitorear")
        if not final_signal and abs(pct_to_sma) < 2 and in_contango:
            alerts.append(f"🔔 Posible entrada pronto — VXX a {abs(pct_to_sma):.1f}% de SMA")
        if vix_close >= 28:
            alerts.append(f"🚨 VIX en zona de crisis ({vix_close:.1f}) — máxima precaución")

        if alerts:
            for a in alerts:
                st.warning(a)

        # ── BB Chart ──
        fig_bb = build_bb_chart(bb_data)
        st.plotly_chart(fig_bb, use_container_width=True, config=dict(displayModeBar=True, displaylogo=False))

        # ── Execution note ──
        exec_date = last_date + timedelta(days=1)
        while exec_date.weekday() >= 5:
            exec_date += timedelta(days=1)

        sig_str = "LONG" if final_signal else "CASH"
        st.markdown(f"""
        <div class="icard" style="text-align:center">
            <span style="font-family:'JetBrains Mono';font-size:0.8rem;color:var(--dim)">
                SEÑAL: <b style="color:{'var(--g)' if final_signal else 'var(--r)'}">{sig_str}</b>
                &nbsp;·&nbsp; BB: <b>{'LONG' if bb_sig else 'CASH'}</b>
                × Contango: <b>{'SÍ' if in_contango else 'NO'}</b>
                &nbsp;·&nbsp; Ejecución (si cambio): <b>{exec_date.strftime('%Y-%m-%d')}</b> al OPEN
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("No se pudieron descargar datos de Yahoo Finance para el monitor operativo.")


# ━━━━━━━━━━━━━━━━━ TAB 3: HELP ━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("""
    ### VIX Controller — Guía

    **Tab 1: Term Structure** — Réplica de VIXCentral.com
    - Datos de CBOE CDN (settlement CSVs por contrato individual)
    - Solo contratos mensuales (sin weeklys)
    - Tabla de contango/diferencia entre meses consecutivos
    - Month 7 to 4 contango (indicador de término medio)
    - Auto-refresh cada 60 segundos

    **Tab 2: Monitor Operativo** — Señal BB × Contango
    - **Bollinger Band Timing**: VXX < SMA(20) = LONG, VXX > BB Superior = EXIT
    - **Filtro Contango**: M2 > M1 (automático desde term structure)
    - **Señal Final** = BB × Contango — ambos deben ser TRUE para LONG
    - Gráfico VXX + BB con zonas de señal coloreadas
    - Alertas automáticas de proximidad a niveles clave

    ---

    **Fuentes de datos:**
    - `cdn.cboe.com` — Settlement prices por contrato VX
    - Yahoo Finance — VIX spot, VXX, SVXY, SVIX, SPY

    **Instrumentos:**

    | Instrumento | Exposición | Descripción |
    |------------|-----------|-------------|
    | SVXY | -0.5x | ProShares Short VIX Short-Term Futures |
    | SVIX | -1x | -1x Short VIX Futures ETF |
    | VXX | +1x | iPath Series B VIX Short-Term Futures ETN |

    **Expiración VIX Futures:** Miércoles 30 días antes del 3er viernes del mes siguiente.
    """)

# ── Footer ──
st.markdown(f"""
<div style="text-align:center;padding:0.8rem 0 0.3rem;border-top:1px solid #30363D;margin-top:1rem;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#484F58;">
        VIX CONTROLLER · Alberto Alarcón González · Estrategia Volatilidad Inversa · Not financial advice
    </span>
</div>
""", unsafe_allow_html=True)
