"""
VIX Central — Term Structure Dashboard
Faithful replica of vixcentral.com
Data: CBOE (primary) + Yahoo Finance (fallback)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta, date
import requests
import io
import warnings
import json
import time

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="VIX Central — Term Structure",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — VIXCentral-faithful dark theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0b0e14;
        --bg-card: #111620;
        --bg-surface: #161c28;
        --border: rgba(56,189,248,0.12);
        --accent: #38bdf8;
        --accent-dim: rgba(56,189,248,0.6);
        --green: #22c55e;
        --red: #ef4444;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-dim: #64748b;
    }

    .stApp { background: var(--bg-primary); }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; max-width: 1200px; }

    /* Master header */
    .vix-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 0 0.75rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.25rem;
    }
    .vix-header .logo {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--accent);
        letter-spacing: -0.5px;
    }
    .vix-header .logo span { color: var(--text-secondary); font-weight: 400; }
    .vix-header .sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-dim);
        margin-left: auto;
    }

    /* Metric strip */
    .metric-strip {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.55rem 1rem;
        flex: 1;
        min-width: 140px;
        text-align: center;
    }
    .metric-pill .mp-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
    }
    .metric-pill .mp-value {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .mp-value.up { color: var(--green); }
    .mp-value.down { color: var(--red); }
    .mp-value.flat { color: var(--accent); }

    /* Contango strip */
    .contango-strip {
        display: flex;
        gap: 3px;
        margin: 0.75rem 0 0.5rem 0;
    }
    .contango-cell {
        flex: 1;
        text-align: center;
        padding: 0.4rem 0.25rem;
        border-radius: 5px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .contango-cell .cc-label {
        font-size: 0.58rem;
        opacity: 0.7;
        margin-bottom: 1px;
    }
    .contango-cell .cc-value {
        font-weight: 600;
        font-size: 0.82rem;
    }
    .contango-cell.pos { background: rgba(34,197,94,0.12); color: var(--green); }
    .contango-cell.neg { background: rgba(239,68,68,0.12); color: var(--red); }

    /* Data table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        margin-top: 0.5rem;
    }
    .data-table th {
        color: var(--accent);
        font-weight: 500;
        padding: 0.5rem 0.75rem;
        text-align: center;
        border-bottom: 1px solid var(--border);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .data-table td {
        padding: 0.45rem 0.75rem;
        text-align: center;
        color: var(--text-primary);
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .data-table tr:hover td { background: rgba(56,189,248,0.04); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.6rem 1.5rem;
    }

    /* Sidebar overrides */
    [data-testid="stSidebar"] { background: var(--bg-card); }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONTH_CODES = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
MONTH_NAMES_SHORT = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}


def vix_futures_expiration(year: int, month: int) -> date:
    """Wednesday 30 days before the 3rd Friday of the NEXT month."""
    nm = month + 1
    ny = year
    if nm > 12:
        nm, ny = 1, year + 1
    first = date(ny, nm, 1)
    dow = first.weekday()  # Mon=0
    days_to_fri = (4 - dow) % 7
    third_fri = first + timedelta(days=days_to_fri + 14)
    return third_fri - timedelta(days=30)


def active_contracts(ref: date = None, n: int = 9):
    """Return list of dicts for the next n active VIX futures."""
    if ref is None:
        ref = date.today()
    out = []
    m, y = ref.month, ref.year
    for i in range(n + 4):
        cm = ((m - 1 + i) % 12) + 1
        cy = y + ((m - 1 + i) // 12)
        exp = vix_futures_expiration(cy, cm)
        if exp >= ref:
            code = MONTH_CODES[cm]
            out.append({
                'month': cm, 'year': cy,
                'exp': exp,
                'dte': (exp - ref).days,
                'label': f"{MONTH_NAMES_SHORT[cm]} {cy}",
                'code': f"M{len(out)+1}",
                'symbol': f"VX{code}{str(cy)[-2:]}",
            })
        if len(out) >= n:
            break
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA FETCHING — CBOE + Yahoo Finance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=120)
def fetch_vix_spot():
    """Get current VIX spot from Yahoo Finance."""
    try:
        vix = yf.Ticker("^VIX")
        h = vix.history(period="5d")
        if not h.empty:
            cur = round(float(h['Close'].iloc[-1]), 2)
            prev = round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else cur
            return {'price': cur, 'prev': prev, 'change': round(cur - prev, 2),
                    'pct': round((cur - prev) / prev * 100, 2) if prev else 0}
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def fetch_cboe_settlement():
    """
    Download latest CBOE VIX futures settlement prices.
    Tries the CBOE CFE daily settlement CSV.
    """
    # Try the CBOE volume/OI master file which includes settle prices
    url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/cfevoloi.csv"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            # Filter VX products only (not VX weekly)
            if 'Product' in df.columns:
                df = df[df['Product'].str.strip() == 'VX'].copy()
            elif 'Symbol' in df.columns:
                df = df[df['Symbol'].str.contains('VX', na=False)].copy()
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def fetch_cboe_individual_contracts(contracts):
    """
    Download individual VIX futures contract CSVs from CBOE CDN.
    URL pattern: https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{YYYY-MM-DD}.csv
    """
    results = {}
    for c in contracts:
        exp_str = c['exp'].strftime('%Y-%m-%d')
        url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{exp_str}.csv"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                if not df.empty:
                    # Get last row (most recent settle)
                    last = df.iloc[-1]
                    settle = None
                    for col in ['Settle', 'Close', 'Last']:
                        if col in df.columns:
                            val = last[col]
                            if pd.notna(val) and float(val) > 0:
                                settle = round(float(val), 2)
                                break
                    if settle:
                        # Get previous day
                        prev_settle = None
                        if len(df) > 1:
                            prev = df.iloc[-2]
                            for col in ['Settle', 'Close', 'Last']:
                                if col in df.columns:
                                    val = prev[col]
                                    if pd.notna(val) and float(val) > 0:
                                        prev_settle = round(float(val), 2)
                                        break
                        results[c['symbol']] = {
                            'price': settle,
                            'prev': prev_settle,
                            'source': 'CBOE'
                        }
        except Exception:
            continue
    return results


@st.cache_data(ttl=120)
def fetch_yahoo_vix_futures(contracts):
    """
    Fallback: try Yahoo Finance continuous month tickers.
    ^VIX for spot, VX=F for front month, etc.
    """
    results = {}

    # Try Yahoo continuous futures symbols
    yahoo_symbols = []
    for i, c in enumerate(contracts):
        # Yahoo sometimes has VXM26.CBF, VX=F, etc.
        m_code = MONTH_CODES[c['month']]
        yr = str(c['year'])[-2:]
        candidates = [
            f"VX{m_code}{yr}.CBF",
            f"VX{m_code}{yr}.CBE",
        ]
        yahoo_symbols.append((c, candidates))

    for c, candidates in yahoo_symbols:
        for sym in candidates:
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d")
                if not h.empty and float(h['Close'].iloc[-1]) > 0:
                    results[c['symbol']] = {
                        'price': round(float(h['Close'].iloc[-1]), 2),
                        'prev': round(float(h['Close'].iloc[-2]), 2) if len(h) > 1 else None,
                        'source': 'Yahoo'
                    }
                    break
            except Exception:
                continue

    return results


def get_futures_data(contracts):
    """
    Multi-source fetch: CBOE first, Yahoo fallback.
    Returns dict keyed by contract symbol.
    """
    # 1. Try CBOE individual contract CSVs
    data = fetch_cboe_individual_contracts(contracts)

    # 2. If we didn't get enough, try Yahoo
    missing = [c for c in contracts if c['symbol'] not in data]
    if missing:
        yahoo_data = fetch_yahoo_vix_futures(missing)
        data.update(yahoo_data)

    return data


@st.cache_data(ttl=600)
def fetch_historical_structure(target: date, n: int = 9):
    """Fetch term structure for a historical date."""
    contracts = active_contracts(ref=target, n=n)

    # VIX spot
    vix_spot = None
    try:
        vix = yf.Ticker("^VIX")
        start = target - timedelta(days=5)
        end = target + timedelta(days=1)
        h = vix.history(start=start, end=end)
        if not h.empty:
            idx = h.index.get_indexer([pd.Timestamp(target)], method='pad')
            if idx[0] >= 0:
                vix_spot = round(float(h['Close'].iloc[idx[0]]), 2)
    except Exception:
        pass

    # Futures from CBOE
    futures = []
    for c in contracts:
        exp_str = c['exp'].strftime('%Y-%m-%d')
        url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{exp_str}.csv"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                if not df.empty and 'Trade Date' in df.columns:
                    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
                    mask = df['Trade Date'] <= pd.Timestamp(target)
                    if mask.any():
                        row = df[mask].iloc[-1]
                        for col in ['Settle', 'Close', 'Last']:
                            if col in df.columns and pd.notna(row[col]) and float(row[col]) > 0:
                                futures.append({
                                    'label': c['label'],
                                    'code': c['code'],
                                    'price': round(float(row[col]), 2),
                                })
                                break
        except Exception:
            continue

    return {'date': target, 'vix_spot': vix_spot, 'futures': futures}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART BUILDER — VIXCentral style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_term_chart(vix_spot, contracts, fdata, show_prev=True, overlays=None, num_months=9):
    """Build the main term structure chart matching VIXCentral's style."""

    fig = go.Figure()

    # ── Collect today's curve data ──
    x_labels = []
    x_idx = []
    y_today = []
    y_prev = []

    # VIX Spot as point 0
    if vix_spot:
        x_labels.append("VIX")
        x_idx.append(0)
        y_today.append(vix_spot['price'])
        y_prev.append(vix_spot.get('prev'))

    for i, c in enumerate(contracts[:num_months]):
        x_labels.append(c['code'])
        x_idx.append(i + 1)
        sym = c['symbol']
        if sym in fdata:
            y_today.append(fdata[sym]['price'])
            y_prev.append(fdata[sym].get('prev'))
        else:
            y_today.append(None)
            y_prev.append(None)

    # Filter valid points
    vx = [x for x, y in zip(x_idx, y_today) if y is not None]
    vy = [y for y in y_today if y is not None]

    # ── Today's curve ──
    if vy:
        fig.add_trace(go.Scatter(
            x=vx, y=vy,
            mode='lines+markers+text',
            name=date.today().strftime('%b %d, %Y'),
            line=dict(color='#38bdf8', width=3, shape='spline'),
            marker=dict(size=10, color='#38bdf8',
                        line=dict(width=2.5, color='#0b0e14')),
            text=[f"{v:.2f}" for v in vy],
            textposition='top center',
            textfont=dict(size=11, color='#38bdf8', family='IBM Plex Mono'),
            hovertemplate='%{text}<extra></extra>',
        ))

    # ── Previous day curve ──
    if show_prev:
        pvx = [x for x, y in zip(x_idx, y_prev) if y is not None]
        pvy = [y for y in y_prev if y is not None]
        if pvy and len(pvy) >= 2:
            fig.add_trace(go.Scatter(
                x=pvx, y=pvy,
                mode='lines+markers',
                name='Previous Day',
                line=dict(color='#f97316', width=2, dash='dot', shape='spline'),
                marker=dict(size=6, color='#f97316',
                            line=dict(width=1, color='#0b0e14')),
                text=[f"{v:.2f}" for v in pvy],
                hovertemplate='Prev: %{text}<extra></extra>',
            ))

    # ── Historical overlays ──
    overlay_colors = [
        '#22c55e', '#ef4444', '#eab308', '#a855f7', '#ec4899',
        '#06b6d4', '#f97316', '#84cc16', '#e879f9', '#14b8a6',
    ]
    if overlays:
        for idx, ov in enumerate(overlays):
            col = overlay_colors[idx % len(overlay_colors)]
            ox, oy = [], []
            if ov.get('vix_spot'):
                ox.append(0)
                oy.append(ov['vix_spot'])
            for j, f in enumerate(ov.get('futures', [])):
                ox.append(j + 1)
                oy.append(f['price'])
            if oy:
                fig.add_trace(go.Scatter(
                    x=ox, y=oy,
                    mode='lines+markers',
                    name=str(ov['date']),
                    line=dict(color=col, width=2, shape='spline'),
                    marker=dict(size=6, color=col),
                    hovertemplate=f"{ov['date']}: " + '%{y:.2f}<extra></extra>',
                ))

    # ── Layout — VIXCentral-faithful ──
    # Compute y range
    all_y = vy + (pvy if show_prev else [])
    if overlays:
        for ov in overlays:
            if ov.get('vix_spot'):
                all_y.append(ov['vix_spot'])
            all_y += [f['price'] for f in ov.get('futures', [])]

    y_min = min(all_y) - 1 if all_y else 10
    y_max = max(all_y) + 2 if all_y else 30

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0b0e14',
        plot_bgcolor='#0f1319',
        height=480,
        margin=dict(l=55, r=25, t=30, b=55),
        xaxis=dict(
            tickvals=x_idx,
            ticktext=x_labels,
            tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono'),
            gridcolor='rgba(148,163,184,0.06)',
            zeroline=False,
            showline=True,
            linecolor='rgba(148,163,184,0.15)',
            linewidth=1,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono'),
            gridcolor='rgba(148,163,184,0.06)',
            zeroline=False,
            showline=True,
            linecolor='rgba(148,163,184,0.15)',
            linewidth=1,
            side='left',
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02, xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#94a3b8', family='IBM Plex Mono'),
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor='#38bdf8',
            font=dict(size=12, family='IBM Plex Mono', color='#e2e8f0'),
        ),
        hovermode='x unified',
    )

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def contango_pct(p1, p2):
    if p1 and p2 and p1 > 0:
        return round((p2 - p1) / p1 * 100, 2)
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Header ──
now_str = datetime.now().strftime("%B %d, %Y · %H:%M")
st.markdown(f"""
<div class="vix-header">
    <div class="logo">VIX<span>Central</span></div>
    <div class="sub">{now_str} · Data: CBOE / Yahoo Finance (delayed)</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_months = st.slider("Futures months", 4, 12, 9, key="nm")
    show_prev = st.checkbox("Show previous day", True)
    show_table = st.checkbox("Show data table", True)
    st.divider()
    st.markdown("### 📅 Compare dates")
    n_compare = st.number_input("Overlay dates", 0, 10, 0, key="nc")
    compare_dates = []
    for i in range(int(n_compare)):
        d = st.date_input(
            f"Date {i+1}",
            value=date.today() - timedelta(days=30*(i+1)),
            max_value=date.today() - timedelta(days=1),
            key=f"cd_{i}"
        )
        compare_dates.append(d)

# ── Tabs ──
tab_live, tab_hist, tab_help = st.tabs(["📈  VIX Term Structure", "📅  Historical", "ℹ️  Help"])

# ━━━━━━━━━━━━━━━━━━━━━━━━ TAB 1: LIVE ━━━━━━━━━━━━━━━━━━━━━━
with tab_live:

    with st.spinner("Loading VIX futures data…"):
        contracts = active_contracts(n=num_months)
        vix_spot = fetch_vix_spot()
        fdata = get_futures_data(contracts)

    # Count how many we got
    found = sum(1 for c in contracts if c['symbol'] in fdata)
    source_label = ""
    if fdata:
        sources = set(v.get('source', '?') for v in fdata.values())
        source_label = " · ".join(sources)

    # ── Metric strip ──
    prices = []
    if vix_spot:
        prices.append(('VIX', vix_spot['price']))
    for c in contracts[:num_months]:
        if c['symbol'] in fdata:
            prices.append((c['code'], fdata[c['symbol']]['price']))

    # Key metrics
    vix_price = vix_spot['price'] if vix_spot else None
    m1_price = fdata[contracts[0]['symbol']]['price'] if contracts and contracts[0]['symbol'] in fdata else None
    m2_price = fdata[contracts[1]['symbol']]['price'] if len(contracts) > 1 and contracts[1]['symbol'] in fdata else None

    front_contango = contango_pct(m1_price, m2_price)
    total_last = None
    for c in reversed(contracts[:num_months]):
        if c['symbol'] in fdata:
            total_last = fdata[c['symbol']]['price']
            break
    total_contango = contango_pct(vix_price, total_last)
    spot_m1_contango = contango_pct(vix_price, m1_price)

    def fmt_price(p):
        return f"{p:.2f}" if p else "—"

    def val_class(v):
        if v is None: return "flat"
        return "up" if v >= 0 else "down"

    def fmt_pct(v):
        if v is None: return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"

    m1_label = contracts[0]['label'] if contracts else ""
    m2_label = contracts[1]['label'] if len(contracts) > 1 else ""
    m1_dte = contracts[0]['dte'] if contracts else "?"
    last_code = "M?"
    for c in reversed(contracts[:num_months]):
        if c['symbol'] in fdata:
            last_code = c['code']
            break

    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-pill">
            <div class="mp-label">VIX Spot</div>
            <div class="mp-value flat">{fmt_price(vix_price)}</div>
        </div>
        <div class="metric-pill">
            <div class="mp-label">M1 · {m1_label} · {m1_dte} DTE</div>
            <div class="mp-value flat">{fmt_price(m1_price)}</div>
        </div>
        <div class="metric-pill">
            <div class="mp-label">M2 · {m2_label}</div>
            <div class="mp-value flat">{fmt_price(m2_price)}</div>
        </div>
        <div class="metric-pill">
            <div class="mp-label">VIX → M1</div>
            <div class="mp-value {val_class(spot_m1_contango)}">{fmt_pct(spot_m1_contango)}</div>
        </div>
        <div class="metric-pill">
            <div class="mp-label">M1 → M2 Contango</div>
            <div class="mp-value {val_class(front_contango)}">{fmt_pct(front_contango)}</div>
        </div>
        <div class="metric-pill">
            <div class="mp-label">VIX → {last_code} Total</div>
            <div class="mp-value {val_class(total_contango)}">{fmt_pct(total_contango)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Historical overlays ──
    overlays = []
    if compare_dates:
        for cd in compare_dates:
            ov = fetch_historical_structure(cd, n=num_months)
            if ov and ov.get('futures'):
                overlays.append(ov)

    # ── Main chart ──
    fig = build_term_chart(vix_spot, contracts, fdata,
                           show_prev=show_prev, overlays=overlays or None,
                           num_months=num_months)
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
    })

    # ── Contango strip (between each consecutive pair) ──
    if len(prices) >= 2:
        cells_html = ""
        for i in range(len(prices) - 1):
            lbl = f"{prices[i][0]}→{prices[i+1][0]}"
            cpct = contango_pct(prices[i][1], prices[i+1][1])
            if cpct is not None:
                cls = "pos" if cpct >= 0 else "neg"
                sign = "+" if cpct >= 0 else ""
                cells_html += f"""
                <div class="contango-cell {cls}">
                    <div class="cc-label">{lbl}</div>
                    <div class="cc-value">{sign}{cpct:.2f}%</div>
                </div>"""
        if cells_html:
            st.markdown(f'<div class="contango-strip">{cells_html}</div>', unsafe_allow_html=True)

    # ── Data table ──
    if show_table and found > 0:
        rows_html = ""
        prev_p = vix_price
        for c in contracts[:num_months]:
            sym = c['symbol']
            if sym in fdata:
                p = fdata[sym]['price']
                prev_day = fdata[sym].get('prev')
                chg = round(p - prev_day, 2) if prev_day else None
                cpct = contango_pct(prev_p, p)
                chg_str = f"{chg:+.2f}" if chg is not None else "—"
                chg_color = "var(--green)" if chg and chg >= 0 else "var(--red)" if chg else "var(--text-dim)"
                cpct_str = f"{cpct:+.2f}%" if cpct is not None else "—"
                cpct_color = "var(--green)" if cpct and cpct >= 0 else "var(--red)" if cpct else "var(--text-dim)"

                rows_html += f"""<tr>
                    <td style="color:var(--accent);font-weight:600">{c['code']}</td>
                    <td>{c['label']}</td>
                    <td style="font-weight:600">{p:.2f}</td>
                    <td style="color:{chg_color}">{chg_str}</td>
                    <td style="color:{cpct_color}">{cpct_str}</td>
                    <td>{c['dte']}</td>
                    <td style="color:var(--text-dim)">{c['exp'].strftime('%Y-%m-%d')}</td>
                    <td style="color:var(--text-dim);font-size:0.68rem">{fdata[sym].get('source','')}</td>
                </tr>"""
                prev_p = p

        st.markdown(f"""
        <table class="data-table">
            <thead><tr>
                <th>Contract</th><th>Month</th><th>Settle</th>
                <th>Chg</th><th>Contango</th><th>DTE</th><th>Expiration</th><th>Source</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    if found == 0:
        st.warning("⚠️ Could not fetch VIX futures prices. CBOE data may be unavailable or Yahoo Finance tickers may have changed. Try refreshing.")
        st.info("💡 If you're running locally, ensure you have internet access to cdn.cboe.com and query2.finance.yahoo.com")

    # Source info
    if source_label:
        st.caption(f"Data source: {source_label} · {found}/{num_months} contracts loaded · Prices delayed ~15 min")

# ━━━━━━━━━━━━━━━━━━━━━━ TAB 2: HISTORICAL ━━━━━━━━━━━━━━━━━━
with tab_hist:
    st.markdown("#### 📅 Historical Term Structure")

    c1, c2 = st.columns([1, 1])
    with c1:
        hist_date = st.date_input("Select date", date.today() - timedelta(days=7),
                                   max_value=date.today(), min_value=date(2013, 1, 1),
                                   key="hist_d")
        hist_n = st.slider("Months", 4, 12, 9, key="hist_n")
        go_btn = st.button("🔍 Get Prices", type="primary")

    with c2:
        multi = st.checkbox("Compare multiple dates")
        multi_dates = []
        if multi:
            n_multi = st.number_input("How many", 2, 20, 3, key="mn")
            cols = st.columns(min(int(n_multi), 4))
            for i in range(int(n_multi)):
                with cols[i % len(cols)]:
                    md = st.date_input(f"#{i+1}", date.today() - timedelta(days=30*(i+1)),
                                       max_value=date.today(), min_value=date(2013, 1, 1),
                                       key=f"md_{i}")
                    multi_dates.append(md)

    if go_btn or multi:
        if multi and multi_dates:
            all_data = []
            for md in multi_dates:
                with st.spinner(f"Loading {md}…"):
                    hd = fetch_historical_structure(md, n=hist_n)
                    if hd and hd.get('futures'):
                        all_data.append(hd)

            if all_data:
                overlay_colors = [
                    '#38bdf8', '#22c55e', '#ef4444', '#eab308', '#a855f7',
                    '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#e879f9',
                    '#14b8a6', '#f43f5e', '#a3e635', '#818cf8', '#fb923c',
                    '#2dd4bf', '#f472b6', '#facc15', '#c084fc', '#34d399',
                ]
                fig = go.Figure()
                for idx, hd in enumerate(all_data):
                    col = overlay_colors[idx % len(overlay_colors)]
                    xv, yv = [], []
                    if hd.get('vix_spot'):
                        xv.append('VIX')
                        yv.append(hd['vix_spot'])
                    for f in hd['futures']:
                        xv.append(f['code'])
                        yv.append(f['price'])
                    fig.add_trace(go.Scatter(
                        x=xv, y=yv,
                        mode='lines+markers+text',
                        name=str(hd['date']),
                        line=dict(color=col, width=2.5, shape='spline'),
                        marker=dict(size=7, color=col, line=dict(width=1.5, color='#0b0e14')),
                        text=[f"{v:.2f}" for v in yv],
                        textposition='top center',
                        textfont=dict(size=9, family='IBM Plex Mono'),
                    ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='#0b0e14', plot_bgcolor='#0f1319',
                    height=520, margin=dict(l=55, r=25, t=40, b=55),
                    title=dict(text=f"VIX Term Structure — {len(all_data)} dates",
                               font=dict(size=14, color='#38bdf8', family='DM Sans'), x=0.5),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.06)',
                               tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono')),
                    xaxis=dict(gridcolor='rgba(148,163,184,0.06)',
                               tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono')),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                bgcolor='rgba(0,0,0,0)',
                                font=dict(size=10, color='#94a3b8', family='IBM Plex Mono')),
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for the selected dates.")

        elif go_btn:
            with st.spinner(f"Loading {hist_date}…"):
                hd = fetch_historical_structure(hist_date, n=hist_n)
            if hd and hd.get('futures'):
                fig = go.Figure()
                xv, yv = [], []
                if hd.get('vix_spot'):
                    xv.append('VIX')
                    yv.append(hd['vix_spot'])
                for f in hd['futures']:
                    xv.append(f['code'])
                    yv.append(f['price'])
                fig.add_trace(go.Scatter(
                    x=xv, y=yv,
                    mode='lines+markers+text',
                    name=str(hist_date),
                    line=dict(color='#38bdf8', width=3, shape='spline'),
                    marker=dict(size=10, color='#38bdf8', line=dict(width=2.5, color='#0b0e14')),
                    text=[f"{v:.2f}" for v in yv],
                    textposition='top center',
                    textfont=dict(size=11, color='#38bdf8', family='IBM Plex Mono'),
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='#0b0e14', plot_bgcolor='#0f1319',
                    height=480, margin=dict(l=55, r=25, t=40, b=55),
                    title=dict(text=f"VIX Term Structure — {hist_date.strftime('%B %d, %Y')}",
                               font=dict(size=14, color='#38bdf8', family='DM Sans'), x=0.5),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.06)',
                               tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono')),
                    xaxis=dict(gridcolor='rgba(148,163,184,0.06)',
                               tickfont=dict(size=11, color='#94a3b8', family='IBM Plex Mono')),
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"VIX Spot: {hd.get('vix_spot', '—')}")
                df_h = pd.DataFrame(hd['futures'])
                st.dataframe(df_h, use_container_width=True, hide_index=True)
            else:
                st.warning("No data available. Try a recent trading day.")

# ━━━━━━━━━━━━━━━━━━━━━━━ TAB 3: HELP ━━━━━━━━━━━━━━━━━━━━━━
with tab_help:
    st.markdown("""
    ### How this works

    This dashboard replicates **[vixcentral.com](https://vixcentral.com)** — the standard tool
    for visualizing the VIX futures term structure.

    **The Term Structure** plots settlement prices of VIX futures contracts (M1 through M9+)
    along with the VIX spot index. The shape tells you about market expectations for volatility.

    **Contango** (upward slope) — futures trade above spot. Normal state, ~82% of trading days.
    Short-vol products like SVXY and SVIX profit from roll yield.

    **Backwardation** (downward slope) — near-term futures above longer-term. Signals crisis/fear.
    Long-vol products like VXX and UVXY benefit.

    ---

    **Data Sources:**
    - **CBOE CDN** — Individual contract settlement CSVs from `cdn.cboe.com`
    - **Yahoo Finance** — VIX spot (`^VIX`) and futures fallback
    - Prices are delayed ~15 minutes

    **VIX Futures Expiration:**
    Wednesday that is 30 calendar days before the 3rd Friday of the following calendar month.

    ---

    | Instrument | Exposure | Description |
    |-----------|----------|-------------|
    | SVXY | -0.5x | ProShares Short VIX Short-Term Futures |
    | SVIX | -1x | -1x Short VIX Futures ETF |
    | VXX | +1x | iPath Series B VIX Short-Term Futures ETN |
    | UVXY | +1.5x | ProShares Ultra VIX Short-Term Futures |
    """)

# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:1.5rem 0 0.5rem; border-top:1px solid rgba(148,163,184,0.08); margin-top:1.5rem;">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#475569;">
        VIX Term Structure Dashboard · Replica of vixcentral.com · Not financial advice
    </span>
</div>
""", unsafe_allow_html=True)
