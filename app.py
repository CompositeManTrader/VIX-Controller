"""
VIX Central — Term Structure Dashboard
Replica of vixcentral.com built with Streamlit + Plotly
Author: Alberto Alarcón González
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, date
import requests
import io
import warnings
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIX Term Structure",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — dark theme matching VIXCentral
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* Dark theme override */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 188, 212, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .main-header h1 {
        color: #00bcd4;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #78909c;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30, #252540);
        border: 1px solid rgba(0, 188, 212, 0.15);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
    }

    .metric-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #78909c;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }

    .metric-card .value.positive { color: #00e676; }
    .metric-card .value.negative { color: #ff5252; }
    .metric-card .value.neutral { color: #00bcd4; }

    /* Contango bar */
    .contango-bar {
        background: #1a1a2e;
        border: 1px solid rgba(0, 188, 212, 0.15);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .contango-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #90a4ae;
    }

    .contango-value {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Table styling */
    .futures-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    .futures-table th {
        background: #1a1a2e;
        color: #00bcd4;
        padding: 0.6rem 1rem;
        text-align: center;
        border-bottom: 2px solid #00bcd4;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .futures-table td {
        padding: 0.5rem 1rem;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: #e0e0e0;
    }

    .futures-table tr:hover td {
        background: rgba(0, 188, 212, 0.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12121f;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

REVERSE_MONTH_CODES = {v: k for k, v in MONTH_CODES.items()}


def get_vix_futures_expiration(year: int, month: int) -> date:
    """
    VIX futures expiration: the Wednesday that is 30 days before
    the third Friday of the calendar month immediately following
    the expiration month.
    """
    # Third Friday of the NEXT month
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1

    # Find third Friday
    first_day = date(next_year, next_month, 1)
    # Day of week: Monday=0, Friday=4
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    third_friday = first_friday + timedelta(weeks=2)

    # 30 days before
    expiration = third_friday - timedelta(days=30)

    return expiration


def generate_active_contracts(ref_date: date = None, num_months: int = 9):
    """Generate list of active VIX futures contracts."""
    if ref_date is None:
        ref_date = date.today()

    contracts = []
    current_month = ref_date.month
    current_year = ref_date.year

    # Start from current month, check if still active
    for i in range(num_months + 3):  # extra buffer
        m = ((current_month - 1 + i) % 12) + 1
        y = current_year + ((current_month - 1 + i) // 12)

        exp = get_vix_futures_expiration(y, m)

        # Only include if not yet expired
        if exp >= ref_date:
            code = MONTH_CODES[m]
            yr_short = str(y)[-2:]
            ticker = f"VX{code}{yr_short}.CBE"
            label = f"{MONTH_NAMES[m]} {y}"
            contracts.append({
                'ticker': ticker,
                'month': m,
                'year': y,
                'expiration': exp,
                'label': label,
                'code': f"M{len(contracts)+1}",
                'days_to_exp': (exp - ref_date).days
            })

        if len(contracts) >= num_months:
            break

    return contracts


@st.cache_data(ttl=60)
def fetch_vix_spot():
    """Fetch current VIX spot price."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            return {
                'price': round(hist['Close'].iloc[-1], 2),
                'date': hist.index[-1].strftime('%Y-%m-%d'),
                'change': round(hist['Close'].iloc[-1] - hist['Close'].iloc[-2], 2) if len(hist) > 1 else 0,
                'history': hist
            }
    except Exception as e:
        st.warning(f"Error fetching VIX spot: {e}")
    return None


@st.cache_data(ttl=60)
def fetch_futures_prices(contracts_info):
    """Fetch current prices for VIX futures contracts."""
    tickers = [c['ticker'] for c in contracts_info]
    prices = {}

    try:
        # Fetch all at once
        data = yf.download(tickers, period="5d", progress=False, threads=True)

        if data.empty:
            return prices

        for contract in contracts_info:
            ticker = contract['ticker']
            try:
                if len(tickers) > 1:
                    if ticker in data['Close'].columns:
                        series = data['Close'][ticker].dropna()
                        if not series.empty:
                            prices[ticker] = {
                                'price': round(float(series.iloc[-1]), 2),
                                'prev_price': round(float(series.iloc[-2]), 2) if len(series) > 1 else None,
                                'date': series.index[-1].strftime('%Y-%m-%d')
                            }
                else:
                    series = data['Close'].dropna()
                    if not series.empty:
                        prices[ticker] = {
                            'price': round(float(series.iloc[-1]), 2),
                            'prev_price': round(float(series.iloc[-2]), 2) if len(series) > 1 else None,
                            'date': series.index[-1].strftime('%Y-%m-%d')
                        }
            except Exception:
                continue

    except Exception as e:
        st.warning(f"Error fetching futures data: {e}")

    # Fallback: fetch individually for missing contracts
    for contract in contracts_info:
        ticker = contract['ticker']
        if ticker not in prices:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if not hist.empty:
                    prices[ticker] = {
                        'price': round(float(hist['Close'].iloc[-1]), 2),
                        'prev_price': round(float(hist['Close'].iloc[-2]), 2) if len(hist) > 1 else None,
                        'date': hist.index[-1].strftime('%Y-%m-%d')
                    }
            except Exception:
                continue

    return prices


@st.cache_data(ttl=300)
def fetch_historical_term_structure(target_date: date, num_months: int = 9):
    """Fetch VIX futures term structure for a historical date."""
    contracts = generate_active_contracts(ref_date=target_date, num_months=num_months)

    # Fetch VIX spot for that date
    vix_spot = None
    try:
        vix = yf.Ticker("^VIX")
        start = target_date - timedelta(days=5)
        end = target_date + timedelta(days=1)
        hist = vix.history(start=start, end=end)
        if not hist.empty:
            # Get closest date
            idx = hist.index.get_indexer([pd.Timestamp(target_date)], method='pad')
            if idx[0] >= 0:
                vix_spot = round(float(hist['Close'].iloc[idx[0]]), 2)
    except Exception:
        pass

    # Fetch futures prices
    result = {'date': target_date, 'vix_spot': vix_spot, 'futures': []}

    tickers = [c['ticker'] for c in contracts]
    try:
        start_dt = target_date - timedelta(days=5)
        end_dt = target_date + timedelta(days=1)
        data = yf.download(tickers, start=start_dt, end=end_dt, progress=False, threads=True)

        if not data.empty:
            for contract in contracts:
                ticker = contract['ticker']
                try:
                    if len(tickers) > 1 and ticker in data['Close'].columns:
                        series = data['Close'][ticker].dropna()
                    elif len(tickers) == 1:
                        series = data['Close'].dropna()
                    else:
                        continue

                    if not series.empty:
                        idx = series.index.get_indexer([pd.Timestamp(target_date)], method='pad')
                        if idx[0] >= 0:
                            price = round(float(series.iloc[idx[0]]), 2)
                            result['futures'].append({
                                'label': contract['label'],
                                'code': contract['code'],
                                'price': price,
                                'expiration': contract['expiration'],
                                'days_to_exp': contract['days_to_exp']
                            })
                except Exception:
                    continue
    except Exception:
        pass

    return result


def calc_contango(price_near, price_far):
    """Calculate contango percentage."""
    if price_near and price_far and price_near > 0:
        return round(((price_far - price_near) / price_near) * 100, 2)
    return None


def build_term_structure_chart(
    vix_spot,
    futures_data,
    contracts,
    title="VIX Futures Term Structure",
    prev_day_data=None,
    additional_dates=None,
    show_wide=False,
):
    """Build the main Plotly term structure chart."""

    fig = go.Figure()

    # Colors
    colors = {
        'today': '#00bcd4',
        'prev': '#ff9800',
        'vix_spot': '#ff5252',
        'grid': 'rgba(255,255,255,0.06)',
        'text': '#b0bec5',
    }
    date_colors = [
        '#00e676', '#ff5252', '#ffd740', '#e040fb',
        '#00b0ff', '#ff6e40', '#69f0ae', '#ea80fc',
        '#40c4ff', '#ff9100', '#b2ff59', '#8c9eff',
        '#18ffff', '#ff6d00', '#ccff90', '#b388ff',
        '#84ffff', '#dd2c00', '#f4ff81', '#651fff',
    ]

    # X-axis labels and positions
    x_labels = ['VIX\nSpot']
    x_positions = [0]
    y_values_today = [vix_spot['price'] if vix_spot else None]

    for i, contract in enumerate(contracts):
        ticker = contract['ticker']
        if ticker in futures_data:
            x_labels.append(f"{contract['code']}\n{contract['label']}")
            x_positions.append(i + 1)
            y_values_today.append(futures_data[ticker]['price'])
        else:
            x_labels.append(f"{contract['code']}\n{contract['label']}")
            x_positions.append(i + 1)
            y_values_today.append(None)

    # Today's curve
    valid_x = [x for x, y in zip(x_positions, y_values_today) if y is not None]
    valid_y = [y for y in y_values_today if y is not None]

    if valid_y:
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=valid_y,
            mode='lines+markers+text',
            name='Today',
            line=dict(color=colors['today'], width=3),
            marker=dict(size=10, color=colors['today'], line=dict(width=2, color='#0e1117')),
            text=[f"{v:.2f}" for v in valid_y],
            textposition='top center',
            textfont=dict(size=11, color=colors['today'], family='JetBrains Mono'),
            hovertemplate='<b>%{text}</b><extra></extra>',
        ))

    # VIX Spot marker (highlighted)
    if vix_spot and vix_spot['price']:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[vix_spot['price']],
            mode='markers',
            name='VIX Spot',
            marker=dict(size=14, color=colors['vix_spot'], symbol='diamond',
                       line=dict(width=2, color='white')),
            hovertemplate=f"<b>VIX Spot: {vix_spot['price']:.2f}</b><extra></extra>",
            showlegend=True,
        ))

    # Previous day curve
    if prev_day_data:
        prev_y = [prev_day_data.get('vix_spot')]
        for contract in contracts:
            ticker = contract['ticker']
            if ticker in futures_data and futures_data[ticker].get('prev_price'):
                prev_y.append(futures_data[ticker]['prev_price'])
            else:
                prev_y.append(None)

        prev_valid_x = [x for x, y in zip(x_positions, prev_y) if y is not None]
        prev_valid_y = [y for y in prev_y if y is not None]

        if prev_valid_y:
            fig.add_trace(go.Scatter(
                x=prev_valid_x,
                y=prev_valid_y,
                mode='lines+markers',
                name='Previous Day',
                line=dict(color=colors['prev'], width=2, dash='dash'),
                marker=dict(size=7, color=colors['prev']),
                text=[f"{v:.2f}" for v in prev_valid_y],
                hovertemplate='<b>Prev: %{text}</b><extra></extra>',
            ))

    # Additional historical dates
    if additional_dates:
        for idx, date_data in enumerate(additional_dates):
            color = date_colors[idx % len(date_colors)]
            hist_y = [date_data.get('vix_spot')]
            for f in date_data.get('futures', []):
                hist_y.append(f['price'])

            hist_x = list(range(len(hist_y)))
            hist_valid_x = [x for x, y in zip(hist_x, hist_y) if y is not None]
            hist_valid_y = [y for y in hist_y if y is not None]

            if hist_valid_y:
                fig.add_trace(go.Scatter(
                    x=hist_valid_x,
                    y=hist_valid_y,
                    mode='lines+markers',
                    name=str(date_data['date']),
                    line=dict(color=color, width=2),
                    marker=dict(size=7, color=color),
                    text=[f"{v:.2f}" for v in hist_valid_y],
                    hovertemplate=f"<b>{date_data['date']}</b>: " + '%{text}<extra></extra>',
                ))

    # Layout
    height = 550 if not show_wide else 650
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#00bcd4', family='Space Grotesk'),
            x=0.5,
        ),
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#12121f',
        height=height,
        margin=dict(l=60, r=40, t=60, b=80),
        xaxis=dict(
            tickvals=x_positions,
            ticktext=x_labels,
            tickfont=dict(size=10, color=colors['text'], family='JetBrains Mono'),
            gridcolor=colors['grid'],
            zeroline=False,
            showline=True,
            linecolor='rgba(255,255,255,0.1)',
        ),
        yaxis=dict(
            title=dict(text='Price', font=dict(size=12, color=colors['text'], family='Space Grotesk')),
            tickfont=dict(size=11, color=colors['text'], family='JetBrains Mono'),
            gridcolor=colors['grid'],
            zeroline=False,
            showline=True,
            linecolor='rgba(255,255,255,0.1)',
        ),
        legend=dict(
            bgcolor='rgba(18,18,31,0.9)',
            bordercolor='rgba(0,188,212,0.2)',
            borderwidth=1,
            font=dict(size=11, color='#e0e0e0', family='JetBrains Mono'),
        ),
        hovermode='x unified',
    )

    return fig


def build_contango_chart(contango_values, labels):
    """Build contango bar chart."""
    colors = ['#00e676' if v >= 0 else '#ff5252' for v in contango_values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=contango_values,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in contango_values],
        textposition='outside',
        textfont=dict(size=11, family='JetBrains Mono'),
    ))

    fig.update_layout(
        title=dict(
            text='Contango / Backwardation Between Months',
            font=dict(size=14, color='#00bcd4', family='Space Grotesk'),
            x=0.5,
        ),
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#12121f',
        height=350,
        margin=dict(l=40, r=40, t=50, b=60),
        xaxis=dict(
            tickfont=dict(size=10, color='#b0bec5', family='JetBrains Mono'),
            gridcolor='rgba(255,255,255,0.06)',
        ),
        yaxis=dict(
            title=dict(text='Contango %', font=dict(size=11, color='#b0bec5', family='Space Grotesk')),
            tickfont=dict(size=10, color='#b0bec5', family='JetBrains Mono'),
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)',
        ),
    )

    return fig


def build_historical_contango_chart():
    """Build historical contango time series (M1-M2 spread)."""
    try:
        # Download historical VIX futures continuous contracts
        vx1 = yf.Ticker("^VIX")  # We'll approximate with VIX data
        hist = vx1.history(period="2y")
        if hist.empty:
            return None

        # This is simplified — would need actual futures data for real contango history
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">📊</span>
        <h2 style="color: #00bcd4; font-family: 'Space Grotesk', sans-serif; margin: 0.5rem 0 0 0;">
            VIX Central
        </h2>
        <p style="color: #78909c; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
            Term Structure Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Number of futures months
    num_futures = st.slider(
        "Number of Futures Months",
        min_value=4, max_value=12, value=9,
        help="Number of VIX futures months to display (like VIXCentral's 7/9 toggle)"
    )

    show_prev_day = st.checkbox("Show Previous Trading Day", value=True)
    show_contango_bars = st.checkbox("Show Contango Bars", value=True)
    show_data_table = st.checkbox("Show Data Table", value=True)

    st.divider()

    # Historical comparison
    st.markdown("### 📅 Historical Comparison")
    st.markdown(
        '<p style="font-size:0.75rem; color:#78909c;">Add up to 5 historical dates to overlay on the chart</p>',
        unsafe_allow_html=True
    )

    num_hist_dates = st.number_input(
        "Number of dates to compare",
        min_value=0, max_value=5, value=0
    )

    historical_dates = []
    for i in range(int(num_hist_dates)):
        d = st.date_input(
            f"Date {i+1}",
            value=date.today() - timedelta(days=30 * (i + 1)),
            max_value=date.today() - timedelta(days=1),
            key=f"hist_date_{i}"
        )
        historical_dates.append(d)

    st.divider()
    st.markdown("""
    <div style="padding: 0.8rem; background: rgba(0,188,212,0.05); border-radius: 8px; border: 1px solid rgba(0,188,212,0.1);">
        <p style="font-size: 0.7rem; color: #78909c; font-family: 'JetBrains Mono', monospace; margin: 0;">
            💡 <b>Tip:</b> Data refreshes every 60s.<br>
            Source: Yahoo Finance (delayed ~15min)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 VIX Futures Term Structure</h1>
    <p>Real-time VIX futures curve · Contango & Backwardation · Historical comparison</p>
</div>
""", unsafe_allow_html=True)

# Tabs like VIXCentral
tab1, tab2, tab3 = st.tabs(["🔴 VIX Term Months", "📊 Historical Prices", "❓ Help"])

# ─────────────────────────── TAB 1: LIVE TERM STRUCTURE ───────────────────────────
with tab1:
    with st.spinner("Fetching VIX futures data..."):
        # Generate contracts
        contracts = generate_active_contracts(num_months=num_futures)

        # Fetch data
        vix_spot = fetch_vix_spot()
        contracts_tuple = tuple((c['ticker'], c['label'], c['code']) for c in contracts)
        futures_prices = fetch_futures_prices(contracts)

    # ── Top metrics row ──
    if vix_spot:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            change_class = "positive" if (vix_spot.get('change', 0) or 0) >= 0 else "negative"
            change_sign = "+" if (vix_spot.get('change', 0) or 0) >= 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">VIX Spot</div>
                <div class="value neutral">{vix_spot['price']:.2f}</div>
                <div class="label">{change_sign}{vix_spot.get('change', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # M1 and M2 prices
        m1_price = None
        m2_price = None
        if len(contracts) >= 1 and contracts[0]['ticker'] in futures_prices:
            m1_price = futures_prices[contracts[0]['ticker']]['price']
        if len(contracts) >= 2 and contracts[1]['ticker'] in futures_prices:
            m2_price = futures_prices[contracts[1]['ticker']]['price']

        with col2:
            m1_label = contracts[0]['label'] if contracts else "M1"
            m1_display = f"{m1_price:.2f}" if m1_price else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">M1 ({m1_label})</div>
                <div class="value neutral">{m1_display}</div>
                <div class="label">{contracts[0].get('days_to_exp', '?')} DTE</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            m2_label = contracts[1]['label'] if len(contracts) > 1 else "M2"
            m2_display = f"{m2_price:.2f}" if m2_price else "N/A"
            m2_dte = contracts[1].get('days_to_exp', '?') if len(contracts) > 1 else '?'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">M2 ({m2_label})</div>
                <div class="value neutral">{m2_display}</div>
                <div class="label">{m2_dte} DTE</div>
            </div>
            """, unsafe_allow_html=True)

        # Contango M1-M2
        contango_12 = calc_contango(m1_price, m2_price) if m1_price and m2_price else None
        with col4:
            if contango_12 is not None:
                c_class = "positive" if contango_12 >= 0 else "negative"
                c_label = "Contango" if contango_12 >= 0 else "Backwardation"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">M1→M2 {c_label}</div>
                    <div class="value {c_class}">{contango_12:+.2f}%</div>
                    <div class="label">Front spread</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">M1→M2</div>
                    <div class="value neutral">N/A</div>
                </div>
                """, unsafe_allow_html=True)

        # Total contango (VIX spot to M7/last)
        last_price = None
        for c in reversed(contracts):
            if c['ticker'] in futures_prices:
                last_price = futures_prices[c['ticker']]['price']
                break

        contango_total = calc_contango(vix_spot['price'], last_price) if vix_spot and last_price else None
        with col5:
            if contango_total is not None:
                c_class = "positive" if contango_total >= 0 else "negative"
                c_label = "Contango" if contango_total >= 0 else "Backwardation"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">VIX→M{num_futures} {c_label}</div>
                    <div class="value {c_class}">{contango_total:+.2f}%</div>
                    <div class="label">Full curve</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">Total Curve</div>
                    <div class="value neutral">N/A</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    # ── Build previous day data ──
    prev_data = None
    if show_prev_day and vix_spot and vix_spot.get('history') is not None and len(vix_spot['history']) > 1:
        prev_data = {'vix_spot': round(float(vix_spot['history']['Close'].iloc[-2]), 2)}

    # ── Fetch historical overlays ──
    additional_hist = []
    if historical_dates:
        for hd in historical_dates:
            with st.spinner(f"Loading {hd}..."):
                hist_data = fetch_historical_term_structure(hd, num_months=num_futures)
                if hist_data and hist_data.get('futures'):
                    additional_hist.append(hist_data)

    # ── Main chart ──
    chart = build_term_structure_chart(
        vix_spot=vix_spot,
        futures_data=futures_prices,
        contracts=contracts,
        title=f"VIX Futures Term Structure — {date.today().strftime('%B %d, %Y')}",
        prev_day_data=prev_data if show_prev_day else None,
        additional_dates=additional_hist if additional_hist else None,
    )
    st.plotly_chart(chart, use_container_width=True)

    # ── Contango bars ──
    if show_contango_bars:
        # Calculate contango between consecutive months
        prices_list = []
        labels_list = []

        if vix_spot:
            prices_list.append(vix_spot['price'])
            labels_list.append('VIX')

        for c in contracts:
            if c['ticker'] in futures_prices:
                prices_list.append(futures_prices[c['ticker']]['price'])
                labels_list.append(c['code'])

        if len(prices_list) >= 2:
            contango_vals = []
            contango_labels = []
            for i in range(len(prices_list) - 1):
                cv = calc_contango(prices_list[i], prices_list[i + 1])
                if cv is not None:
                    contango_vals.append(cv)
                    contango_labels.append(f"{labels_list[i]}→{labels_list[i+1]}")

            if contango_vals:
                contango_chart = build_contango_chart(contango_vals, contango_labels)
                st.plotly_chart(contango_chart, use_container_width=True)

    # ── Data table ──
    if show_data_table:
        st.markdown("### 📋 Futures Data")

        table_data = []
        prev_price_val = vix_spot['price'] if vix_spot else None

        for c in contracts:
            ticker = c['ticker']
            if ticker in futures_prices:
                price = futures_prices[ticker]['price']
                contango = calc_contango(prev_price_val, price) if prev_price_val else None
                table_data.append({
                    'Contract': c['code'],
                    'Month': c['label'],
                    'Price': f"${price:.2f}",
                    'DTE': c['days_to_exp'],
                    'Expiration': c['expiration'].strftime('%Y-%m-%d'),
                    'Contango vs Prev': f"{contango:+.2f}%" if contango is not None else "N/A",
                    'Ticker': ticker,
                })
                prev_price_val = price

        if table_data:
            df_table = pd.DataFrame(table_data)
            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Contract': st.column_config.TextColumn('Contract', width='small'),
                    'Month': st.column_config.TextColumn('Month', width='medium'),
                    'Price': st.column_config.TextColumn('Price', width='small'),
                    'DTE': st.column_config.NumberColumn('DTE', width='small'),
                    'Expiration': st.column_config.TextColumn('Expiration', width='medium'),
                    'Contango vs Prev': st.column_config.TextColumn('Contango', width='medium'),
                    'Ticker': st.column_config.TextColumn('YF Ticker', width='medium'),
                }
            )


# ─────────────────────────── TAB 2: HISTORICAL PRICES ───────────────────────────
with tab2:
    st.markdown("### 📅 Historical Term Structure")
    st.markdown(
        '<p style="color: #78909c; font-size: 0.85rem;">Select a date to view the VIX futures term structure as of that day.</p>',
        unsafe_allow_html=True
    )

    col_date1, col_date2 = st.columns([1, 2])

    with col_date1:
        hist_target = st.date_input(
            "Select Date",
            value=date.today() - timedelta(days=7),
            max_value=date.today(),
            min_value=date(2010, 1, 1),
            key="hist_main_date"
        )

        hist_months = st.slider(
            "Futures Months", 4, 12, 9,
            key="hist_months"
        )

        fetch_hist = st.button("🔍 Get Prices", type="primary", use_container_width=True)

    with col_date2:
        # Multiple dates mode
        st.markdown("**Compare Multiple Dates**")
        multi_mode = st.checkbox("Enable multi-date comparison", value=False)

        multi_dates = []
        if multi_mode:
            num_compare = st.number_input("Number of dates", 2, 20, 3, key="multi_num")
            cols_per_row = 4
            for row_start in range(0, int(num_compare), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col in enumerate(row_cols):
                    idx = row_start + j
                    if idx < int(num_compare):
                        with col:
                            d = st.date_input(
                                f"Date {idx+1}",
                                value=date.today() - timedelta(days=30 * (idx + 1)),
                                max_value=date.today(),
                                min_value=date(2010, 1, 1),
                                key=f"multi_date_{idx}"
                            )
                            multi_dates.append(d)

    if fetch_hist or multi_mode:
        if multi_mode and multi_dates:
            all_hist_data = []
            for md in multi_dates:
                with st.spinner(f"Fetching {md}..."):
                    hd = fetch_historical_term_structure(md, num_months=hist_months)
                    if hd and hd.get('futures'):
                        all_hist_data.append(hd)

            if all_hist_data:
                # Build multi-date chart
                fig = go.Figure()
                date_colors = [
                    '#00bcd4', '#00e676', '#ff5252', '#ffd740', '#e040fb',
                    '#00b0ff', '#ff6e40', '#69f0ae', '#ea80fc', '#40c4ff',
                    '#ff9100', '#b2ff59', '#8c9eff', '#18ffff', '#ff6d00',
                    '#ccff90', '#b388ff', '#84ffff', '#dd2c00', '#f4ff81',
                ]

                for idx, hdata in enumerate(all_hist_data):
                    color = date_colors[idx % len(date_colors)]
                    x_vals = []
                    y_vals = []

                    if hdata.get('vix_spot'):
                        x_vals.append('VIX Spot')
                        y_vals.append(hdata['vix_spot'])

                    for f in hdata['futures']:
                        x_vals.append(f['code'])
                        y_vals.append(f['price'])

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers+text',
                        name=str(hdata['date']),
                        line=dict(color=color, width=2.5),
                        marker=dict(size=8, color=color),
                        text=[f"{v:.2f}" for v in y_vals],
                        textposition='top center',
                        textfont=dict(size=9, family='JetBrains Mono'),
                    ))

                fig.update_layout(
                    title=dict(
                        text=f'Historical VIX Term Structure — {len(all_hist_data)} Dates',
                        font=dict(size=16, color='#00bcd4', family='Space Grotesk'),
                        x=0.5,
                    ),
                    template='plotly_dark',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#12121f',
                    height=600,
                    margin=dict(l=60, r=40, t=60, b=80),
                    yaxis=dict(title='Price', gridcolor='rgba(255,255,255,0.06)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                    legend=dict(
                        bgcolor='rgba(18,18,31,0.9)',
                        bordercolor='rgba(0,188,212,0.2)',
                        font=dict(family='JetBrains Mono', size=10),
                    ),
                    hovermode='x unified',
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected dates.")

        else:
            with st.spinner(f"Fetching term structure for {hist_target}..."):
                hist_result = fetch_historical_term_structure(hist_target, num_months=hist_months)

            if hist_result and hist_result.get('futures'):
                # Build single date chart
                fig = go.Figure()

                x_vals = []
                y_vals = []

                if hist_result.get('vix_spot'):
                    x_vals.append('VIX Spot')
                    y_vals.append(hist_result['vix_spot'])

                for f in hist_result['futures']:
                    x_vals.append(f'{f["code"]}\n{f["label"]}' if 'label' in f else f['code'])
                    y_vals.append(f['price'])

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers+text',
                    name=str(hist_target),
                    line=dict(color='#00bcd4', width=3),
                    marker=dict(size=10, color='#00bcd4', line=dict(width=2, color='#0e1117')),
                    text=[f"{v:.2f}" for v in y_vals],
                    textposition='top center',
                    textfont=dict(size=11, color='#00bcd4', family='JetBrains Mono'),
                ))

                fig.update_layout(
                    title=dict(
                        text=f'VIX Term Structure — {hist_target.strftime("%B %d, %Y")}',
                        font=dict(size=16, color='#00bcd4', family='Space Grotesk'),
                        x=0.5,
                    ),
                    template='plotly_dark',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#12121f',
                    height=550,
                    margin=dict(l=60, r=40, t=60, b=80),
                    yaxis=dict(title='Price', gridcolor='rgba(255,255,255,0.06)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                    hovermode='x unified',
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show data
                st.markdown(f"**VIX Spot:** {hist_result.get('vix_spot', 'N/A')}")
                df_hist = pd.DataFrame(hist_result['futures'])
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.warning("No data available for the selected date. Try a recent trading day.")


# ─────────────────────────── TAB 3: HELP ───────────────────────────
with tab3:
    st.markdown("""
    ### ℹ️ About This Dashboard

    This dashboard replicates the functionality of **[VIXCentral.com](https://vixcentral.com)** — the go-to tool for visualizing the VIX futures term structure.

    ---

    #### 🔹 What is the VIX Term Structure?

    The **VIX Futures Term Structure** plots the prices of VIX futures contracts across different expiration months. The shape of this curve tells you about market expectations for future volatility.

    **Contango** (upward sloping): The market expects future VIX to be higher than current VIX — this is the normal state (~82% of the time). Short volatility products like **SVXY** and **SVIX** benefit from roll yield decay when the curve is in contango.

    **Backwardation** (downward sloping): Near-term futures are priced higher than longer-term futures — this typically occurs during market stress/crisis events. Long volatility products like **VXX** and **UVXY** benefit during backwardation.

    ---

    #### 🔹 How to Use

    - **VIX Term Months tab**: Shows the live term structure with up to 12 months of futures
    - **Historical Prices tab**: View the term structure for any past date since 2010
    - **Previous Day overlay**: Toggle to compare today's curve vs yesterday
    - **Multi-date comparison**: Overlay up to 20 dates on a single chart
    - **Contango bars**: Visual breakdown of contango between each consecutive month

    ---

    #### 🔹 Key Metrics

    | Metric | Description |
    |--------|-------------|
    | **VIX Spot** | Current VIX Index value (30-day implied volatility of S&P 500) |
    | **M1, M2, ...** | Front-month, second-month, etc. VIX futures contracts |
    | **DTE** | Days to expiration for each futures contract |
    | **Contango %** | Percentage difference between consecutive months |
    | **M1→M2 Contango** | The critical front spread — drives SVXY/VXX daily roll yield |

    ---

    #### 🔹 Data Source

    Data sourced from **Yahoo Finance** (delayed ~15 minutes for futures).
    VIX futures contracts follow the naming convention `VX{MonthCode}{Year}.CBE`.

    ---

    #### 🔹 Related Instruments

    | Instrument | Exposure | Description |
    |-----------|----------|-------------|
    | **SVXY** | -0.5x | ProShares Short VIX Short-Term Futures ETF |
    | **SVIX** | -1x | -1x Short VIX Futures ETF |
    | **VXX** | +1x | Barclays iPath Series B S&P 500 VIX Short-Term Futures ETN |
    | **UVXY** | +1.5x | ProShares Ultra VIX Short-Term Futures ETF |

    ---

    <p style="color: #546e7a; font-size: 0.75rem; text-align: center; margin-top: 2rem;">
        Built by Alberto Alarcón González · Powered by Streamlit + Plotly + yfinance<br>
        Inspired by <a href="https://vixcentral.com" style="color: #00bcd4;">VIXCentral.com</a>
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────── FOOTER ───────────────────────────
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0 0.5rem 0; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 2rem;">
    <p style="color: #546e7a; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
        VIX Term Structure Dashboard · Data: Yahoo Finance (delayed) · Not financial advice
    </p>
</div>
""", unsafe_allow_html=True)
