"""
VIX Central — Term Structure Dashboard
Updated to source the live VIX futures curve directly from Cboe's VIX Futures page,
filtering monthly contracts and excluding weekly expirations.
Historical lookups continue to use official Cboe contract CSVs when available.
"""

import io
import re
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

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
MONTHLY_VX_RE = re.compile(r"^VX/[FGHJKMNQUVXZ]\d$")
CBOE_VIX_FUTURES_URL = "https://www.cboe.com/tradable-products/vix/vix-futures/"
CBOE_VIX_SETTLEMENT_CSV = "https://www.cboe.com/us/futures/market_statistics/settlement/csv"
CBOE_VIX_HISTORICAL_CSV = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{exp}.csv"
USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


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
    """Return list of dicts for the next n active MONTHLY VIX futures."""
    if ref is None:
        ref = date.today()
    out = []
    m, y = ref.month, ref.year
    for i in range(n + 6):
        cm = ((m - 1 + i) % 12) + 1
        cy = y + ((m - 1 + i) // 12)
        exp = vix_futures_expiration(cy, cm)
        if exp >= ref:
            out.append({
                "month": cm,
                "year": cy,
                "exp": exp,
                "dte": (exp - ref).days,
                "label": f"{MONTH_NAMES_SHORT[cm]} {cy}",
                "code": f"M{len(out) + 1}",
                "symbol": f"VX/{MONTH_CODES[cm]}{str(cy)[-1]}",
            })
        if len(out) >= n:
            break
    return out


def is_monthly_vx_symbol(symbol: str) -> bool:
    return bool(MONTHLY_VX_RE.fullmatch(str(symbol).strip().upper()))


def safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if cleaned in {"", "-", "--", "N/A", "nan", "None"}:
                return None
            value = cleaned
        num = float(value)
        if np.isnan(num):
            return None
        return float(num)
    except Exception:
        return None


def fmt_price(p):
    return f"{p:.2f}" if p is not None and pd.notna(p) else "—"


def val_class(v):
    if v is None or pd.isna(v):
        return "flat"
    return "up" if v >= 0 else "down"


def fmt_pct(v):
    if v is None or pd.isna(v):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def contango_pct(p1, p2):
    if p1 is not None and p2 is not None and pd.notna(p1) and pd.notna(p2) and p1 > 0:
        return round((p2 - p1) / p1 * 100, 2)
    return None


def _normalize_live_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        c_low = str(c).strip().lower()
        if "symbol" in c_low:
            cols[c] = "Symbol"
        elif "expiration" in c_low:
            cols[c] = "Expiration"
        elif "last" in c_low and "price" in c_low:
            cols[c] = "Last"
        elif c_low == "last":
            cols[c] = "Last"
        elif "change" in c_low:
            cols[c] = "Change"
        elif "settlement" in c_low:
            cols[c] = "Settlement"
        elif "volume" in c_low:
            cols[c] = "Volume"
        elif "high" == c_low or c_low.endswith(" high"):
            cols[c] = "High"
        elif "low" == c_low or c_low.endswith(" low"):
            cols[c] = "Low"
    df = df.rename(columns=cols)
    needed = [c for c in ["Symbol", "Expiration", "Last", "Change", "Settlement", "Volume"] if c in df.columns]
    df = df[needed].copy()
    if "Symbol" not in df.columns or "Expiration" not in df.columns:
        return pd.DataFrame()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].apply(is_monthly_vx_symbol)].copy()
    if df.empty:
        return df
    df["Expiration"] = pd.to_datetime(df["Expiration"], errors="coerce")
    for col in ["Last", "Change", "Settlement", "Volume"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
    if "Last" not in df.columns:
        df["Last"] = np.nan
    if "Settlement" not in df.columns:
        df["Settlement"] = np.nan
    df["Price"] = df["Last"]
    missing_last = df["Price"].isna() | (df["Price"] <= 0)
    df.loc[missing_last, "Price"] = df.loc[missing_last, "Settlement"]
    df = df.dropna(subset=["Expiration", "Price"])
    df = df.sort_values("Expiration").reset_index(drop=True)
    return df


@st.cache_data(ttl=300)
def fetch_cboe_live_monthly_curve() -> pd.DataFrame:
    """
    Primary source: official Cboe daily settlement CSV for all futures on the selected date.
    This is more stable than scraping the rendered product page.
    Fallback: try parsing the public VIX futures page.
    """
    today_str = pd.Timestamp(date.today()).strftime("%Y-%m-%d")

    # Attempt 1: official settlement CSV endpoint for today
    try:
        resp = requests.get(
            CBOE_VIX_SETTLEMENT_CSV,
            params={"dt": today_str},
            headers=USER_AGENT,
            timeout=20,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        # Expected columns typically include Product, Symbol, Expiration Date, Daily Settlement Price
        rename_map = {}
        for c in df.columns:
            c_low = str(c).strip().lower()
            if c_low == "symbol":
                rename_map[c] = "Symbol"
            elif "expiration" in c_low:
                rename_map[c] = "Expiration"
            elif "settlement" in c_low or c_low == "price":
                rename_map[c] = "Settlement"
            elif c_low == "product":
                rename_map[c] = "Product"
        df = df.rename(columns=rename_map)

        if "Product" in df.columns:
            df = df[df["Product"].astype(str).str.upper().eq("VX")].copy()
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
            df = df[df["Symbol"].apply(is_monthly_vx_symbol)].copy()
        if "Expiration" in df.columns:
            df["Expiration"] = pd.to_datetime(df["Expiration"], errors="coerce")
        if "Settlement" in df.columns:
            df["Settlement"] = df["Settlement"].apply(safe_float)

        if not df.empty and {"Symbol", "Expiration", "Settlement"}.issubset(df.columns):
            df["Last"] = np.nan
            df["Change"] = np.nan
            df["Price"] = df["Settlement"]
            df = df.dropna(subset=["Expiration", "Price"]).sort_values("Expiration").reset_index(drop=True)
            return df[["Symbol", "Expiration", "Last", "Change", "Settlement", "Price"]]
    except Exception:
        pass

    # Attempt 2: product page scrape fallback
    resp = requests.get(CBOE_VIX_FUTURES_URL, headers=USER_AGENT, timeout=20)
    resp.raise_for_status()

    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception:
        tables = []

    for table in tables:
        normalized = _normalize_live_table(table)
        if not normalized.empty:
            return normalized

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(VX(?:\d+)?/[FGHJKMNQUVXZ]\d)\s+(\d{2}/\d{2}/\d{4})(.*)$", line)
        if not m:
            continue
        symbol = m.group(1).upper()
        if not is_monthly_vx_symbol(symbol):
            continue
        expiration = pd.to_datetime(m.group(2), errors="coerce")
        numeric_tail = [safe_float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", m.group(3))]
        numeric_tail = [x for x in numeric_tail if x is not None]
        settlement = numeric_tail[-1] if numeric_tail else None
        if expiration is not pd.NaT and settlement is not None:
            rows.append({
                "Symbol": symbol,
                "Expiration": expiration,
                "Last": np.nan,
                "Change": np.nan,
                "Settlement": settlement,
                "Price": settlement,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        return df.sort_values("Expiration").reset_index(drop=True)

    raise ValueError("No se pudo extraer la curva mensual de VX desde Cboe.")

@st.cache_data(ttl=300)
def fetch_vix_spot():
    """Get current VIX spot from Yahoo Finance as fallback for spot only."""
    try:
        vix = yf.Ticker("^VIX")
        h = vix.history(period="5d", auto_adjust=False)
        if not h.empty:
            cur = round(float(h["Close"].iloc[-1]), 2)
            prev = round(float(h["Close"].iloc[-2]), 2) if len(h) > 1 else cur
            return {
                "price": cur,
                "prev": prev,
                "change": round(cur - prev, 2),
                "pct": round((cur - prev) / prev * 100, 2) if prev else 0,
                "source": "Yahoo",
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def fetch_contract_history(expiration_dt: date | datetime | pd.Timestamp):
    exp = pd.Timestamp(expiration_dt).strftime("%Y-%m-%d")
    url = CBOE_VIX_HISTORICAL_CSV.format(exp=exp)
    resp = requests.get(url, headers=USER_AGENT, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    if "Trade Date" in df.columns:
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        df = df.sort_values("Trade Date").reset_index(drop=True)
    for col in ["Settle", "Close", "Last", "Volume", "Open Interest"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
    return df


def latest_history_price(expiration_dt, as_of: date | None = None):
    df = fetch_contract_history(expiration_dt)
    if df.empty:
        return None
    if as_of is not None and "Trade Date" in df.columns:
        df = df[df["Trade Date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return None
    row = df.iloc[-1]
    for col in ["Settle", "Close", "Last"]:
        if col in df.columns:
            val = safe_float(row.get(col))
            if val is not None and val > 0:
                return round(val, 2)
    return None


def build_live_contracts(num_months: int):
    live_df = fetch_cboe_live_monthly_curve().head(num_months).copy()
    if live_df.empty:
        return [], {}

    contracts = []
    fdata = {}
    today = pd.Timestamp(date.today())

    for idx, row in live_df.iterrows():
        exp_ts = pd.Timestamp(row["Expiration"])
        exp_date = exp_ts.date()
        price = safe_float(row.get("Price"))
        prev = latest_history_price(exp_date)
        change_from_page = safe_float(row.get("Change"))
        if change_from_page is None and price is not None and prev is not None:
            change_from_page = round(price - prev, 2)

        contracts.append({
            "month": exp_date.month,
            "year": exp_date.year,
            "exp": exp_date,
            "dte": int((exp_ts.normalize() - today).days),
            "label": f"{MONTH_NAMES_SHORT[exp_date.month]} {exp_date.year}",
            "code": f"M{idx + 1}",
            "symbol": row["Symbol"],
        })
        fdata[row["Symbol"]] = {
            "price": round(price, 2) if price is not None else None,
            "prev": prev,
            "change": change_from_page,
            "settlement": safe_float(row.get("Settlement")),
            "source": "CBOE",
        }

    return contracts, fdata


@st.cache_data(ttl=1800)
def fetch_historical_structure(target: date, n: int = 9):
    """Fetch monthly term structure for a historical date using official Cboe contract CSVs."""
    contracts = active_contracts(ref=target, n=n)

    vix_spot = None
    try:
        vix = yf.Ticker("^VIX")
        start = target - timedelta(days=7)
        end = target + timedelta(days=2)
        h = vix.history(start=start, end=end, auto_adjust=False)
        if not h.empty:
            idx = h.index.get_indexer([pd.Timestamp(target)], method="pad")
            if idx[0] >= 0:
                vix_spot = round(float(h["Close"].iloc[idx[0]]), 2)
    except Exception:
        pass

    futures = []
    for c in contracts:
        price = latest_history_price(c["exp"], as_of=target)
        if price is not None:
            futures.append({
                "label": c["label"],
                "code": c["code"],
                "price": price,
                "symbol": c["symbol"],
                "exp": c["exp"],
            })

    return {"date": target, "vix_spot": vix_spot, "futures": futures}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART BUILDER — VIXCentral style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_term_chart(vix_spot, contracts, fdata, show_prev=True, overlays=None, num_months=9):
    """Build the main term structure chart matching VIXCentral's style."""
    fig = go.Figure()

    x_labels, x_idx, y_today, y_prev = [], [], [], []

    if vix_spot and vix_spot.get("price") is not None:
        x_labels.append("VIX")
        x_idx.append(0)
        y_today.append(vix_spot["price"])
        y_prev.append(vix_spot.get("prev"))

    for i, c in enumerate(contracts[:num_months]):
        x_labels.append(c["code"])
        x_idx.append(i + 1)
        sym = c["symbol"]
        if sym in fdata:
            y_today.append(fdata[sym].get("price"))
            y_prev.append(fdata[sym].get("prev"))
        else:
            y_today.append(None)
            y_prev.append(None)

    vx = [x for x, y in zip(x_idx, y_today) if y is not None]
    vy = [y for y in y_today if y is not None]

    if vy:
        fig.add_trace(go.Scatter(
            x=vx, y=vy,
            mode="lines+markers+text",
            name=date.today().strftime("%b %d, %Y"),
            line=dict(color="#38bdf8", width=3, shape="spline"),
            marker=dict(size=10, color="#38bdf8", line=dict(width=2.5, color="#0b0e14")),
            text=[f"{v:.2f}" for v in vy],
            textposition="top center",
            textfont=dict(size=11, color="#38bdf8", family="IBM Plex Mono"),
            hovertemplate="%{text}<extra></extra>",
        ))

    pvy = []
    if show_prev:
        pvx = [x for x, y in zip(x_idx, y_prev) if y is not None]
        pvy = [y for y in y_prev if y is not None]
        if pvy and len(pvy) >= 2:
            fig.add_trace(go.Scatter(
                x=pvx, y=pvy,
                mode="lines+markers",
                name="Previous Day",
                line=dict(color="#f97316", width=2, dash="dot", shape="spline"),
                marker=dict(size=6, color="#f97316", line=dict(width=1, color="#0b0e14")),
                text=[f"{v:.2f}" for v in pvy],
                hovertemplate="Prev: %{text}<extra></extra>",
            ))

    overlay_colors = [
        "#22c55e", "#ef4444", "#eab308", "#a855f7", "#ec4899",
        "#06b6d4", "#f97316", "#84cc16", "#e879f9", "#14b8a6",
    ]
    if overlays:
        for idx, ov in enumerate(overlays):
            col = overlay_colors[idx % len(overlay_colors)]
            ox, oy = [], []
            if ov.get("vix_spot") is not None:
                ox.append(0)
                oy.append(ov["vix_spot"])
            for j, f in enumerate(ov.get("futures", [])):
                ox.append(j + 1)
                oy.append(f["price"])
            if oy:
                fig.add_trace(go.Scatter(
                    x=ox, y=oy,
                    mode="lines+markers",
                    name=str(ov["date"]),
                    line=dict(color=col, width=2, shape="spline"),
                    marker=dict(size=6, color=col),
                    hovertemplate=f"{ov['date']}: " + "%{y:.2f}<extra></extra>",
                ))

    all_y = vy + pvy
    if overlays:
        for ov in overlays:
            if ov.get("vix_spot") is not None:
                all_y.append(ov["vix_spot"])
            all_y += [f["price"] for f in ov.get("futures", []) if f.get("price") is not None]

    y_min = min(all_y) - 1 if all_y else 10
    y_max = max(all_y) + 2 if all_y else 30

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0f1319",
        height=480,
        margin=dict(l=55, r=25, t=30, b=55),
        xaxis=dict(
            tickvals=x_idx,
            ticktext=x_labels,
            tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono"),
            gridcolor="rgba(148,163,184,0.06)",
            zeroline=False,
            showline=True,
            linecolor="rgba(148,163,184,0.15)",
            linewidth=1,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono"),
            gridcolor="rgba(148,163,184,0.06)",
            zeroline=False,
            showline=True,
            linecolor="rgba(148,163,184,0.15)",
            linewidth=1,
            side="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#94a3b8", family="IBM Plex Mono"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#38bdf8",
            font=dict(size=12, family="IBM Plex Mono", color="#e2e8f0"),
        ),
        hovermode="x unified",
    )

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

now_str = datetime.now().strftime("%B %d, %Y · %H:%M")
st.markdown(f"""
<div class="vix-header">
    <div class="logo">VIX<span>Central</span></div>
    <div class="sub">{now_str} · Live curve: CBOE monthly VX table · Spot: Yahoo fallback</div>
</div>
""", unsafe_allow_html=True)

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
            value=date.today() - timedelta(days=30 * (i + 1)),
            max_value=date.today() - timedelta(days=1),
            key=f"cd_{i}",
        )
        compare_dates.append(d)


tab_live, tab_hist, tab_help = st.tabs(["📈  VIX Term Structure", "📅  Historical", "ℹ️  Help"])

with tab_live:
    with st.spinner("Loading CBOE VIX futures curve…"):
        live_error = None
        contracts, fdata = [], {}
        try:
            contracts, fdata = build_live_contracts(num_months=num_months)
        except Exception as exc:
            live_error = str(exc)
        vix_spot = fetch_vix_spot()

    found = sum(1 for c in contracts if c["symbol"] in fdata and fdata[c["symbol"]].get("price") is not None)
    source_label = "CBOE"

    prices = []
    if vix_spot and vix_spot.get("price") is not None:
        prices.append(("VIX", vix_spot["price"]))
    for c in contracts[:num_months]:
        if c["symbol"] in fdata and fdata[c["symbol"]].get("price") is not None:
            prices.append((c["code"], fdata[c["symbol"]]["price"]))

    vix_price = vix_spot["price"] if vix_spot else None
    m1_price = fdata[contracts[0]["symbol"]]["price"] if contracts else None
    m2_price = fdata[contracts[1]["symbol"]]["price"] if len(contracts) > 1 else None

    front_contango = contango_pct(m1_price, m2_price)
    total_last = None
    for c in reversed(contracts[:num_months]):
        if c["symbol"] in fdata and fdata[c["symbol"]].get("price") is not None:
            total_last = fdata[c["symbol"]]["price"]
            break
    total_contango = contango_pct(vix_price, total_last)
    spot_m1_contango = contango_pct(vix_price, m1_price)

    m1_label = contracts[0]["label"] if contracts else ""
    m2_label = contracts[1]["label"] if len(contracts) > 1 else ""
    m1_dte = contracts[0]["dte"] if contracts else "?"
    last_code = next((c["code"] for c in reversed(contracts[:num_months]) if c["symbol"] in fdata), "M?")

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

    overlays = []
    if compare_dates:
        for cd in compare_dates:
            ov = fetch_historical_structure(cd, n=num_months)
            if ov and ov.get("futures"):
                overlays.append(ov)

    if found > 0:
        fig = build_term_chart(vix_spot, contracts, fdata, show_prev=show_prev, overlays=overlays or None, num_months=num_months)
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

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

        if show_table:
            rows_html = ""
            prev_p = vix_price
            for c in contracts[:num_months]:
                sym = c["symbol"]
                if sym in fdata and fdata[sym].get("price") is not None:
                    p = fdata[sym]["price"]
                    prev_day = fdata[sym].get("prev")
                    chg = fdata[sym].get("change")
                    if chg is None and prev_day is not None:
                        chg = round(p - prev_day, 2)
                    cpct = contango_pct(prev_p, p)
                    chg_str = f"{chg:+.2f}" if chg is not None else "—"
                    chg_color = "var(--green)" if chg is not None and chg >= 0 else "var(--red)" if chg is not None else "var(--text-dim)"
                    cpct_str = f"{cpct:+.2f}%" if cpct is not None else "—"
                    cpct_color = "var(--green)" if cpct is not None and cpct >= 0 else "var(--red)" if cpct is not None else "var(--text-dim)"

                    rows_html += f"""<tr>
                        <td style="color:var(--accent);font-weight:600">{c['code']}</td>
                        <td>{c['label']}</td>
                        <td style="font-weight:600">{p:.2f}</td>
                        <td style="color:{chg_color}">{chg_str}</td>
                        <td style="color:{cpct_color}">{cpct_str}</td>
                        <td>{c['dte']}</td>
                        <td style="color:var(--text-dim)">{c['exp'].strftime('%Y-%m-%d')}</td>
                        <td style="color:var(--text-dim);font-size:0.68rem">{sym}</td>
                    </tr>"""
                    prev_p = p

            st.markdown(f"""
            <table class="data-table">
                <thead><tr>
                    <th>Contract</th><th>Month</th><th>Price</th>
                    <th>Chg</th><th>Contango</th><th>DTE</th><th>Expiration</th><th>CBOE Symbol</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No se pudo cargar la curva mensual de VIX desde CBOE.")
        if live_error:
            st.caption(f"Detalle técnico: {live_error}")

    if found > 0:
        st.caption(f"Live futures source: {source_label} monthly VX table · {found}/{num_months} contratos cargados · Spot source: {vix_spot.get('source', 'N/A') if vix_spot else 'N/A'}")

with tab_hist:
    st.markdown("#### 📅 Historical Term Structure")

    c1, c2 = st.columns([1, 1])
    with c1:
        hist_date = st.date_input("Select date", date.today() - timedelta(days=7), max_value=date.today(), min_value=date(2013, 1, 1), key="hist_d")
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
                    md = st.date_input(
                        f"#{i+1}",
                        date.today() - timedelta(days=30 * (i + 1)),
                        max_value=date.today(),
                        min_value=date(2013, 1, 1),
                        key=f"md_{i}",
                    )
                    multi_dates.append(md)

    if go_btn or multi:
        if multi and multi_dates:
            all_data = []
            for md in multi_dates:
                with st.spinner(f"Loading {md}…"):
                    hd = fetch_historical_structure(md, n=hist_n)
                    if hd and hd.get("futures"):
                        all_data.append(hd)

            if all_data:
                overlay_colors = [
                    "#38bdf8", "#22c55e", "#ef4444", "#eab308", "#a855f7",
                    "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#e879f9",
                    "#14b8a6", "#f43f5e", "#a3e635", "#818cf8", "#fb923c",
                    "#2dd4bf", "#f472b6", "#facc15", "#c084fc", "#34d399",
                ]
                fig = go.Figure()
                for idx, hd in enumerate(all_data):
                    col = overlay_colors[idx % len(overlay_colors)]
                    xv, yv = [], []
                    if hd.get("vix_spot") is not None:
                        xv.append("VIX")
                        yv.append(hd["vix_spot"])
                    for f in hd["futures"]:
                        xv.append(f["code"])
                        yv.append(f["price"])
                    fig.add_trace(go.Scatter(
                        x=xv, y=yv,
                        mode="lines+markers+text",
                        name=str(hd["date"]),
                        line=dict(color=col, width=2.5, shape="spline"),
                        marker=dict(size=7, color=col, line=dict(width=1.5, color="#0b0e14")),
                        text=[f"{v:.2f}" for v in yv],
                        textposition="top center",
                        textfont=dict(size=9, family="IBM Plex Mono"),
                    ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0e14", plot_bgcolor="#0f1319",
                    height=520, margin=dict(l=55, r=25, t=40, b=55),
                    title=dict(text=f"VIX Term Structure — {len(all_data)} dates", font=dict(size=14, color="#38bdf8", family="DM Sans"), x=0.5),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.06)", tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono")),
                    xaxis=dict(gridcolor="rgba(148,163,184,0.06)", tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94a3b8", family="IBM Plex Mono")),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for the selected dates.")

        elif go_btn:
            with st.spinner(f"Loading {hist_date}…"):
                hd = fetch_historical_structure(hist_date, n=hist_n)
            if hd and hd.get("futures"):
                fig = go.Figure()
                xv, yv = [], []
                if hd.get("vix_spot") is not None:
                    xv.append("VIX")
                    yv.append(hd["vix_spot"])
                for f in hd["futures"]:
                    xv.append(f["code"])
                    yv.append(f["price"])
                fig.add_trace(go.Scatter(
                    x=xv, y=yv,
                    mode="lines+markers+text",
                    name=str(hist_date),
                    line=dict(color="#38bdf8", width=3, shape="spline"),
                    marker=dict(size=10, color="#38bdf8", line=dict(width=2.5, color="#0b0e14")),
                    text=[f"{v:.2f}" for v in yv],
                    textposition="top center",
                    textfont=dict(size=11, color="#38bdf8", family="IBM Plex Mono"),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0e14", plot_bgcolor="#0f1319",
                    height=480, margin=dict(l=55, r=25, t=40, b=55),
                    title=dict(text=f"VIX Term Structure — {hist_date.strftime('%B %d, %Y')}", font=dict(size=14, color="#38bdf8", family="DM Sans"), x=0.5),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.06)", tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono")),
                    xaxis=dict(gridcolor="rgba(148,163,184,0.06)", tickfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono")),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"VIX Spot: {hd.get('vix_spot', '—')}")
                df_h = pd.DataFrame(hd["futures"])
                st.dataframe(df_h, use_container_width=True, hide_index=True)
            else:
                st.warning("No data available. Try a recent trading day.")

with tab_help:
    st.markdown("""
    ### How this works

    This dashboard replicates **vixcentral.com** style while sourcing the **live monthly VX curve directly from Cboe's VIX Futures page**.

    **Live curve logic**
    - Reads the VIX futures table from Cboe.
    - Filters out weekly symbols like `VX12/H6`, `VX13/J6`, etc.
    - Keeps only monthly symbols like `VX/J6`, `VX/K6`, `VX/M6`.
    - Uses `Last` when available, otherwise falls back to `Settlement`.

    **Historical logic**
    - Uses official Cboe per-contract CSV files by expiration date.
    - Builds M1, M2, M3... from the standard monthly expiration calendar.

    **Interpretation**
    - **Contango**: front months below deferred months.
    - **Backwardation**: front months above deferred months.

    **Data sources**
    - **Cboe VIX Futures page** for live VX term structure.
    - **Cboe CDN historical contract CSVs** for previous-day / historical settlements.
    - **Yahoo Finance** only for VIX spot fallback.
    """)

st.markdown("""
<div style="text-align:center; padding:1.5rem 0 0.5rem; border-top:1px solid rgba(148,163,184,0.08); margin-top:1.5rem;">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#475569;">
        VIX Term Structure Dashboard · Monthly VX curve direct from CBOE · Not financial advice
    </span>
</div>
""", unsafe_allow_html=True)
