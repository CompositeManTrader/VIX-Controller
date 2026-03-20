import io
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="VX Term Structure Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
    :root {
        --bg:#07111f;
        --panel:#0c1829;
        --panel-2:#0e2138;
        --line:#1d3557;
        --text:#e8eef8;
        --muted:#90a4c3;
        --accent:#4ea7ff;
        --green:#38c172;
        --red:#ff6b6b;
        --amber:#e6b450;
    }
    .stApp {background: linear-gradient(180deg,#06101c 0%, #081524 100%);}
    #MainMenu, footer, header {visibility:hidden;}
    .block-container {max-width: 1400px; padding-top: 1rem; padding-bottom: 1rem;}
    .topbar {
        border-bottom: 1px solid rgba(78,167,255,0.18);
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
        display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
    }
    .brand {font-size: 2rem; font-weight: 700; color: var(--text); letter-spacing:-0.03em;}
    .brand span {color: var(--accent);}
    .subline {color: var(--muted); font-size: 0.86rem;}
    .micro {color: var(--muted); font-size: 0.76rem;}
    .kpi-grid {display:grid; grid-template-columns: repeat(6, minmax(140px,1fr)); gap: 0.7rem; margin: 0.6rem 0 1rem;}
    .kpi-card {
        background: linear-gradient(180deg, rgba(12,24,41,0.96) 0%, rgba(9,19,33,0.96) 100%);
        border: 1px solid rgba(78,167,255,0.16);
        border-radius: 14px; padding: 0.8rem 1rem; min-height: 92px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    }
    .kpi-label {color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;}
    .kpi-value {color: var(--text); font-size: 1.55rem; font-weight: 700; margin-top: 0.25rem;}
    .kpi-sub {color: var(--muted); font-size: 0.78rem; margin-top: 0.22rem;}
    .pos {color: var(--green)!important;}
    .neg {color: var(--red)!important;}
    .warnbox {
        background: rgba(230,180,80,0.12);
        border: 1px solid rgba(230,180,80,0.28);
        color: #f6d98f; border-radius: 12px; padding: 0.9rem 1rem; margin: 0.6rem 0 1rem;
    }
    .panel-title {font-size: 1rem; color: var(--text); font-weight: 700; margin-bottom: 0.2rem;}
    .panel-sub {font-size: 0.78rem; color: var(--muted); margin-bottom: 0.8rem;}
    .signal-box {
        background: linear-gradient(180deg, rgba(12,24,41,0.96) 0%, rgba(9,19,33,0.96) 100%);
        border:1px solid rgba(78,167,255,0.16); border-radius: 16px; padding:1rem 1rem 0.8rem;
    }
    .signal-banner {
        border-radius: 14px; padding: 1rem 1.2rem; text-align:center; font-size: 2rem; font-weight:800;
        margin-bottom: 0.8rem; letter-spacing: 0.06em;
    }
    .banner-long {background: rgba(56,193,114,0.14); color: #7ee2a6; border:1px solid rgba(56,193,114,0.30);}
    .banner-cash {background: rgba(255,107,107,0.14); color: #ff9a9a; border:1px solid rgba(255,107,107,0.30);}
    .rule-table {width:100%; border-collapse:collapse; font-size:0.84rem;}
    .rule-table th, .rule-table td {padding: 0.52rem 0.55rem; border-bottom:1px solid rgba(144,164,195,0.12);}
    .rule-table th {text-align:left; color:var(--muted); font-weight:600; font-size:0.76rem; text-transform:uppercase; letter-spacing:0.08em;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS / HELPERS
# =========================================================
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
MONTH_NAMES_SHORT = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}


def fmt_num(x: Optional[float], digits: int = 2, default: str = "—") -> str:
    try:
        if x is None or pd.isna(x):
            return default
        return f"{float(x):,.{digits}f}"
    except Exception:
        return default


def fmt_signed(x: Optional[float], digits: int = 2, suffix: str = "") -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):+,.{digits}f}{suffix}"
    except Exception:
        return "—"


def value_class(x: Optional[float]) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
        if float(x) > 0:
            return "pos"
        if float(x) < 0:
            return "neg"
        return ""
    except Exception:
        return ""


def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _safe_float(v) -> float:
    try:
        f = float(str(v).replace(",", ""))
        return round(f, 4)
    except (ValueError, TypeError):
        return np.nan


def _safe_int(v) -> int:
    try:
        return int(float(str(v).replace(",", "")))
    except (ValueError, TypeError):
        return 0


# =========================================================
# VX CALENDAR
# =========================================================
def vix_futures_expiration(year: int, month: int) -> date:
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1
    first = date(next_year, next_month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    third_friday = first + timedelta(days=days_to_friday + 14)
    return third_friday - timedelta(days=30)


@st.cache_data(ttl=3600)
def active_monthly_contracts(ref: Optional[date] = None, n: int = 8) -> List[dict]:
    ref = ref or date.today()
    out: List[dict] = []
    month = ref.month
    year = ref.year
    for i in range(n + 6):
        cm = ((month - 1 + i) % 12) + 1
        cy = year + ((month - 1 + i) // 12)
        exp = vix_futures_expiration(cy, cm)
        if exp >= ref:
            code = MONTH_CODES[cm]
            out.append(
                {
                    "month": cm,
                    "year": cy,
                    "exp": exp,
                    "dte": (exp - ref).days,
                    "label": f"{MONTH_NAMES_SHORT[cm]} {str(cy)[-2:]}",
                    "slash_symbol": f"VX/{code}{str(cy)[-1]}",
                    "legacy_symbol": f"VX{code}{str(cy)[-2:]}",
                }
            )
        if len(out) >= n:
            break
    return out


def is_monthly_cboe_symbol(symbol: str) -> bool:
    if not isinstance(symbol, str):
        return False
    symbol = symbol.strip().upper()
    return re.fullmatch(r"VX/[FGHJKMNQUVXZ]\d", symbol) is not None


# =========================================================
# DATA FETCHERS
# =========================================================
@st.cache_data(ttl=55)
def fetch_cboe_delayed_quotes() -> Tuple[Dict[str, dict], List[List[str]], str]:
    """
    Scrape VX futures from CBOE delayed quotes page.
    This is the same source logic the user indicated was working correctly.
    """
    results: Dict[str, dict] = {}
    raw_rows: List[List[str]] = []
    source_used = ""
    url = "https://www.cboe.com/delayed_quotes/futures/future_quotes"

    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        resp = session.get(url, timeout=25)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        vx_table = None
        for tbl in tables:
            if tbl.find(string=lambda t: t and "VX/" in t):
                vx_table = tbl
                break

        if vx_table:
            idx = {"symbol": 0, "expiration": 1, "last": 2, "change": 3, "high": 4, "low": 5, "settlement": 6, "volume": 7}
            tbody = vx_table.find("tbody")
            if tbody:
                for tr in tbody.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cells) < 7:
                        continue
                    symbol = cells[idx["symbol"]]
                    if not is_monthly_cboe_symbol(symbol):
                        continue

                    raw_rows.append(cells)
                    last_val = _safe_float(cells[idx["last"]])
                    settle_val = _safe_float(cells[idx["settlement"]])
                    price_val = last_val if pd.notna(last_val) and last_val != 0 else settle_val
                    results[symbol] = {
                        "price": price_val,
                        "last": last_val,
                        "settlement": settle_val,
                        "change": _safe_float(cells[idx["change"]]),
                        "high": _safe_float(cells[idx["high"]]),
                        "low": _safe_float(cells[idx["low"]]),
                        "volume": _safe_int(cells[idx["volume"]]) if len(cells) > 7 else 0,
                        "expiration_raw": cells[idx["expiration"]],
                        "src": "CBOE delayed quotes",
                    }

        if results:
            source_used = "CBOE delayed quotes"
    except Exception:
        pass

    return results, raw_rows, source_used


@st.cache_data(ttl=300)
def fetch_contract_daily_history(expiration: str) -> pd.DataFrame:
    url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{expiration}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    rename = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs == "trade date":
            rename[c] = "trade_date"
        elif cs == "open":
            rename[c] = "open"
        elif cs == "high":
            rename[c] = "high"
        elif cs == "low":
            rename[c] = "low"
        elif cs == "close":
            rename[c] = "close"
        elif cs in {"last", "last sale"}:
            rename[c] = "last"
        elif "settle" in cs or cs == "price":
            rename[c] = "settlement"
        elif "volume" in cs:
            rename[c] = "volume"
    df = df.rename(columns=rename)

    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for c in ["open", "high", "low", "close", "last", "settlement", "volume"]:
        if c in df.columns:
            df[c] = clean_numeric_series(df[c])
    return df


@st.cache_data(ttl=120)
def fetch_vix_spot() -> Optional[dict]:
    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="7d", auto_adjust=False)
        if hist.empty:
            return None
        latest = hist.iloc[-1]
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else float(latest["Close"])
        return {
            "last": float(latest["Close"]),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "prev_close": prev_close,
            "change": float(latest["Close"]) - prev_close,
            "pct_change": ((float(latest["Close"]) / prev_close) - 1) * 100 if prev_close else None,
            "timestamp": hist.index[-1],
        }
    except Exception:
        return None


@st.cache_data(ttl=600)
def build_monthly_curve(n_months: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    contracts = active_monthly_contracts(n=n_months)
    live_map, raw_rows, source_used = fetch_cboe_delayed_quotes()

    rows = []
    for idx, c in enumerate(contracts, start=1):
        hist_df = fetch_contract_daily_history(c["exp"].strftime("%Y-%m-%d"))
        hist_df = hist_df.sort_values("trade_date") if (not hist_df.empty and "trade_date" in hist_df.columns) else hist_df
        hist_row = hist_df.iloc[-1].to_dict() if not hist_df.empty else {}
        prev_close = hist_df["close"].iloc[-2] if (not hist_df.empty and len(hist_df) >= 2 and "close" in hist_df.columns) else np.nan

        live_row = live_map.get(c["slash_symbol"], {})
        last_px = live_row.get("last")
        if pd.isna(last_px) or last_px == 0:
            csv_last = hist_row.get("last")
            last_px = csv_last if pd.notna(csv_last) and csv_last != 0 else np.nan

        close_px = hist_row.get("close")
        settlement_px = live_row.get("settlement")
        if pd.isna(settlement_px) or settlement_px == 0:
            settlement_px = hist_row.get("settlement")
        open_px = hist_row.get("open")
        high_px = live_row.get("high")
        if pd.isna(high_px) or high_px == 0:
            high_px = hist_row.get("high")
        low_px = live_row.get("low")
        if pd.isna(low_px) or low_px == 0:
            low_px = hist_row.get("low")
        change_px = live_row.get("change")
        if pd.isna(change_px) and pd.notna(last_px) and pd.notna(prev_close):
            change_px = float(last_px) - float(prev_close)
        volume_px = live_row.get("volume")
        if not volume_px:
            volume_px = hist_row.get("volume")

        term_price = last_px
        if pd.isna(term_price) or term_price == 0:
            term_price = close_px
        if pd.isna(term_price) or term_price == 0:
            term_price = settlement_px

        rows.append(
            {
                "m": f"M{idx}",
                "label": c["label"],
                "symbol": c["slash_symbol"],
                "expiration": c["exp"],
                "dte": c["dte"],
                "term_price": term_price,
                "last": last_px,
                "close": close_px,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "settlement": settlement_px,
                "change": change_px,
                "prev_close": prev_close,
                "volume": volume_px,
                "source_last": "CBOE delayed quotes" if c["slash_symbol"] in live_map else "CBOE contract CSV fallback",
                "source_ohlc": "CBOE contract CSV",
            }
        )

    curve = pd.DataFrame(rows)
    curve["contango_pct_vs_prev"] = (curve["term_price"] / curve["term_price"].shift(1) - 1) * 100
    curve["difference_vs_prev"] = curve["term_price"] - curve["term_price"].shift(1)

    raw_df = pd.DataFrame(
        raw_rows,
        columns=["Symbol", "Expiration", "Last", "Change", "High", "Low", "Settlement", "Volume"] if raw_rows else None,
    )
    return curve, raw_df, source_used


# =========================================================
# STRATEGY LAYER
# =========================================================
@st.cache_data(ttl=300)
def fetch_strategy_market_data(days: int = 320) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    symbols = {"VXX": "VXX", "SVXY": "SVXY", "SVIX": "SVIX", "VIX": "^VIX", "SPY": "SPY"}
    out = pd.DataFrame()
    for alias, symbol in symbols.items():
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            continue
        out[f"{alias}_Open"] = df["Open"]
        out[f"{alias}_High"] = df["High"]
        out[f"{alias}_Low"] = df["Low"]
        out[f"{alias}_Close"] = df["Close"]
    out = out.sort_index().dropna(subset=["VXX_Close"], how="any")
    out["SMA20"] = out["VXX_Close"].rolling(20).mean()
    out["STD20"] = out["VXX_Close"].rolling(20).std()
    out["BB_Upper"] = out["SMA20"] + 2.0 * out["STD20"]
    out["BB_Lower"] = out["SMA20"] - 2.0 * out["STD20"]

    clean = out.dropna(subset=["SMA20", "BB_Upper"]).copy()
    pos = 0
    state = []
    for _, row in clean.iterrows():
        p = row["VXX_Close"]
        sma = row["SMA20"]
        upper = row["BB_Upper"]
        if pos == 0 and p < sma:
            pos = 1
        elif pos == 1 and p > upper:
            pos = 0
        state.append(pos)
    clean["bb_sig"] = state
    return clean


def build_strategy_snapshot(curve: pd.DataFrame) -> dict:
    data = fetch_strategy_market_data()
    last = data.iloc[-1]
    m1 = curve.iloc[0]
    m2 = curve.iloc[1] if len(curve) > 1 else None

    contango_pct = None
    in_contango = False
    if m2 is not None and pd.notna(m1["term_price"]) and pd.notna(m2["term_price"]):
        contango_pct = (float(m2["term_price"]) / float(m1["term_price"]) - 1) * 100
        in_contango = float(m2["term_price"]) > float(m1["term_price"])

    bb_sig = int(last["bb_sig"])
    final_signal = int(bb_sig and in_contango)

    exec_date = pd.Timestamp(data.index[-1]).to_pydatetime().date() + timedelta(days=1)
    while exec_date.weekday() >= 5:
        exec_date += timedelta(days=1)

    return {
        "data": data,
        "last": last,
        "m1": m1,
        "m2": m2,
        "contango_pct": contango_pct,
        "in_contango": in_contango,
        "bb_sig": bb_sig,
        "final_signal": final_signal,
        "pct_to_sma": (float(last["VXX_Close"]) / float(last["SMA20"]) - 1) * 100,
        "pct_to_upper": (float(last["VXX_Close"]) / float(last["BB_Upper"]) - 1) * 100,
        "exec_date": exec_date,
    }


# =========================================================
# CHARTS
# =========================================================
def build_term_structure_chart(curve: pd.DataFrame, spot: Optional[dict]) -> go.Figure:
    fig = go.Figure()
    x = curve["label"].tolist()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=curve["term_price"],
            mode="lines+markers+text",
            text=[fmt_num(v, 2) for v in curve["term_price"]],
            textposition="top center",
            name="Term Price",
            line=dict(color="#39a0ff", width=3.2, shape="spline"),
            marker=dict(size=8, color="#39a0ff"),
            hovertemplate="<b>%{x}</b><br>Term Price: %{y:.2f}<extra></extra>",
        )
    )

    for col, name, color in [
        ("last", "Last", "#4ea7ff"),
        ("close", "Close", "#cbd5e1"),
        ("open", "Open", "#F2C14E"),
        ("high", "High", "#4CD7A2"),
        ("low", "Low", "#FF7F7F"),
        ("settlement", "Settlement", "#8b9bb4"),
    ]:
        if col in curve.columns and curve[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=curve[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(width=1.5, dash="dot", color=color),
                    marker=dict(size=5, color=color),
                    visible=True if name in {"Close", "Settlement"} else "legendonly",
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}}<extra></extra>",
                )
            )

    if spot and spot.get("last") is not None:
        fig.add_hline(
            y=float(spot["last"]),
            line_width=2,
            line_color="#6DD36F",
            line_dash="dash",
            annotation_text=f"VIX Spot {spot['last']:.2f}",
            annotation_position="top right",
            annotation_font_color="#6DD36F",
        )

    fig.update_layout(
        title=dict(text="VIX Futures Term Structure", x=0.5, font=dict(size=26, color="#F3F7FF")),
        paper_bgcolor="#081321",
        plot_bgcolor="#0A182A",
        font=dict(color="#E6EEF9"),
        height=560,
        margin=dict(l=40, r=30, t=80, b=55),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(8,19,33,0.7)",
            bordercolor="rgba(61,165,255,0.15)",
            borderwidth=1,
        ),
        xaxis=dict(title="Future Month", gridcolor="rgba(168,179,199,0.10)", zeroline=False),
        yaxis=dict(title="Volatility", gridcolor="rgba(168,179,199,0.10)", zeroline=False),
        hovermode="x unified",
    )
    return fig


def build_vxx_timing_chart(strategy: dict) -> go.Figure:
    data = strategy["data"].tail(120).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["VXX_Close"], mode="lines", name="VXX Close", line=dict(color="#4ea7ff", width=2.4)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20", line=dict(color="#c0cad8", width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="#ff8a80", width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower", line=dict(color="#56d364", width=1.2, dash="dot")))
    fig.update_layout(
        height=420,
        paper_bgcolor="#0b1728",
        plot_bgcolor="#0b1728",
        font=dict(color="#d7e4f5"),
        margin=dict(l=30, r=20, t=45, b=30),
        title=dict(text="VXX Timing Model — Bollinger Bands (20, 2σ)", x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(144,164,195,0.08)"),
        yaxis=dict(gridcolor="rgba(144,164,195,0.08)", title="Price"),
        hovermode="x unified",
    )
    return fig


# =========================================================
# APP
# =========================================================
curve_error = None
spot_error = None
strategy_error = None
curve = None
spot = None
strategy = None
raw_live = pd.DataFrame()
curve_source = ""

try:
    curve, raw_live, curve_source = build_monthly_curve(n_months=8)
except Exception as e:
    curve_error = str(e)

try:
    spot = fetch_vix_spot()
except Exception as e:
    spot_error = str(e)

if curve is not None:
    try:
        strategy = build_strategy_snapshot(curve)
    except Exception as e:
        strategy_error = str(e)

now_txt = datetime.now().strftime("%B %d, %Y · %H:%M")
st.markdown(
    f"""
<div class="topbar">
    <div>
        <div class="brand">VX <span>Term Structure Monitor</span></div>
        <div class="subline">Institutional dashboard for monthly VIX futures curve, regime diagnostics and short-vol execution monitor.</div>
    </div>
    <div class="micro">{now_txt} · Live curve source: {curve_source or '—'} · OHLC source: CBOE contract history CSVs · Spot: Yahoo</div>
</div>
""",
    unsafe_allow_html=True,
)

if curve is not None and not curve.empty:
    m1 = curve.iloc[0]
    m2 = curve.iloc[1] if len(curve) > 1 else None
    spot_last = spot.get("last") if spot else None
    m1m2_pct = ((float(m2['term_price']) / float(m1['term_price']) - 1) * 100) if m2 is not None and pd.notna(m1['term_price']) and pd.notna(m2['term_price']) else None
    vix_m1 = (float(m1['term_price']) - float(spot_last)) if (spot_last is not None and pd.notna(m1['term_price'])) else None
    total_curve = ((float(curve.iloc[-1]['term_price']) / float(m1['term_price']) - 1) * 100) if len(curve) > 1 and pd.notna(curve.iloc[-1]['term_price']) and pd.notna(m1['term_price']) else None

    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">VIX Spot</div><div class="kpi-value">{fmt_num(spot_last)}</div><div class="kpi-sub {value_class(spot.get('change') if spot else None)}">{fmt_signed(spot.get('change') if spot else None)} ({fmt_signed(spot.get('pct_change') if spot else None, suffix='%')})</div></div>
  <div class="kpi-card"><div class="kpi-label">M1 · {m1['dte']} DTE</div><div class="kpi-value">{fmt_num(m1['term_price'])}</div><div class="kpi-sub">{m1['symbol']} · Last source: {m1['source_last']}</div></div>
  <div class="kpi-card"><div class="kpi-label">M2</div><div class="kpi-value">{fmt_num(m2['term_price'] if m2 is not None else None)}</div><div class="kpi-sub">{m2['symbol'] if m2 is not None else '—'}</div></div>
  <div class="kpi-card"><div class="kpi-label">VIX ↔ M1 basis</div><div class="kpi-value {value_class(vix_m1)}">{fmt_signed(vix_m1)}</div><div class="kpi-sub">M1 minus spot</div></div>
  <div class="kpi-card"><div class="kpi-label">M1 → M2 contango</div><div class="kpi-value {value_class(m1m2_pct)}">{fmt_signed(m1m2_pct, suffix='%')}</div><div class="kpi-sub">Positive = contango</div></div>
  <div class="kpi-card"><div class="kpi-label">M1 → M8 total curve</div><div class="kpi-value {value_class(total_curve)}">{fmt_signed(total_curve, suffix='%')}</div><div class="kpi-sub">Back-end slope</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

if curve_error:
    st.markdown(f'<div class="warnbox"><b>No se pudo construir la curva mensual de VIX.</b><br>Detalle técnico: {curve_error}</div>', unsafe_allow_html=True)
if spot_error:
    st.markdown(f'<div class="warnbox"><b>No se pudo cargar VIX spot.</b><br>Detalle técnico: {spot_error}</div>', unsafe_allow_html=True)

term_tab, strategy_tab, diag_tab = st.tabs(["Term Structure", "Strategy Monitor", "Raw Data & Diagnostics"])

with term_tab:
    if curve is not None and not curve.empty:
        col_chart, col_table = st.columns([2.0, 1.15], gap="large")
        with col_chart:
            st.plotly_chart(build_term_structure_chart(curve, spot), use_container_width=True)
            table_contango = pd.DataFrame(
                {
                    "Month": curve["m"],
                    "% Contango vs Prev": curve["contango_pct_vs_prev"].map(lambda x: fmt_signed(x, suffix="%")),
                    "Difference vs Prev": curve["difference_vs_prev"].map(fmt_signed),
                }
            )
            st.dataframe(table_contango, use_container_width=True, hide_index=True)
        with col_table:
            st.markdown('<div class="panel-title">Monthly VX quote panel</div><div class="panel-sub">The displayed term price uses CBOE LAST first, then CLOSE, then SETTLEMENT. Open/High/Low/Close come from the official CBOE contract history files. M1 is always the nearest active monthly VX future.</div>', unsafe_allow_html=True)
            display = curve[["m", "label", "symbol", "expiration", "dte", "term_price", "last", "close", "open", "high", "low", "settlement", "change", "volume"]].copy()
            display["expiration"] = pd.to_datetime(display["expiration"]).dt.strftime("%Y-%m-%d")
            for c in ["term_price", "last", "close", "open", "high", "low", "settlement"]:
                display[c] = display[c].map(fmt_num)
            display["change"] = display["change"].map(fmt_signed)
            display["volume"] = display["volume"].map(lambda x: "—" if pd.isna(x) else f"{int(x):,}")
            st.dataframe(display, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("Sin curva disponible.")

with strategy_tab:
    if strategy is not None:
        banner_class = "banner-long" if strategy["final_signal"] else "banner-cash"
        banner_text = "LONG" if strategy["final_signal"] else "CASH"
        st.markdown('<div class="panel-title">Short-vol operating monitor</div><div class="panel-sub">Rules adapted from your notebook: timing via Bollinger Bands on VXX, filtered by live M1/M2 contango from the current CBOE monthly curve.</div>', unsafe_allow_html=True)
        left, right = st.columns([1.05, 1.65], gap="large")
        with left:
            st.markdown('<div class="signal-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="signal-banner {banner_class}">{banner_text}</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <table class="rule-table">
                    <thead><tr><th>Component</th><th>Reading</th><th>Status</th></tr></thead>
                    <tbody>
                        <tr><td>BB Timing</td><td>VXX &lt; SMA20</td><td class="{'pos' if strategy['bb_sig'] else 'neg'}">{'LONG' if strategy['bb_sig'] else 'CASH'}</td></tr>
                        <tr><td>Contango Filter</td><td>M2 &gt; M1</td><td class="{'pos' if strategy['in_contango'] else 'neg'}">{fmt_signed(strategy['contango_pct'], suffix='%')}</td></tr>
                        <tr><td>Final Signal</td><td>BB × Contango</td><td class="{'pos' if strategy['final_signal'] else 'neg'}">{'LONG' if strategy['final_signal'] else 'CASH'}</td></tr>
                        <tr><td>Execution</td><td>Next session</td><td>{strategy['exec_date'].strftime('%Y-%m-%d')}</td></tr>
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
            s_last = strategy["last"]
            detail = pd.DataFrame(
                [
                    ["VXX Close", fmt_num(s_last['VXX_Close'])],
                    ["SMA20", fmt_num(s_last['SMA20'])],
                    ["BB Upper", fmt_num(s_last['BB_Upper'])],
                    ["Distance vs SMA", fmt_signed(strategy['pct_to_sma'], suffix='%')],
                    ["Distance vs Upper BB", fmt_signed(strategy['pct_to_upper'], suffix='%')],
                    ["M1", f"{strategy['m1']['symbol']} · {fmt_num(strategy['m1']['term_price'])}"],
                    ["M2", f"{strategy['m2']['symbol']} · {fmt_num(strategy['m2']['term_price'])}" if strategy['m2'] is not None else "—"],
                    ["SVXY Close", fmt_num(s_last.get('SVXY_Close'))],
                    ["SVIX Close", fmt_num(s_last.get('SVIX_Close'))],
                    ["SPY Close", fmt_num(s_last.get('SPY_Close'))],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(detail, use_container_width=True, hide_index=True, height=360)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.plotly_chart(build_vxx_timing_chart(strategy), use_container_width=True)
    else:
        if strategy_error:
            st.markdown(f'<div class="warnbox"><b>No se pudo construir el monitor operativo.</b><br>Detalle técnico: {strategy_error}</div>', unsafe_allow_html=True)
        else:
            st.info("Monitor operativo no disponible.")

with diag_tab:
    st.markdown('<div class="panel-title">Diagnostics</div><div class="panel-sub">Raw delayed-quotes rows and the merged monthly curve actually used by the dashboard.</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**CBOE delayed quotes rows (monthly only)**")
        if raw_live is not None and not raw_live.empty:
            st.dataframe(raw_live, use_container_width=True, hide_index=True)
        else:
            st.info("No raw delayed-quote rows captured.")
    with d2:
        st.markdown("**Merged monthly curve used by the app**")
        if curve is not None and not curve.empty:
            dbg = curve.copy()
            dbg["expiration"] = pd.to_datetime(dbg["expiration"]).dt.strftime("%Y-%m-%d")
            st.dataframe(dbg, use_container_width=True, hide_index=True)
        else:
            st.info("No merged curve available.")

st.caption("Merged version: live VX monthly prices from the same delayed-quotes logic that was already working in your prior app, plus the institutional layout and strategy monitor from app_institutional_v2.")
