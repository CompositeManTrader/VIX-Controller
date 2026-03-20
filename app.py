import io
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

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
    .block-container {max-width: 1380px; padding-top: 1.0rem; padding-bottom: 1rem;}
    .topbar {
        border-bottom: 1px solid rgba(78,167,255,0.18);
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
        display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
    }
    .brand {font-size: 2.0rem; font-weight: 700; color: var(--text); letter-spacing:-0.03em;}
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
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}


# ---------------- formatting ----------------
def fmt_num(x: Optional[float], digits: int = 2, default: str = "—") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return f"{float(x):,.{digits}f}"
    except Exception:
        return default


def fmt_signed(x: Optional[float], digits: int = 2, suffix: str = "") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):+,.{digits}f}{suffix}"
    except Exception:
        return "—"


def value_class(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
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


# ---------------- VX calendar ----------------
def vix_futures_expiration(year: int, month: int) -> date:
    """Wednesday 30 days before the 3rd Friday of the following month."""
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1
    first = date(next_year, next_month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    third_friday = first + timedelta(days=days_to_friday + 14)
    return third_friday - timedelta(days=30)


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
# CBOE FETCHERS
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if "symbol" in cs:
            rename[c] = "symbol"
        elif "expiration" in cs:
            rename[c] = "expiration"
        elif cs in {"last", "last price"} or "last price" in cs:
            rename[c] = "last"
        elif "change" in cs:
            rename[c] = "change"
        elif cs == "high" or cs.endswith(" high"):
            rename[c] = "high"
        elif cs == "low" or cs.endswith(" low"):
            rename[c] = "low"
        elif cs in {"settlement", "settle", "settlement price", "price", "daily settlement price"} or "settle" in cs:
            rename[c] = "settlement"
        elif "volume" in cs:
            rename[c] = "volume"
        elif cs in {"open", "opening price"}:
            rename[c] = "open"
        elif cs in {"close", "closing price"}:
            rename[c] = "close"
    return df.rename(columns=rename)


@st.cache_data(ttl=120)
def fetch_cboe_live_table() -> pd.DataFrame:
    """Scrape the official Cboe VIX Futures product page and extract the monthly VX table.
    The page contains both weeklies and monthlies; we keep only symbols like VX/J6, VX/K6, etc.
    The user confirmed via browser console that the target table headers are:
    SYMBOL, EXPIRATION, LAST, CHANGE, HIGH, LOW, SETTLEMENT, VOLUME.
    """
    url = "https://www.cboe.com/tradable-products/vix/vix-futures/"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    html = resp.text

    def finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = normalize_columns(df)
        if "symbol" not in df.columns or "expiration" not in df.columns:
            raise ValueError("La tabla scrapeada no trae symbol/expiration.")
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df[df["symbol"].apply(is_monthly_cboe_symbol)].copy()
        if df.empty:
            raise ValueError("La tabla de CBOE no trajo contratos mensuales VX.")
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        for c in ["last", "change", "high", "low", "settlement", "volume"]:
            if c in df.columns:
                df[c] = clean_numeric_series(df[c])
        return df.sort_values("expiration").reset_index(drop=True)

    # Attempt 1: parse HTML tables directly
    try:
        tables = pd.read_html(io.StringIO(html))
        candidates = []
        for t in tables:
            t2 = normalize_columns(t.copy())
            cols = [str(c).strip().lower() for c in t2.columns]
            if "symbol" in cols and "expiration" in cols:
                sym_col = [c for c in t2.columns if str(c).strip().lower() == "symbol"][0]
                symbols = t2[sym_col].astype(str)
                monthly_count = symbols.str.match(r"^VX/[FGHJKMNQUVXZ]\d$", na=False).sum()
                any_vx = symbols.str.contains("VX/", regex=False, na=False).sum()
                # Prefer the table that actually contains monthly VX rows and the expected market-data columns
                score = monthly_count * 10 + any_vx + sum(1 for x in ["last", "change", "high", "low", "settlement", "volume"] if x in cols)
                if monthly_count > 0:
                    candidates.append((score, t2))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return finalize(candidates[0][1])
    except Exception:
        pass

    # Attempt 2: manual BeautifulSoup table parse using the exact header pattern the user found in DevTools.
    soup = BeautifulSoup(html, "html.parser")
    expected = {"symbol", "expiration", "last", "change", "high", "low", "settlement", "volume"}
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        norm_headers = [h.replace(" price", "").strip() for h in headers]
        if not {"symbol", "expiration"}.issubset(set(norm_headers)):
            continue
        rows = []
        for tr in table.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if not cells:
                continue
            if any("VX/" in c for c in cells):
                rows.append(cells)
        if not rows:
            continue
        headers_for_df = norm_headers[:len(rows[0])]
        df = pd.DataFrame(rows, columns=headers_for_df)
        try:
            return finalize(df)
        except Exception:
            continue

    # Attempt 3: fallback to text block from the rendered page.
    TEMPPLACEHOLDER
    marker = "Symbol Expiration Last Price Change High Low Settlement Volume"
    if marker not in text:
        marker = "Symbol Expiration Last Change High Low Settlement Volume"
    if marker in text:
        section = text.split(marker, 1)[1]
        lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
        rows = []
        row_regex = re.compile(r"^(VX\d{0,2}/[FGHJKMNQUVXZ]\d)\s+(\d{2}/\d{2}/\d{4})(.*)$")
        for line in lines:
            if line.startswith("# ") or line.startswith("The Next Generation"):
                break
            m = row_regex.match(line)
            if not m:
                continue
            sym, exp, tail = m.groups()
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
            row = {"symbol": sym, "expiration": exp}
            keys = ["last", "change", "high", "low", "settlement", "volume"]
            for k, v in zip(keys, nums):
                row[k] = v
            rows.append(row)
        if rows:
            return finalize(pd.DataFrame(rows))

    raise ValueError("No se pudo scrapear la tabla de VIX Futures en CBOE usando table scraping ni text parsing.")

@st.cache_data(ttl=300)
def fetch_contract_daily_history(expiration: str) -> pd.DataFrame:
    url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/products/csv/VX/VX_{expiration}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = normalize_columns(df)
    # Normalize common headers from product CSVs
    extra_map = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs == "trade date":
            extra_map[c] = "trade_date"
        elif cs == "open":
            extra_map[c] = "open"
        elif cs == "high":
            extra_map[c] = "high"
        elif cs == "low":
            extra_map[c] = "low"
        elif cs == "close":
            extra_map[c] = "close"
        elif cs in {"last", "last sale"}:
            extra_map[c] = "last"
        elif "settle" in cs or cs == "price":
            extra_map[c] = "settlement"
        elif "volume" in cs:
            extra_map[c] = "volume"
    df = df.rename(columns=extra_map)
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
def build_monthly_curve(n_months: int = 8) -> pd.DataFrame:
    contracts = active_monthly_contracts(n=n_months)
    live = fetch_cboe_live_table()
    live = normalize_columns(live)
    if "symbol" not in live.columns:
        raise ValueError("CBOE live table no trajo columna de símbolo.")
    live["symbol"] = live["symbol"].astype(str).str.strip()
    live = live[live["symbol"].apply(is_monthly_cboe_symbol)].copy()
    if "expiration" in live.columns:
        live["expiration"] = pd.to_datetime(live["expiration"], errors="coerce")

    live_map = {row["symbol"]: row.to_dict() for _, row in live.iterrows()}

    rows = []
    for idx, c in enumerate(contracts, start=1):
        hist_df = fetch_contract_daily_history(c["exp"].strftime("%Y-%m-%d"))
        if hist_df.empty:
            hist_row = {}
        else:
            hist_df = hist_df.sort_values("trade_date") if "trade_date" in hist_df.columns else hist_df
            hist_row = hist_df.iloc[-1].to_dict()
        lrow = live_map.get(c["slash_symbol"], {})

        close_px = hist_row.get("close")
        last_px = lrow.get("last")
        if pd.isna(last_px) or last_px in (None, 0):
            last_px = close_px
        settlement_px = lrow.get("settlement") if pd.notna(lrow.get("settlement")) else hist_row.get("settlement")
        open_px = hist_row.get("open")
        high_px = lrow.get("high") if pd.notna(lrow.get("high")) else hist_row.get("high")
        low_px = lrow.get("low") if pd.notna(lrow.get("low")) else hist_row.get("low")
        volume_px = lrow.get("volume") if pd.notna(lrow.get("volume")) else hist_row.get("volume")
        prev_close = None
        if isinstance(hist_df, pd.DataFrame) and not hist_df.empty and len(hist_df) >= 2 and "close" in hist_df.columns:
            prev_close = hist_df["close"].iloc[-2]
        change_px = None
        if pd.notna(lrow.get("change")):
            change_px = lrow.get("change")
        elif pd.notna(last_px) and pd.notna(prev_close):
            change_px = float(last_px) - float(prev_close)

        rows.append(
            {
                "m": f"M{idx}",
                "label": c["label"],
                "symbol": c["slash_symbol"],
                "expiration": c["exp"],
                "dte": c["dte"],
                "last": last_px,
                "close": close_px,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "settlement": settlement_px,
                "change": change_px,
                "volume": volume_px,
                "prev_close": prev_close,
                "source_last": "CBOE web scraping" if pd.notna(lrow.get("last")) else "CBOE contract CSV fallback",
                "source_ohlc": "CBOE contract CSV + live high/low/volume when available",
            }
        )

    curve = pd.DataFrame(rows)
    curve["term_price"] = curve["last"].where(curve["last"].notna(), curve["close"])
    curve["contango_pct_vs_prev"] = (curve["term_price"] / curve["term_price"].shift(1) - 1) * 100
    curve["difference_vs_prev"] = curve["term_price"] - curve["term_price"].shift(1)
    return curve


# =========================================================
# STRATEGY
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
    prev = data.iloc[-2] if len(data) >= 2 else last
    m1 = curve.iloc[0]
    m2 = curve.iloc[1] if len(curve) > 1 else None

    contango_pct = None
    in_contango = False
    if m2 is not None and pd.notna(m1["term_price"]) and pd.notna(m2["term_price"]):
        contango_pct = (float(m2["term_price"]) / float(m1["term_price"]) - 1) * 100
        in_contango = float(m2["term_price"]) > float(m1["term_price"])

    bb_sig = int(last["bb_sig"])
    vxx_below_sma = float(last["VXX_Close"]) < float(last["SMA20"])
    vxx_above_upper = float(last["VXX_Close"]) > float(last["BB_Upper"])
    final_signal = int(bb_sig and in_contango)

    exec_date = pd.Timestamp(data.index[-1]).to_pydatetime().date() + timedelta(days=1)
    while exec_date.weekday() >= 5:
        exec_date += timedelta(days=1)

    return {
        "data": data,
        "last": last,
        "prev": prev,
        "m1": m1,
        "m2": m2,
        "contango_pct": contango_pct,
        "in_contango": in_contango,
        "bb_sig": bb_sig,
        "final_signal": final_signal,
        "vxx_below_sma": vxx_below_sma,
        "vxx_above_upper": vxx_above_upper,
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
            name="Last / Term Price",
            line=dict(color="#39a0ff", width=3.2, shape="spline"),
            marker=dict(size=8, color="#39a0ff"),
            hovertemplate="<b>%{x}</b><br>Term Price: %{y:.2f}<extra></extra>",
        )
    )

    series_meta = [
        ("close", "Close", "#A8B3C7", "dot"),
        ("open", "Open", "#F2C14E", "dot"),
        ("high", "High", "#4CD7A2", "dot"),
        ("low", "Low", "#FF7F7F", "dot"),
    ]
    for col, name, color, dash in series_meta:
        if col in curve.columns and curve[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=curve[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=1.6, dash=dash),
                    marker=dict(size=5, color=color),
                    visible="legendonly" if name != "Close" else True,
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
        xaxis=dict(
            title="Future Month",
            gridcolor="rgba(168,179,199,0.10)",
            zeroline=False,
            showline=True,
            linecolor="rgba(168,179,199,0.18)",
        ),
        yaxis=dict(
            title="Volatility",
            gridcolor="rgba(168,179,199,0.10)",
            zeroline=False,
            showline=True,
            linecolor="rgba(168,179,199,0.18)",
        ),
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
# UI
# =========================================================
curve_error = None
spot_error = None
strategy_error = None
curve = None
spot = None
strategy = None

try:
    curve = build_monthly_curve(n_months=8)
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
    <div class="micro">{now_txt} · Monthly curve: CBOE official VX contract history CSVs · Settlement overlay: CBOE settlement CSV · Spot: Yahoo fallback</div>
</div>
""",
    unsafe_allow_html=True,
)

if curve is not None:
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
    st.markdown(f'<div class="warnbox"><b>No se pudo construir la curva mensual de VIX desde CBOE.</b><br>Detalle técnico: {curve_error}</div>', unsafe_allow_html=True)

if spot_error:
    st.markdown(f'<div class="warnbox"><b>No se pudo cargar VIX spot.</b><br>Detalle técnico: {spot_error}</div>', unsafe_allow_html=True)

term_tab, strategy_tab, diag_tab = st.tabs(["Term Structure", "Strategy Monitor", "Raw Data & Diagnostics"])

with term_tab:
    if curve is not None:
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
            st.markdown('<div class="panel-title">Monthly VX quote panel</div><div class="panel-sub">M1, M2 and the full curve are built from the nearest monthly VX expirations. The displayed term price uses CBOE LAST first, then CLOSE, then SETTLEMENT. OHLC columns come from the official CBOE contract history files.</div>', unsafe_allow_html=True)
            display = curve[["m", "label", "symbol", "expiration", "dte", "term_price", "last", "close", "open", "high", "low", "settlement", "change", "volume"]].copy()
            display["expiration"] = pd.to_datetime(display["expiration"]).dt.strftime("%Y-%m-%d")
            for c in ["term_price", "last", "close", "open", "high", "low", "settlement", "change", "volume"]:
                if c == "change":
                    display[c] = display[c].map(fmt_signed)
                elif c == "volume":
                    display[c] = display[c].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
                else:
                    display[c] = display[c].map(fmt_num)
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
            rows_html = f"""
            <table class="rule-table">
                <thead><tr><th>Component</th><th>Reading</th><th>Status</th></tr></thead>
                <tbody>
                    <tr><td>BB Timing</td><td>VXX &lt; SMA20</td><td class="{'pos' if strategy['bb_sig'] else 'neg'}">{'LONG' if strategy['bb_sig'] else 'CASH'}</td></tr>
                    <tr><td>Contango Filter</td><td>M2 &gt; M1</td><td class="{'pos' if strategy['in_contango'] else 'neg'}">{fmt_signed(strategy['contango_pct'], suffix='%')}</td></tr>
                    <tr><td>Final Signal</td><td>BB × Contango</td><td class="{'pos' if strategy['final_signal'] else 'neg'}">{'LONG' if strategy['final_signal'] else 'CASH'}</td></tr>
                    <tr><td>Execution</td><td>Next session</td><td>{strategy['exec_date'].strftime('%Y-%m-%d')}</td></tr>
                </tbody>
            </table>
            """
            st.markdown(rows_html, unsafe_allow_html=True)

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
                    ["SVXY Close", fmt_num(s_last['SVXY_Close'])],
                    ["SVIX Close", fmt_num(s_last['SVIX_Close'])],
                    ["SPY Close", fmt_num(s_last['SPY_Close'])],
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
    st.markdown('<div class="panel-title">Diagnostics</div><div class="panel-sub">Raw tables used by the dashboard. This section is useful for validating whether the monthly curve is being sourced and merged correctly.</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        try:
            live_df = fetch_cboe_live_table().copy()
            if "expiration" in live_df.columns:
                live_df["expiration"] = pd.to_datetime(live_df["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.markdown("**CBOE live table (raw filtered monthly rows)**")
            st.dataframe(live_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Live table error: {e}")
    with d2:
        if curve is not None:
            st.markdown("**Merged monthly curve used by the app**")
            dbg = curve.copy()
            dbg["expiration"] = pd.to_datetime(dbg["expiration"]).dt.strftime("%Y-%m-%d")
            st.dataframe(dbg, use_container_width=True, hide_index=True)

st.caption("Monthly curve uses web scraping from the official CBOE VIX Futures product page for live monthly VX rows and CBOE individual contract history CSVs for close/open plus fallback values. Strategy logic adapted from your uploaded Monitor_Operativo_v3 notebook.")
