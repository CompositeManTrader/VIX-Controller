import io
import re
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="VIX Futures Term Structure", page_icon="📈", layout="wide")

# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
    .stApp { background: #f6f3ea; }
    .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 2rem; }
    h1, h2, h3, p, div, span, label { color: #2b2b2b; }
    .small-subtitle { text-align:center; color:#6b6b6b; font-size:0.95rem; margin-top:-0.6rem; }
    .mini-note { text-align:center; color:#5674a6; font-size:0.9rem; margin-top:0.1rem; }
    .panel {
        background: #f8f5ee;
        border: 1px solid #b7ab91;
        border-radius: 2px;
        padding: 0.5rem 0.75rem;
    }
    .metric-box {
        border: 1px solid #b7ab91;
        background: #fbf8f1;
        padding: 0.5rem;
        text-align: center;
        border-radius: 2px;
    }
    .metric-label { font-size: 0.8rem; color:#5f5f5f; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color:#1f5da6; }
    .tiny { font-size: 0.8rem; color:#666; }
    .tbl-wrap table { width:100%; border-collapse: collapse; }
    .tbl-wrap td, .tbl-wrap th {
        border: 1px solid #6c685c;
        padding: 0.45rem 0.5rem;
        text-align:center;
        font-size:0.95rem;
        background:#f8f5ee;
    }
    .tbl-wrap th { background:#efe8d8; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

MONTH_MAP = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"
}
MONTH_ORDER = {k: i for i, k in enumerate(["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"], start=1)}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "symbol" == cl or cl.endswith("symbol"):
            mapping[c] = "Symbol"
        elif "expiration" in cl:
            mapping[c] = "Expiration"
        elif cl in {"last", "last price", "last_price"}:
            mapping[c] = "Last"
        elif "settle" in cl:
            mapping[c] = "Settle"
        elif "open" == cl:
            mapping[c] = "Open"
        elif "high" == cl:
            mapping[c] = "High"
        elif "low" == cl:
            mapping[c] = "Low"
        elif "bid" == cl:
            mapping[c] = "Bid"
        elif "ask" == cl:
            mapping[c] = "Ask"
        elif "change" == cl:
            mapping[c] = "Change"
    return df.rename(columns=mapping)


def monthly_contract_key(symbol: str):
    if not isinstance(symbol, str):
        return None
    s = symbol.strip().upper()
    # accepts VX/J6 and VX+VXT/J6; excludes VX12/H6 and VX+VXT01/F6
    m = re.fullmatch(r"(?:VX|VX\+VXT)/([FGHJKMNQUVXZ])(\d{1,2})", s)
    if not m:
        return None
    return m.group(1), m.group(2)


def month_label_from_symbol(symbol: str) -> str:
    key = monthly_contract_key(symbol)
    if not key:
        return symbol
    mon_code, yr = key
    yr4 = 2000 + int(yr[-2:])
    return f"{MONTH_MAP[mon_code]}"


def pretty_symbol(symbol: str) -> str:
    key = monthly_contract_key(symbol)
    if not key:
        return symbol
    mon_code, yr = key
    return f"VX/{mon_code}{yr[-1]}" if len(yr) == 1 else f"VX/{mon_code}{yr}"


def sort_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        key = monthly_contract_key(row.get("Symbol"))
        if key:
            mon_code, yr = key
            yr_num = 2000 + int(yr[-2:])
            rows.append((yr_num, MONTH_ORDER[mon_code]))
        else:
            rows.append((9999, 99))
    out = df.copy()
    out[["_year", "_month_ord"]] = rows
    out = out.sort_values(["_year", "_month_ord"]).drop(columns=["_year", "_month_ord"]).reset_index(drop=True)
    return out


@st.cache_data(ttl=900)
def fetch_settlement_for_date(target: date) -> pd.DataFrame:
    url = f"https://www.cboe.com/us/futures/market_statistics/settlement/csv?dt={target:%Y-%m-%d}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/octet-stream,text/plain,*/*",
    }
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    text = r.text.strip()
    if not text or "<!DOCTYPE" in text.upper() or "<HTML" in text.upper():
        raise ValueError("El endpoint de settlement no devolvió CSV válido.")
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        raise ValueError("CSV vacío.")
    return normalize_columns(df)


@st.cache_data(ttl=900)
def fetch_latest_available_curve(lookback_days: int = 7):
    errors = []
    for i in range(lookback_days + 1):
        d = date.today() - timedelta(days=i)
        try:
            df = fetch_settlement_for_date(d)
            monthly = df[df["Symbol"].astype(str).map(monthly_contract_key).notna()].copy()
            if monthly.empty:
                errors.append(f"{d}: sin contratos mensuales")
                continue
            monthly = sort_monthly_df(monthly)
            return d, monthly
        except Exception as e:
            errors.append(f"{d}: {e}")
    raise ValueError("No se encontró settlement reciente de CBOE. " + " | ".join(errors[:3]))


@st.cache_data(ttl=600)
def fetch_vix_spot():
    try:
        h = yf.Ticker("^VIX").history(period="7d")
        if h.empty:
            return None
        close = h["Close"].dropna()
        if close.empty:
            return None
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else None
        return {"price": round(price, 2), "prev": round(prev, 2) if prev is not None else None}
    except Exception:
        return None


def choose_price_col(df: pd.DataFrame) -> str:
    for c in ["Last", "Settle"]:
        if c in df.columns:
            return c
    raise ValueError("No encontré columna Last/Settle en settlement de CBOE.")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_curve(df_today: pd.DataFrame, df_prev: pd.DataFrame | None = None) -> pd.DataFrame:
    today = df_today.copy()
    price_col = choose_price_col(today)
    today[price_col] = clean_numeric(today[price_col])
    for col in ["Open", "Bid", "Ask", "High", "Low", "Change", "Settle"]:
        if col in today.columns:
            today[col] = clean_numeric(today[col])
    today = today[today[price_col].notna()].copy()
    today = sort_monthly_df(today)
    today["Month"] = today["Symbol"].map(month_label_from_symbol)
    today["DisplaySymbol"] = today["Symbol"].map(pretty_symbol)
    today["LastDisplay"] = today[price_col].round(3)
    today["Price"] = today[price_col].round(3)

    if df_prev is not None and not df_prev.empty:
        prev = df_prev.copy()
        prev_col = choose_price_col(prev)
        prev[prev_col] = clean_numeric(prev[prev_col])
        prev = prev[prev["Symbol"].astype(str).map(monthly_contract_key).notna()].copy()
        prev = prev[["Symbol", prev_col]].rename(columns={prev_col: "PreviousClose"})
        today = today.merge(prev, on="Symbol", how="left")
    else:
        today["PreviousClose"] = pd.NA

    return today.reset_index(drop=True)


def calc_term_metrics(curve: pd.DataFrame):
    px = curve["Price"].tolist()
    diffs = [None]
    contango = [None]
    for i in range(1, len(px)):
        diff = px[i] - px[i - 1] if pd.notna(px[i]) and pd.notna(px[i - 1]) else None
        diffs.append(diff)
        contango.append((diff / px[i - 1] * 100) if diff is not None and px[i - 1] else None)
    return diffs, contango




def fmt_num(x, digits=2, suffix=""):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def build_chart(curve: pd.DataFrame, spot: dict | None):
    x = curve["Month"].tolist()
    y = curve["Price"].tolist()
    prev = curve["PreviousClose"].tolist() if "PreviousClose" in curve.columns else [None] * len(curve)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers+text",
            name="Last",
            line=dict(color="#35a7ff", width=3, shape="spline"),
            marker=dict(size=8, color="#35a7ff"),
            text=[f"{v:.3f}" if pd.notna(v) else "" for v in y],
            textposition="top center",
            textfont=dict(color="#000000", size=13),
            hovertemplate="%{x}: %{y:.3f}<extra>Last</extra>",
        )
    )

    if any(pd.notna(v) for v in prev):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=prev,
                mode="lines+markers",
                name="Previous Close",
                line=dict(color="#707070", width=1.5, dash="dot"),
                marker=dict(size=5, color="#707070"),
                hovertemplate="%{x}: %{y:.3f}<extra>Previous Close</extra>",
            )
        )

    if spot and spot.get("price") is not None:
        fig.add_hline(
            y=spot["price"],
            line_dash="dash",
            line_color="#2f8f2f",
            line_width=3,
            annotation_text=f"{spot['price']:.2f}",
            annotation_position="top right",
            annotation_font_color="#000",
        )
        fig.add_trace(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[spot["price"], spot["price"]],
                mode="lines",
                name="VIX Index",
                line=dict(color="#2f8f2f", width=3, dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    y_all = [v for v in y if pd.notna(v)]
    if spot and spot.get("price") is not None:
        y_all.append(spot["price"])
    ymin = min(y_all) - 0.2
    ymax = max(y_all) + 0.25

    fig.update_layout(
        title=dict(text="VIX Futures Term Structure", x=0.5, xanchor="center", font=dict(size=22, color="#1f2f50")),
        paper_bgcolor="#f8f5ee",
        plot_bgcolor="#f8f5ee",
        margin=dict(l=70, r=40, t=70, b=60),
        height=560,
        legend=dict(x=1.01, y=1.0, bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=13)),
        xaxis=dict(title="Future Month", showgrid=False, tickfont=dict(size=14, color="#2b2b2b")),
        yaxis=dict(
            title="Volatility",
            range=[ymin, ymax],
            tick0=round(ymin * 4) / 4,
            dtick=0.05,
            gridcolor="#d4d1c9",
            zeroline=False,
            tickfont=dict(size=14, color="#2b2b2b"),
        ),
        hovermode="x unified",
    )
    return fig


# =========================
# UI / Data load
# =========================
header_cols = st.columns([8, 1])
with header_cols[0]:
    st.markdown("<h1 style='text-align:center; margin-bottom:0;'>VIX Futures Term Structure</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-subtitle'>Source: CBOE Delayed Quotes</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-note'>vixcentral.com</div>", unsafe_allow_html=True)
with header_cols[1]:
    with st.popover("☰"):
        months_to_show = st.slider("Meses a mostrar", 4, 9, 8)
        show_prev = st.checkbox("Mostrar previous close", True)
        hist_mode = st.checkbox("Usar fecha histórica")
        target_date = st.date_input("Fecha", value=date.today()) if hist_mode else None

try:
    asof, df_today_raw = fetch_latest_available_curve(lookback_days=10) if not hist_mode else (target_date, fetch_settlement_for_date(target_date))
    df_today_raw = df_today_raw[df_today_raw["Symbol"].astype(str).map(monthly_contract_key).notna()].copy()

    prev_df = None
    if show_prev:
        for i in range(1, 6):
            try:
                prev_df = fetch_settlement_for_date(asof - timedelta(days=i))
                break
            except Exception:
                continue

    curve = prepare_curve(df_today_raw, prev_df).head(months_to_show)
    if curve.empty:
        raise ValueError("CBOE no devolvió contratos mensuales utilizables.")

    spot = fetch_vix_spot()
    diffs, contango = calc_term_metrics(curve)

    mcols = st.columns(5)
    m1 = curve.iloc[0]["Price"] if len(curve) >= 1 else None
    m2 = curve.iloc[1]["Price"] if len(curve) >= 2 else None
    with mcols[0]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>VIX Spot</div><div class='metric-value'>{fmt_num(spot['price'] if spot else None, 2)}</div></div>", unsafe_allow_html=True)
    with mcols[1]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>M1</div><div class='metric-value'>{fmt_num(m1, 2)}</div></div>", unsafe_allow_html=True)
    with mcols[2]:
        st.markdown(f"<div class='metric-box'><div class='metric-label'>M2</div><div class='metric-value'>{fmt_num(m2, 2)}</div></div>", unsafe_allow_html=True)
    with mcols[3]:
        spread = (m2 - m1) if (m1 is not None and m2 is not None) else None
        st.markdown(f"<div class='metric-box'><div class='metric-label'>Difference</div><div class='metric-value'>{fmt_num(spread, 2)}</div></div>", unsafe_allow_html=True)
    with mcols[4]:
        c12 = contango[1] if len(contango) > 1 else None
        st.markdown(f"<div class='metric-box'><div class='metric-label'>% Contango</div><div class='metric-value'>{fmt_num(c12, 2, '%')}</div></div>", unsafe_allow_html=True)

    fig = build_chart(curve, spot)
    st.plotly_chart(fig, use_container_width=True)

    info = []
    info.append(f"Curva con settlement CBOE de {asof:%Y-%m-%d}")
    if prev_df is not None:
        info.append("previous close cargado")
    if spot:
        info.append("VIX spot vía Yahoo")
    st.caption(" · ".join(info))

    # main table style matching screenshot
    idx_headers = list(range(1, len(curve) + 1))
    cont_cells = ["—"] + [f"{v:.2f}%" if v is not None else "—" for v in contango[1:]]
    diff_cells = ["—"] + [f"{v:.2f}" if v is not None else "—" for v in diffs[1:]]

    contango_html = "<div class='tbl-wrap'><table><tr><th>% Contango</th>"
    for i in idx_headers:
        contango_html += f"<th>{i}</th>"
    contango_html += "</tr><tr><td></td>"
    for val in cont_cells:
        contango_html += f"<td>{val}</td>"
    contango_html += "</tr><tr><th>Difference</th>"
    for i in idx_headers:
        contango_html += f"<th>{i}</th>"
    contango_html += "</tr><tr><td></td>"
    for val in diff_cells:
        contango_html += f"<td>{val}</td>"
    contango_html += "</tr></table></div>"
    st.markdown(contango_html, unsafe_allow_html=True)

    # month 7 to 4 block, like screenshot
    if len(curve) >= 7:
        c74 = ((curve.iloc[6]["Price"] - curve.iloc[3]["Price"]) / curve.iloc[3]["Price"] * 100) if curve.iloc[3]["Price"] else None
        d74 = curve.iloc[6]["Price"] - curve.iloc[3]["Price"]
        mini_html = "<div class='tbl-wrap' style='width: 300px; margin-top: 10px;'><table>"
        mini_html += "<tr><th>Month 7 to 4 contango</th>"
        mini_html += f"<td>{c74:.2f}%</td><td>{d74:.2f}</td></tr></table></div>"
        st.markdown(mini_html, unsafe_allow_html=True)

    with st.expander("Ver tabla de contratos"):
        show_cols = [c for c in ["DisplaySymbol", "Month", "Price", "PreviousClose", "Open", "Bid", "Ask", "High", "Low", "Change", "Expiration"] if c in curve.columns]
        st.dataframe(curve[show_cols], use_container_width=True, hide_index=True)

except Exception as e:
    st.error("No se pudo cargar la curva mensual de VIX desde CBOE.")
    st.caption(f"Detalle técnico: {e}")
