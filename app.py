import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

st.set_page_config(page_title="VIX Term Structure", page_icon="📈", layout="wide")

CBOE_URL = "https://www.cboe.com/tradable-products/vix/vix-futures/"
MONTH_ORDER = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5e7eb; }
    .block-container { max-width: 1280px; padding-top: 1rem; }
    .hero {
        padding: 1rem 1.25rem; border: 1px solid rgba(148,163,184,.18); border-radius: 14px;
        background: linear-gradient(180deg, rgba(15,23,42,.9), rgba(2,6,23,.96));
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.8rem; color: #f8fafc; }
    .hero p { margin: .35rem 0 0 0; color: #94a3b8; }
    .smallnote { color: #94a3b8; font-size: .88rem; }
    .metricbox {
        border: 1px solid rgba(148,163,184,.14); border-radius: 12px; padding: .85rem 1rem;
        background: rgba(15,23,42,.78);
    }
    .metricbox .label { color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
    .metricbox .value { color: #f8fafc; font-size: 1.45rem; font-weight: 700; }
    .section-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: .75rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_monthly_vx(symbol: str) -> bool:
    return bool(re.fullmatch(r"VX/[FGHJKMNQUVXZ]\d", str(symbol).strip().upper()))



def infer_sort_key(symbol: str):
    s = str(symbol).strip().upper()
    m = re.fullmatch(r"VX/([FGHJKMNQUVXZ])(\d)", s)
    if not m:
        return (9999, 99)
    month_code, year_digit = m.groups()
    # assume current decade for single-digit CBOE symbol style, adequate for near-term VX curve
    current_decade = (datetime.utcnow().year // 10) * 10
    year = current_decade + int(year_digit)
    if year < datetime.utcnow().year - 1:
        year += 10
    return (year, MONTH_ORDER.get(month_code, 99))



def build_driver() -> webdriver.Chrome:
    options = Options()
    options.binary_location = "/usr/bin/chromium"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")

    service = Service("/usr/bin/chromedriver")
    return webdriver.Chrome(service=service, options=options)


@st.cache_data(ttl=120)
def fetch_cboe_vix_table() -> pd.DataFrame:
    driver = None
    try:
        driver = build_driver()
        driver.get(CBOE_URL)

        WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        WebDriverWait(driver, 45).until(
            lambda d: d.execute_script(
                """
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const rows = [...table.querySelectorAll('tbody tr')];
                    for (const r of rows) {
                        const txt = (r.innerText || '').trim();
                        if (txt.includes('VX/')) return true;
                    }
                }
                return false;
                """
            )
        )

        payload = driver.execute_script(
            """
            const allTables = document.querySelectorAll('table');
            let found = null;
            allTables.forEach((table, idx) => {
              const headers = [...table.querySelectorAll('th')].map(h => h.innerText.trim());
              const rows = [...table.querySelectorAll('tbody tr')].map(r =>
                [...r.querySelectorAll('td')].map(c => c.innerText.trim())
              ).filter(r => r.some(c => c.includes('VX/')));
              if (rows.length > 0 && !found) {
                found = { index: idx, headers, rows };
              }
            });
            return found;
            """
        )

        if not payload or not payload.get("rows"):
            raise RuntimeError("No encontré una tabla renderizada con filas VX/ en CBOE.")

        headers = [str(h).strip().title() for h in payload["headers"]]
        df = pd.DataFrame(payload["rows"], columns=headers)

        rename_map = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in {"symbol", "future", "future symbol"}:
                rename_map[c] = "Symbol"
            elif lc in {"expiration", "expiration date"}:
                rename_map[c] = "Expiration"
            elif lc in {"last", "last price", "price"} or "last" in lc:
                rename_map[c] = "Last"
            elif lc == "change" or "change" in lc:
                rename_map[c] = "Change"
            elif lc == "high" or "high" in lc:
                rename_map[c] = "High"
            elif lc == "low" or "low" in lc:
                rename_map[c] = "Low"
            elif lc in {"settlement", "settlement price", "daily settlement price"} or "settlement" in lc:
                rename_map[c] = "Settlement"
            elif lc == "volume" or "volume" in lc:
                rename_map[c] = "Volume"
        df = df.rename(columns=rename_map)

        required = ["Symbol", "Expiration", "Last", "Change", "High", "Low", "Settlement", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"La tabla renderizada no trajo las columnas esperadas. Faltan: {', '.join(missing)}. Detectadas: {list(df.columns)}")

        df = df[df["Symbol"].apply(is_monthly_vx)].copy()
        if df.empty:
            raise RuntimeError("La tabla renderizada no devolvió contratos mensuales VX válidos.")

        for col in ["Last", "Change", "High", "Low", "Settlement"]:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False).replace({"": None, "-": None, "--": None, "----": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Volume"] = (
            df["Volume"].astype(str).str.replace(",", "", regex=False).replace({"": None, "-": None, "--": None})
        )
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Expiration"] = pd.to_datetime(df["Expiration"], errors="coerce")

        df = df.sort_values(["Expiration", "Symbol"]).reset_index(drop=True)
        df["M"] = [f"M{i}" for i in range(1, len(df) + 1)]
        df["term_price"] = df["Last"]
        return df
    except TimeoutException as e:
        raise RuntimeError("Timeout esperando a que CBOE renderizara la tabla de VIX futures.") from e
    except WebDriverException as e:
        raise RuntimeError(f"No pude iniciar Chromium/ChromeDriver en el entorno de deploy: {e}") from e
    finally:
        if driver is not None:
            driver.quit()



def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(df) - 1):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]
        if pd.notna(cur["Last"]) and pd.notna(nxt["Last"]) and cur["Last"] != 0:
            pct = (nxt["Last"] / cur["Last"] - 1) * 100
            pts = nxt["Last"] - cur["Last"]
        else:
            pct = None
            pts = None
        rows.append({
            "Spread": f"{cur['M']}→{nxt['M']}",
            "% Contango": pct,
            "Difference": pts,
        })
    return pd.DataFrame(rows)



def month_7_to_4(df: pd.DataFrame):
    if len(df) < 7:
        return None, None
    m4 = df.iloc[3]["Last"]
    m7 = df.iloc[6]["Last"]
    if pd.isna(m4) or pd.isna(m7) or m4 == 0:
        return None, None
    return (m7 / m4 - 1) * 100, m7 - m4



def chart_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["M"],
            y=df["Last"],
            mode="lines+markers+text",
            text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in df["Last"]],
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=10),
            hovertemplate="%{x}<br>Last: %{y:.2f}<extra></extra>",
            name="VIX Futures Last",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title="VIX Monthly Term Structure (CBOE Last)",
        xaxis_title="Contract Month",
        yaxis_title="Price",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


st.markdown(
    f"""
    <div class='hero'>
      <h1>VIX Term Structure Dashboard</h1>
      <p>Curva mensual de futuros VX extraída de la tabla renderizada de CBOE. La curva usa <b>Last</b>, no Settlement.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1, 4])
with col_a:
    refresh = st.button("Refresh data", use_container_width=True)
with col_b:
    st.markdown("<div class='smallnote'>Si actualizaste archivos en Streamlit Cloud, haz un reboot de la app además del refresh.</div>", unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()

try:
    curve = fetch_cboe_vix_table()
except Exception as e:
    st.error(f"No se pudo construir la curva mensual de VIX desde CBOE.\n\nDetalle técnico: {e}")
    st.stop()

spreads = compute_spreads(curve)
m7_pct, m7_pts = month_7_to_4(curve)

m1 = curve.iloc[0]["Last"] if len(curve) > 0 else None
m2 = curve.iloc[1]["Last"] if len(curve) > 1 else None
m1m2_pct = ((m2 / m1 - 1) * 100) if pd.notna(m1) and pd.notna(m2) and m1 else None
m1m2_pts = (m2 - m1) if pd.notna(m1) and pd.notna(m2) else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    val = f"{m1:.2f}" if pd.notna(m1) else "N/A"
    st.markdown(f"<div class='metricbox'><div class='label'>M1 Last</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)
with c2:
    val = f"{m2:.2f}" if pd.notna(m2) else "N/A"
    st.markdown(f"<div class='metricbox'><div class='label'>M2 Last</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)
with c3:
    val = f"{m1m2_pct:.2f}%" if m1m2_pct is not None else "N/A"
    st.markdown(f"<div class='metricbox'><div class='label'>M1→M2 % Contango</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)
with c4:
    val = f"{m7_pct:.2f}%" if m7_pct is not None else "N/A"
    st.markdown(f"<div class='metricbox'><div class='label'>Month 7 to 4 %</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)

left, right = st.columns([2.1, 1])
with left:
    st.plotly_chart(chart_curve(curve), use_container_width=True)
with right:
    st.markdown("<div class='section-title'>Spread Metrics</div>", unsafe_allow_html=True)
    spreads_display = spreads.copy()
    spreads_display["% Contango"] = spreads_display["% Contango"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    spreads_display["Difference"] = spreads_display["Difference"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    st.dataframe(
        spreads_display,
        use_container_width=True,
        hide_index=True,
    )
    extra = pd.DataFrame([
        {"Metric": "M1→M2 Difference", "Value": None if m1m2_pts is None else round(m1m2_pts, 2)},
        {"Metric": "Month 7 to 4 % Contango", "Value": None if m7_pct is None else round(m7_pct, 2)},
        {"Metric": "Month 7 to 4 Difference", "Value": None if m7_pts is None else round(m7_pts, 2)},
    ])
    st.dataframe(extra, use_container_width=True, hide_index=True)

st.markdown("<div class='section-title'>Monthly VX Table</div>", unsafe_allow_html=True)
show_df = curve[["M", "Symbol", "Expiration", "Last", "Change", "High", "Low", "Settlement", "Volume"]].copy()
show_df["Expiration"] = show_df["Expiration"].dt.strftime("%Y-%m-%d")
show_display = show_df.copy()
for col, fmt in {
    "Last": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    "Change": lambda x: f"{x:.3f}" if pd.notna(x) else "N/A",
    "High": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    "Low": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    "Settlement": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
    "Volume": lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A",
}.items():
    show_display[col] = show_display[col].apply(fmt)

st.dataframe(
    show_display,
    use_container_width=True,
    hide_index=True,
)

with st.expander("Ideas para agregar a la app"):
    st.markdown(
        """
- Overlay de curvas históricas: hoy vs ayer vs 1 semana vs 1 mes.
- Basis del spot VIX vs M1 y M2.
- Curvature: `M3 - 2*M2 + M1`.
- Semáforo de régimen: contango, flat, backwardation.
- Alertas cuando `M2 < M1` o cuando el `Month 7 to 4` cambie de signo.
- Dashboard de instrumentos ligados a vol: SVXY, SVIX, VXX, UVXY.
- Exportación a Excel/CSV y snapshot PNG para distribución institucional.
- Histórico diario guardado en un archivo local o bucket para backtesting.
        """
    )

with st.expander("Diagnóstico"):
    st.write("Fuente:", CBOE_URL)
    st.write("Hora UTC:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    st.dataframe(curve, use_container_width=True, hide_index=True)
