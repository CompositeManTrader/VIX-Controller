"""
options_page.py — Página "Opciones": lo que descuenta el mercado de opciones.

Para un corto en VXX importan tres cosas, y la página va en ese orden:
  1. Cuánto se paga hoy por la volatilidad y por la protección (SPY).
  2. Si los dealers amortiguan o amplifican los movimientos (gamma).
  3. Cuánto cuesta una call del VIX: es exactamente el riesgo de la posición.
Más el movimiento que descuentan los próximos datos macro y la historia de
la foto diaria (data/options_history.parquet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vix_controller import calendario as cal
from vix_controller.data import yahoo_options as yo
from vix_controller.quant import options_desk as od
from vix_controller.quant import vix_inverse as vi
from vix_controller.rates import get_dividend_yield, get_risk_free_rate
from vix_controller.ui import theme as T

HIST_PATH = Path("data/options_history.parquet")
ne, pe = vi.num_es, vi.pct_es
DIAS_EVENTOS = 30


# ──────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def _cargar(ticker: str, eventos: tuple, cercanos: int, objetivos: tuple, rango: tuple) -> dict:
    """Cacheado 15 min, también si falla: no se vuelve a pedir a Yahoo en cada clic."""
    try:
        cad, spot, estado = yo.descargar(ticker, cercanos=cercanos, objetivos=objetivos,
                                         eventos=eventos, rango=rango, pausa=0.3)
        return {"cadenas": cad, "spot": spot, "estado": estado,
                "hora": pd.Timestamp.now(tz="America/Mexico_City").strftime("%d/%m %H:%M")}
    except Exception as e:                      # noqa: BLE001 — se muestra, no se oculta
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def _tipos() -> tuple[float, float]:
    return float(get_risk_free_rate()), float(get_dividend_yield("SPY"))


@st.cache_data(ttl=600, show_spinner=False)
def _historia(mtime: float) -> pd.DataFrame:
    h = pd.read_parquet(HIST_PATH)
    h.index = pd.DatetimeIndex(h.index).normalize()
    return h.sort_index()


def _pct(v: float, dec: int = 1, signo: bool = False) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{ne(v * 100, dec, signo)} %"


def _pts(v: float, dec: int = 1) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{ne(v * 100, dec, signo=True)} pts"


# ──────────────────────────────────────────────────────────────────────
# Gráficas
# ──────────────────────────────────────────────────────────────────────
def _chart_estructura(est: pd.DataFrame, eventos: list[tuple], hoy: pd.Timestamp) -> go.Figure:
    e = est.dropna(subset=["atm"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["dte"], y=(e["atm"] * 100).tolist(), mode="lines+markers",
                             name="IV ATM", line=dict(color=T.AMBER, width=2), marker=dict(size=7),
                             customdata=e["vencimiento"],
                             hovertemplate="%{customdata} · %{x} días<br>IV ATM %{y:.1f} %<extra></extra>"))
    for _, nombre, f in eventos:
        d = (f - hoy).days
        if e.empty or d > e["dte"].max():
            continue
        fig.add_vline(x=d, line_color=T.GRAY, line_width=1, line_dash="dot",
                      annotation_text=f"{nombre} {f.strftime('%d/%m')}", annotation_position="top",
                      annotation_font=dict(family=T.FONT_MONO, size=9, color=T.GRAY))
    fig.update_xaxes(title=dict(text="días al vencimiento", font=dict(size=10)))
    fig.update_yaxes(ticksuffix=" %")
    fig.update_layout(height=300, showlegend=False, hovermode="closest",
                      margin=dict(l=48, r=16, t=24, b=36))
    return fig


def _chart_sonrisas(cad: dict, spot: float, r: float, q: float) -> go.Figure:
    venc = sorted(cad.items(), key=lambda x: x[1]["dte"])
    elegidos = []
    for objetivo in (7, 30, 90):
        cand = [v for v in venc if v[1]["dte"] >= min(objetivo, 7)]
        if cand:
            elegido = min(cand, key=lambda v: abs(v[1]["dte"] - objetivo))
            if elegido[0] not in [x[0] for x in elegidos]:
                elegidos.append(elegido)
    estilos = [(T.WHITE, 1.2, "dot"), (T.AMBER, 2.2, None), (T.GRAY, 1.4, "dash")]
    fig = go.Figure()
    for (venc_s, dato), (color, w, dash) in zip(elegidos, estilos):
        s = od.sonrisa(dato, spot, r, q)
        s = s[s["m"].between(-0.15, 0.08)]
        fig.add_trace(go.Scatter(x=(s["m"] * 100).tolist(), y=(s["iv"] * 100).tolist(), mode="lines",
                                 name=f"{pd.Timestamp(venc_s).strftime('%d/%m')} · {dato['dte']} días",
                                 line=dict(color=color, width=w, dash=dash),
                                 hovertemplate="%{x:+.1f} % · IV %{y:.1f} %<extra></extra>"))
    fig.add_vline(x=0, line_color=T.ZERO, line_width=1)
    fig.update_xaxes(title=dict(text="strike frente al spot (%) · izquierda = puts de protección",
                                font=dict(size=10)), ticksuffix=" %")
    fig.update_yaxes(ticksuffix=" %")
    fig.update_layout(height=320, hovermode="x unified", margin=dict(l=48, r=16, t=10, b=40),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _chart_gex_strikes(gex: pd.DataFrame, spot: float, giro: float | None, mu: dict) -> go.Figure:
    g = gex[gex["strike"].between(spot * 0.94, spot * 1.06)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["strike"], y=g["calls"].tolist(), name="Calls", marker_color=T.PROFIT,
                         opacity=0.75, hovertemplate="K %{x} · calls %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=g["strike"], y=g["puts"].tolist(), name="Puts", marker_color=T.LOSS,
                         opacity=0.75, hovertemplate="K %{x} · puts %{y:.2f}<extra></extra>"))
    fig.add_vline(x=spot, line_color=T.WHITE, line_width=1.5,
                  annotation_text=f"SPY {ne(spot, 2)}", annotation_position="top",
                  annotation_font=dict(family=T.FONT_MONO, size=10, color=T.WHITE))
    if giro:
        fig.add_vline(x=giro, line_color=T.AMBER, line_width=1.5, line_dash="dash")
    fig.update_xaxes(title=dict(text="strike", font=dict(size=10)))
    fig.update_yaxes(title=dict(text="miles de millones $ por 1 %", font=dict(size=10)))
    fig.update_layout(height=320, barmode="relative", bargap=0.1, hovermode="x unified",
                      margin=dict(l=56, r=16, t=24, b=36), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def _chart_gex_precio(curva: pd.Series, spot: float, giro: float | None) -> go.Figure:
    fig = go.Figure()
    y = curva.to_numpy()
    fig.add_trace(go.Scatter(x=curva.index, y=np.where(y < 0, y, 0).tolist(), mode="lines",
                             fill="tozeroy", fillcolor="rgba(234,57,67,0.18)", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=curva.index, y=np.where(y > 0, y, 0).tolist(), mode="lines",
                             fill="tozeroy", fillcolor="rgba(22,199,132,0.15)", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=curva.index, y=y.tolist(), mode="lines", name="GEX total",
                             line=dict(color=T.WHITE, width=1.8),
                             hovertemplate="SPY %{x:.0f} · GEX %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=T.ZERO, line_width=1)
    fig.add_vline(x=spot, line_color=T.WHITE, line_width=1, line_dash="dot",
                  annotation_text="hoy", annotation_position="top",
                  annotation_font=dict(family=T.FONT_MONO, size=10, color=T.WHITE))
    if giro:
        fig.add_vline(x=giro, line_color=T.AMBER, line_width=1.5, line_dash="dash",
                      annotation_text=f"giro {ne(giro, 0)}", annotation_position="bottom",
                      annotation_font=dict(family=T.FONT_MONO, size=10, color=T.AMBER))
    fig.update_xaxes(title=dict(text="precio hipotético del SPY", font=dict(size=10)))
    fig.update_yaxes(title=dict(text="miles de millones $ por 1 %", font=dict(size=10)))
    fig.update_layout(height=320, showlegend=False, hovermode="x unified",
                      margin=dict(l=56, r=16, t=24, b=36))
    return fig


def _chart_vix(s: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure()
    F = s.attrs.get("F", np.nan)
    for tipo, color, nombre in (("P", T.GRAY, "Puts"), ("C", T.AMBER, "Calls")):
        x = s[s["tipo"] == tipo]
        fig.add_trace(go.Scatter(x=x["strike"], y=(x["iv"] * 100).tolist(), mode="lines+markers",
                                 name=nombre, line=dict(color=color, width=1.8), marker=dict(size=5),
                                 customdata=x["mid"],
                                 hovertemplate="K %{x} · IV %{y:.0f} % · $%{customdata:.2f}<extra></extra>"))
    if np.isfinite(F):
        fig.add_vline(x=F, line_color=T.WHITE, line_width=1, line_dash="dot",
                      annotation_text=f"futuro {ne(F, 2)}", annotation_position="bottom right",
                      annotation_font=dict(family=T.FONT_MONO, size=10, color=T.WHITE))
    fig.update_xaxes(title=dict(text=f"strike del VIX · {titulo}", font=dict(size=10)))
    fig.update_yaxes(ticksuffix=" %")
    fig.update_layout(height=320, hovermode="x unified", margin=dict(l=48, r=16, t=24, b=36),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def _chart_hist(h: pd.Series, nombre: str, color: str, suf: str = "", dec: int = 1,
                cero: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.tolist(), mode="lines+markers", name=nombre,
                             line=dict(color=color, width=1.6), marker=dict(size=4),
                             hovertemplate=f"%{{x|%d/%m/%Y}} · %{{y:.{dec}f}}{suf}<extra></extra>"))
    if cero:
        fig.add_hline(y=0, line_color=T.ZERO, line_width=1)
    fig.update_layout(height=220, showlegend=False, hovermode="x unified",
                      margin=dict(l=48, r=16, t=8, b=28))
    return fig


# ──────────────────────────────────────────────────────────────────────
# Piezas
# ──────────────────────────────────────────────────────────────────────
def _card(label: str, big: str, cls: str, filas: list[tuple[str, str]], nota: str = "") -> str:
    rows = "".join(f'<div class="stc-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in filas)
    nota = f'<div class="stc-sub" style="margin-top:0.55rem;font-size:0.78rem">{nota}</div>' if nota else ""
    return (f'<div class="stc-card"><div class="stc-label">{label}</div>'
            f'<div class="stc-big {cls}">{big}</div>{rows}{nota}</div>')


def _tabla_eventos(cad: dict, spot: float, iv30: float, eventos: list[tuple],
                   hoy: pd.Timestamp) -> str:
    filas = []
    venc = sorted(cad.items(), key=lambda x: x[1]["dte"])
    for _, nombre, f in eventos:
        tras = [(v, d) for v, d in venc if pd.Timestamp(v) >= f]
        if not tras:
            continue
        v, dato = tras[0]
        mov = od.movimiento_descontado(dato, spot)
        base = 0.8 * iv30 * np.sqrt(max(dato["dte"], 0.5) / 365) * 100 if np.isfinite(iv30) else np.nan
        exceso = mov - base
        filas.append(f'<tr><td>{nombre}</td><td>{f.strftime("%d/%m")}</td>'
                     f'<td>{pd.Timestamp(v).strftime("%d/%m")} · {dato["dte"]} d</td>'
                     f'<td>± {ne(mov, 2)} %</td><td>± {ne(base, 2)} %</td>'
                     f'<td class="{"am" if exceso > 0.3 else ""}">{ne(exceso, 2, signo=True)} pts</td></tr>')
    if not filas:
        return ('<p class="stc-note">Sin datos macro en los próximos 30 días con vencimiento '
                'cargado.</p>')
    return ('<div style="overflow-x:auto"><table class="stc-table"><tr><th>Dato</th><th>Fecha</th>'
            '<th>Vencimiento</th><th>Descontado</th><th>Sin evento</th>'
            '<th>Prima</th></tr>' + "".join(filas) + "</table></div>")


def _tabla_vix(vix: dict, r: float, principal: str | None) -> str:
    filas = []
    for v, dato in sorted(vix.items(), key=lambda x: x[1]["dte"]):
        m = od.metricas_vix(dato, r)
        if not m:
            continue
        oi = m["oi_calls"] + m["oi_puts"]
        cls = ' class="open"' if v == principal else ""
        filas.append(f'<tr{cls}><td>{pd.Timestamp(v).strftime("%d/%m/%Y")}</td><td>{m["dte"]}</td>'
                     f'<td>{ne(m["forward"], 2)}</td><td>{_pct(m["atm"], 0)}</td>'
                     f'<td>{_pct(m["call25"], 0)}</td><td>{_pts(m["sesgo"], 0)}</td>'
                     f'<td>{ne(oi / 1000, 0)} mil</td></tr>')
    return ('<div style="overflow-x:auto"><table class="stc-table"><tr><th>Vencimiento</th>'
            '<th>Días</th><th>Futuro implícito</th><th>IV ATM</th><th>IV call 25Δ</th>'
            '<th>Sesgo de calls</th><th>Interés abierto</th></tr>' + "".join(filas) + "</table></div>")


def _error_yahoo(ticker: str, msg: str) -> None:
    st.markdown(f'<div class="stc-alert loss"><b>Sin opciones de {ticker}</b>Yahoo Finance no '
                f'devolvió cotizaciones válidas ({T.esc(msg[:160])}). En Streamlit Cloud suele ser el '
                f'límite de peticiones: vuelve a intentarlo en 15 minutos o con el mercado abierto.'
                f'</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Página
# ──────────────────────────────────────────────────────────────────────
def render(vix_spot: float | None = None) -> None:
    hoy = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    eventos = cal.proximos(hoy, DIAS_EVENTOS)
    r, q = _tipos()

    st.markdown(f"""{T.eyebrow("Opciones")}
        <div class="stc-h1">Lo que descuenta el mercado de opciones</div>
        <p class="stc-lead">Para un corto en VXX importan tres cosas: cuánto se paga hoy por la
        volatilidad y la protección en el SPY, si los dealers amortiguan o amplifican los movimientos,
        y cuánto cuesta una call del VIX, que es exactamente el riesgo de la posición.</p>""",
                unsafe_allow_html=True)

    c_bt, _ = st.columns([1, 4])
    with c_bt:
        if st.button("Actualizar opciones", key="btn_opts", width="stretch"):
            _cargar.clear()
            st.rerun()

    with st.spinner("Descargando cadenas de SPY y del VIX de Yahoo…"):
        spy = _cargar("SPY", tuple(f.strftime("%Y-%m-%d") for _, _, f in eventos), 3, (30, 60, 90),
                      (0.70, 1.30))
        vix = _cargar("^VIX", (), 2, (30, 60), (0.50, 4.00))

    if "error" in spy:
        _error_yahoo("SPY", spy["error"])
        return
    spot = spy["spot"]
    cad = od.preparar(spy["cadenas"], spot, r, q)           # IV de cada opción, una sola vez
    abierto = spy["estado"] == "REGULAR"
    b_mercado = T.badge("Mercado abierto" if abierto else "Mercado cerrado",
                        "profit" if abierto else "amber")
    b_hora = T.badge("Descargado " + spy["hora"] + " CDMX")
    st.markdown(f'<div class="stc-badges">{T.badge("Yahoo Finance · ~15 min de retraso")}'
                f'{b_mercado}{b_hora}</div>', unsafe_allow_html=True)
    if not abierto:
        st.markdown('<div class="stc-alert"><b>Mercado cerrado</b>Fuera de horario Yahoo deja muchas '
                    'cotizaciones a cero: faltan strikes y las medidas son de la última sesión. Para '
                    'decidir, mírala con el mercado abierto (9:30 a 16:00 de Nueva York).</div>', unsafe_allow_html=True)

    est = od.estructura(cad, spot, r, q)
    iv30 = od.a_30_dias(est, "atm")
    rr30, bf30 = od.a_30_dias(est, "rr25"), od.a_30_dias(est, "bf25")
    p25 = od.a_30_dias(est, "put25")
    gex = od.gex_por_strike(cad, spot, r, q)
    curva = od.gex_en_precios(cad, spot, r, q, np.linspace(spot * 0.92, spot * 1.08, 81))
    giro = od.nivel_giro(curva, spot)
    mu = od.muros(gex, spot)
    gex_tot = float(gex["neto"].sum()) if not gex.empty else np.nan
    vcad = vix.get("cadenas", {}) if "error" not in vix else {}
    principal = od.vix_principal(vcad) if vcad else None
    mv = od.metricas_vix(vcad[principal], r) if principal else {}

    # ── 1. Lectura de hoy ───────────────────────────────────────────
    st.markdown(T.section("Hoy", "Lectura de hoy"), unsafe_allow_html=True)
    filas_iv = [("Movimiento diario implícito",
                 f"± {ne(iv30 * 100 / np.sqrt(252), 2)} %" if np.isfinite(iv30) else "—")]
    if vix_spot:
        filas_iv = [("VIX (toda la sonrisa)", ne(vix_spot, 2)),
                    ("VIX − IV ATM", f"{ne(vix_spot - iv30 * 100, 1, signo=True)} pts")] + filas_iv
    amort = np.isfinite(gex_tot) and gex_tot > 0
    giro_txt = (f"{ne(giro, 0)} ({ne((giro / spot - 1) * 100, 1, signo=True)} %)" if giro
                else "fuera de ±8 %")
    muros_txt = (f"{ne(mu['put'], 0) if mu['put'] else '—'} / "
                 f"{ne(mu['call'], 0) if mu['call'] else '—'}")
    tarjetas = [
        _card("IV a 30 días · SPY en el dinero", _pct(iv30), "amber", filas_iv,
              "El VIX va por encima porque incluye las alas: la diferencia es lo que se paga por "
              "las colas."),
        _card("Sesgo 25Δ a 30 días", _pts(rr30),
              "loss" if np.isfinite(rr30) and rr30 < -0.05 else "white",
              [("IV put 25Δ", _pct(p25)), ("IV call 25Δ", _pct(od.a_30_dias(est, "call25"))),
               ("Butterfly 25Δ", _pts(bf30))],
              "Call menos put a la misma distancia en delta. Cuanto más negativo, más se paga por "
              "protegerse de caídas."),
        _card("Gamma de los dealers", "AMORTIGUAN" if amort else "AMPLIFICAN",
              "profit" if amort else "loss",
              [("GEX total", f"{ne(gex_tot, 2, signo=True)} mil M$ / 1 %"),
               ("Nivel de giro", giro_txt), ("Muros put / call", muros_txt)],
              "Por debajo del giro los dealers venden en las caídas: es donde nacen los saltos del "
              "VIX."),
    ]
    if mv:
        tarjetas.append(_card(
            f"Calls del VIX · {pd.Timestamp(principal).strftime('%d/%m')}", _pts(mv["sesgo"], 0),
            "amber",
            [("Futuro implícito", ne(mv["forward"], 2)),
             ("IV ATM / call 25Δ", f"{_pct(mv['atm'], 0)} / {_pct(mv['call25'], 0)}"),
             (f"Call {ne(mv['k50'], 0)} (≈1,5× futuro)", f"$ {ne(mv['precio50'], 2)}")],
            "Sesgo = IV de la call 25Δ menos la ATM: lo que se paga por un salto del VIX, el "
            "riesgo de tu corto."))
    else:
        tarjetas.append(_card("Calls del VIX", "—", "white", [],
                              "Yahoo no devolvió la cadena del VIX: "
                              + T.esc(vix.get("error", ""))[:120]))
    st.markdown(f'<div class="stc-grid">{"".join(tarjetas)}</div>', unsafe_allow_html=True)

    # ── 2. Estructura y datos macro ─────────────────────────────────
    st.markdown(T.section("Plazos", "Estructura de la volatilidad y datos macro",
                          "IV en el dinero por vencimiento. Un vencimiento cercano por encima de los "
                          "siguientes significa que el mercado paga por un evento inminente. La tabla "
                          "aísla lo que se paga por cada dato: el movimiento que descuenta el straddle "
                          "frente al que daría la IV a 30 días en el mismo plazo."),
                unsafe_allow_html=True)
    st.plotly_chart(_chart_estructura(est, eventos, hoy), width="stretch",
                    config={"displayModeBar": False})
    st.markdown(_tabla_eventos(cad, spot, iv30, eventos, hoy), unsafe_allow_html=True)
    st.markdown('<p class="stc-note" style="margin-top:0.5rem">Movimiento medio esperado hasta el '
                'vencimiento (straddle en el dinero / spot) frente al que daría la IV a 30 días en el '
                'mismo plazo. En ámbar, datos por los que se paga una prima clara.</p>',
                unsafe_allow_html=True)

    # ── 3. Sesgo ─────────────────────────────────────────────────────
    st.markdown(T.section("Sesgo", "La sonrisa del SPY",
                          "IV por strike, del lado fuera del dinero (puts a la izquierda, calls a la "
                          "derecha). Una pendiente izquierda que se empina es demanda de protección."),
                unsafe_allow_html=True)
    st.plotly_chart(_chart_sonrisas(cad, spot, r, q), width="stretch", config={"displayModeBar": False})

    # ── 4. Gamma ─────────────────────────────────────────────────────
    st.markdown(T.section("Gamma", "Dónde amortiguan y dónde amplifican los dealers",
                          "Convención estándar: los dealers tienen las calls que venden los clientes "
                          "(gamma positiva) y han vendido las puts que estos compran (gamma negativa). "
                          "A la derecha, la GEX total si el SPY estuviera en cada precio: el cruce con "
                          "cero es el nivel de giro."), unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="medium")
    with h1:
        st.plotly_chart(_chart_gex_strikes(gex, spot, giro, mu), width="stretch",
                        config={"displayModeBar": False})
    with h2:
        st.plotly_chart(_chart_gex_precio(curva, spot, giro), width="stretch",
                        config={"displayModeBar": False})
    st.markdown(f'<p class="stc-note">Con {len(cad)} vencimientos (los más cercanos, donde está casi '
                f'toda la gamma, y los de 30, 60 y 90 días). Sin el vencimiento del día. Es una '
                f'aproximación: el posicionamiento real de los dealers no es público.</p>',
                unsafe_allow_html=True)

    # ── 5. VIX ───────────────────────────────────────────────────────
    st.markdown(T.section("VIX", "Opciones del VIX: el precio del salto",
                          "Se valoran contra el futuro de su vencimiento (Black-76), no contra el VIX de "
                          "contado. Un sesgo de calls que sube con el VIX quieto es alguien pagando por "
                          "un salto: el escenario que más daña un corto en VXX."), unsafe_allow_html=True)
    if principal:
        sv = od.sonrisa_vix(vcad[principal], r)
        st.plotly_chart(_chart_vix(sv, f"vencimiento {pd.Timestamp(principal).strftime('%d/%m/%Y')}"),
                        width="stretch", config={"displayModeBar": False})
        st.markdown(_tabla_vix(vcad, r, principal), unsafe_allow_html=True)
        st.markdown('<p class="stc-note" style="margin-top:0.5rem">Fila resaltada: el vencimiento '
                    'mensual con más interés abierto, el que se usa arriba. Los semanales tienen poca '
                    'liquidez y sonrisas con huecos.</p>', unsafe_allow_html=True)
    else:
        _error_yahoo("VIX", vix.get("error", "sin vencimiento mensual con datos"))

    # ── 6. Historia ──────────────────────────────────────────────────
    st.markdown(T.section("Historia", "La foto diaria",
                          "Una medida suelta no dice si es alta o baja. Cada día hábil, a media sesión, "
                          "una tarea automática guarda estas medidas."), unsafe_allow_html=True)
    h = _historia(HIST_PATH.stat().st_mtime) if HIST_PATH.exists() else pd.DataFrame()
    if len(h) < 5:
        st.markdown(f'<p class="stc-note">La historia empieza a acumularse ({len(h)} '
                    f'{"día" if len(h) == 1 else "días"} guardados). Las gráficas aparecen a partir de '
                    f'cinco.</p>', unsafe_allow_html=True)
    else:
        k1, k2 = st.columns(2, gap="medium")
        with k1:
            st.markdown('<div class="stc-label">IV a 30 días · SPY</div>', unsafe_allow_html=True)
            st.plotly_chart(_chart_hist(h["iv30"], "IV 30 d", T.AMBER, " %"), width="stretch",
                            config={"displayModeBar": False})
            st.markdown('<div class="stc-label">GEX total (mil M$ por 1 %)</div>', unsafe_allow_html=True)
            st.plotly_chart(_chart_hist(h["gex"], "GEX", T.WHITE, "", 2, cero=True), width="stretch",
                            config={"displayModeBar": False})
        with k2:
            st.markdown('<div class="stc-label">Sesgo 25Δ a 30 días (pts)</div>', unsafe_allow_html=True)
            st.plotly_chart(_chart_hist(h["rr25_30"], "RR25", T.LOSS, " pts", cero=True), width="stretch",
                            config={"displayModeBar": False})
            if "vix_sesgo" in h.columns:
                st.markdown('<div class="stc-label">Sesgo de calls del VIX (pts)</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(_chart_hist(h["vix_sesgo"], "Sesgo VIX", T.AMBER, " pts"),
                                width="stretch", config={"displayModeBar": False})

    # ── 7. Cómo usarla ───────────────────────────────────────────────
    st.markdown(T.section("Uso", "Cómo leerla con un corto en VXX"), unsafe_allow_html=True)
    st.markdown("""<div class="stc-card" style="padding:1rem 1.3rem">
        <div class="stc-row"><span class="k">Amortiguan + sesgo normal + calls del VIX baratas</span>
          <span class="v up">Entorno cómodo para el carry</span></div>
        <div class="stc-row"><span class="k">SPY por debajo del nivel de giro</span>
          <span class="v am">Los movimientos se amplifican: vigila el VIX3M/VIX</span></div>
        <div class="stc-row"><span class="k">Sesgo de calls del VIX subiendo con el VIX quieto</span>
          <span class="v am">Alguien paga por un salto: es tu riesgo</span></div>
        <div class="stc-row"><span class="k">Prima del evento alta antes de FOMC o CPI</span>
          <span class="v am">El salto suele venir el día del dato</span></div>
        <div class="stc-row"><span class="k">La salida de la estrategia</span>
          <span class="v">la deciden solo las dos medidas de la curva</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<p class="stc-note" style="margin-top:0.6rem">Nada de esta página cambia la regla de '
                'la estrategia: sirve para saber en qué entorno estás. Precios medios entre compra y venta; '
                'IV por Black-Scholes con el tipo a 3 meses y el dividendo del SPY.</p>',
                unsafe_allow_html=True)
