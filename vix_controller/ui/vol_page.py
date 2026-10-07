"""
vol_page.py — Página "Volatilidad": qué tan cara está y cuánto riesgo hay detrás.

Responde, en este orden:
  1. ¿Dónde está hoy el VIX frente a su historia y frente a lo que se realiza?
  2. ¿Cuánto ha cobrado de verdad el vendedor de volatilidad a cada nivel de VIX?
  3. En el entorno de hoy, ¿qué le ha pasado después a un corto en VXX?
  4. ¿Hay señales de tensión fuera del VIX (VVIX, estructura, cola, crédito)?

Fuentes: data/baro_history.parquet (índices de CBOE y ETFs, actualizado cada
día), data/vx_curve_history.parquet (liquidación de futuros de CBOE desde
2004) y data/vix_inverse_history.parquet (la estrategia).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from vix_controller.quant import vix_inverse as vi
from vix_controller.quant import vix_inverse_live as vl
from vix_controller.quant import vol_desk as vd
from vix_controller.ui import theme as T

BARO_PATH = Path("data/baro_history.parquet")
CURVE_PATH = Path("data/vx_curve_history.parquet")
HIST_PATH = Path("data/vix_inverse_history.parquet")

ne, pe = vi.num_es, vi.pct_es

# Calendario macro (fechas oficiales publicadas por la Fed y el BLS)
EVENTOS = {
    "FOMC": ["2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
             "2026-09-16", "2026-10-28", "2026-12-16"],
    "CPI": ["2026-01-14", "2026-02-12", "2026-03-11", "2026-04-14", "2026-05-13", "2026-06-10",
            "2026-07-15", "2026-08-12", "2026-09-10", "2026-10-13", "2026-11-12", "2026-12-10"],
    "Empleo": ["2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
               "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"],
}
VENTANAS = {"1 año": 1, "3 años": 3, "5 años": 5, "10 años": 10, "Todo": None}


# ──────────────────────────────────────────────────────────────────────
# Datos (cacheados por fecha de modificación del fichero)
# ──────────────────────────────────────────────────────────────────────
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(ttl=600, show_spinner="Cargando datos…")
def _datos(m_baro: float, m_curva: float, m_hist: float) -> dict:
    baro = pd.read_parquet(BARO_PATH)
    baro.index = pd.DatetimeIndex(baro.index).normalize()
    baro = baro.sort_index()
    curva = pd.read_parquet(CURVE_PATH)
    curva.index = pd.DatetimeIndex(curva.index).normalize()
    curva = curva[curva["front_ok"].astype(bool)] if "front_ok" in curva.columns else curva
    hist = pd.read_parquet(HIST_PATH)
    hist.index = pd.DatetimeIndex(hist.index).normalize()

    prima = vd.tabla_prima(baro["VIX"], baro["SPY"])
    medidas = vl.medidas_extendidas(hist, curva)
    contango = vi.señal_contango(medidas)
    vxx = vd.indice_corto_plazo(curva)
    return {
        "baro": baro, "prima": prima, "nivel": vd.prima_por_nivel(prima),
        "regimen": vd.tabla_regimen(baro["VIX"], contango, vxx),
        "contango": contango,
        "credito": vd.caida_credito(baro["HYG"], baro["IEF"]),
    }


def _fecha(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def _desde(idx: pd.DatetimeIndex, anios: int | None) -> pd.Timestamp | None:
    return None if anios is None else idx[-1] - pd.DateOffset(years=anios)


def _corta(s: pd.Series | pd.DataFrame, desde: pd.Timestamp | None):
    return s if desde is None else s[s.index >= desde]


def _pctl(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"p{v:.0f}"


# ──────────────────────────────────────────────────────────────────────
# Gráficas
# ──────────────────────────────────────────────────────────────────────
def _chart_prima(d: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=d.index, y=d["vix"].tolist(), name="VIX",
                             mode="lines", line=dict(color=T.AMBER, width=1.4),
                             hovertemplate="VIX %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rv_fut"].tolist(),
                             name="Realizada siguiente · 21 sesiones",
                             mode="lines", line=dict(color=T.WHITE, width=1.1),
                             hovertemplate="realizada después %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rv"].tolist(), name="Realizada previa · 21 sesiones",
                             mode="lines", line=dict(color=T.GRAY, width=1, dash="dot"),
                             visible="legendonly",
                             hovertemplate="realizada antes %{y:.1f}<extra></extra>"), row=1, col=1)
    p = d["prima"]
    fig.add_trace(go.Bar(x=d.index, y=p.tolist(), name="Prima",
                         marker_color=np.where(p.fillna(0) >= 0, T.PROFIT, T.LOSS).tolist(),
                         marker_line_width=0, opacity=0.85,
                         hovertemplate="prima %{y:+.1f} pts<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color=T.ZERO, line_width=1, row=2, col=1)
    pend = d[d["rv_fut"].isna() & d["vix"].notna()]
    if not pend.empty:
        fig.add_vrect(x0=pend.index[0], x1=pend.index[-1], fillcolor="rgba(154,161,169,0.10)",
                      line_width=0, row=2, col=1)
    fig.update_yaxes(title=dict(text="volatilidad %", font=dict(size=10)), row=1, col=1)
    fig.update_yaxes(title=dict(text="puntos", font=dict(size=10)), row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", bargap=0,
                      margin=dict(l=56, r=16, t=10, b=30),
                      legend=dict(orientation="h", y=1.07, x=0))
    return fig


def _chart_bandas(s: pd.Series, completa: pd.Series, nombre: str, color: str,
                  dec: int = 0) -> go.Figure:
    """Serie con líneas en sus percentiles 20/50/80 de los últimos cinco años."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.tolist(), name=nombre, mode="lines",
                             line=dict(color=color, width=1.4),
                             hovertemplate=f"{nombre} %{{y:.{dec}f}}<extra></extra>"))
    ref = completa.tail(vd.VENTANA_PERCENTIL)
    for q, dash in ((0.2, "dot"), (0.5, "dash"), (0.8, "dot")):
        v = float(ref.quantile(q))
        fig.add_hline(y=v, line_color=T.GRAY, line_width=1, line_dash=dash,
                      annotation_text=f"p{int(q * 100)} · {ne(v, dec)}", annotation_position="right",
                      annotation_font=dict(family=T.FONT_MONO, size=9, color=T.GRAY))
    fig.update_layout(height=280, hovermode="x unified", showlegend=False,
                      margin=dict(l=48, r=78, t=8, b=28))
    return fig


def _chart_estructura(b: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "VIX9D" in b.columns:
        fig.add_trace(go.Scatter(x=b.index, y=(b["VIX"] / b["VIX9D"]).tolist(), name="VIX/VIX9D",
                                 mode="lines", line=dict(color=T.GRAY, width=0.9), opacity=0.65,
                                 hovertemplate="VIX/VIX9D %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=b.index, y=(b["VIX3M"] / b["VIX"]).tolist(), name="VIX3M/VIX",
                             mode="lines", line=dict(color=T.AMBER, width=1.5),
                             hovertemplate="VIX3M/VIX %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line_color=T.LOSS, line_width=1, line_dash="dash")
    fig.update_layout(height=280, hovermode="x unified", margin=dict(l=48, r=16, t=8, b=28),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _chart_credito(cred: pd.Series, vix: pd.Series) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=cred.index, y=cred.tolist(), name="HYG/IEF frente a su máximo de 1 año",
                             mode="lines", fill="tozeroy", fillcolor="rgba(234,57,67,0.08)",
                             line=dict(color=T.LOSS, width=1),
                             hovertemplate="crédito %{y:.1f} %<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=vix.index, y=vix.tolist(), name="VIX (eje dcho.)", mode="lines",
                             line=dict(color=T.GRAY, width=1), opacity=0.7,
                             hovertemplate="VIX %{y:.1f}<extra></extra>"), secondary_y=True)
    fig.update_yaxes(ticksuffix=" %", secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    fig.update_layout(height=280, hovermode="x unified", margin=dict(l=48, r=40, t=8, b=28),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _chart_curva_indices(ci: pd.DataFrame) -> go.Figure:
    plazos = {"VIX9D": "9 días", "VIX": "30 días", "VIX3M": "3 meses", "VIX6M": "6 meses"}
    estilo = {"hoy": (T.AMBER, 2.4, None), "hace 1 semana": (T.WHITE, 1.3, "dot"),
              "hace 1 mes": (T.GRAY, 1.3, "dash")}
    fig = go.Figure()
    for nombre, fila in ci.iterrows():
        color, w, dash = estilo.get(nombre, (T.GRAY, 1, None))
        fig.add_trace(go.Scatter(x=[plazos[c] for c in fila.index], y=fila.tolist(), name=nombre,
                                 mode="lines+markers", line=dict(color=color, width=w, dash=dash),
                                 marker=dict(size=6),
                                 hovertemplate=f"{nombre} %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=280, hovermode="x unified", margin=dict(l=48, r=16, t=8, b=28),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def _chart_distribucion(vix: pd.Series, hoy: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vix.clip(upper=60).tolist(), xbins=dict(start=8, end=60, size=1),
                               marker_color=T.AMBER_DIM, opacity=0.75,
                               hovertemplate="VIX %{x}: %{y} días<extra></extra>"))
    fig.add_vline(x=hoy, line_color=T.AMBER, line_width=2,
                  annotation_text=f"hoy {ne(hoy, 1)}", annotation_position="top right",
                  annotation_font=dict(family=T.FONT_MONO, size=10, color=T.AMBER))
    fig.update_xaxes(title=dict(text="VIX al cierre (cortado en 60)", font=dict(size=10)))
    fig.update_yaxes(title=dict(text="días", font=dict(size=10)))
    fig.update_layout(height=280, showlegend=False, bargap=0.05,
                      margin=dict(l=48, r=16, t=8, b=36))
    return fig


# ──────────────────────────────────────────────────────────────────────
# Tablas y tarjetas
# ──────────────────────────────────────────────────────────────────────
def _card(label: str, big: str, cls: str, filas: list[tuple[str, str]], nota: str = "") -> str:
    rows = "".join(f'<div class="stc-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in filas)
    nota = f'<div class="stc-sub" style="margin-top:0.55rem;font-size:0.78rem">{nota}</div>' if nota else ""
    return (f'<div class="stc-card hero"><div class="stc-label">{label}</div>'
            f'<div class="stc-big {cls}" style="font-size:2.3rem">{big}</div>{rows}{nota}</div>')


def _tabla_nivel(t: pd.DataFrame, actual: str | None) -> str:
    filas = []
    for tramo, r in t.iterrows():
        if not r["dias"]:
            continue
        cls = ' class="open"' if tramo == actual else ""
        filas.append(f'<tr{cls}><td>VIX {tramo}</td><td>{ne(r["peso"], 0)} %</td>'
                     f'<td>{ne(r["vix"], 1)}</td><td>{ne(r["rv_fut"], 1)}</td>'
                     f'<td>{T.arrow(r["prima"])} {ne(r["prima"], 1, signo=True)}</td>'
                     f'<td>{ne(r["gana"], 0)} %</td><td>{ne(r["p10"], 1, signo=True)}</td></tr>')
    return ('<div style="overflow-x:auto"><table class="stc-table"><tr><th>Nivel del VIX</th>'
            '<th>% del tiempo</th><th>VIX medio</th><th>Realizada después</th>'
            '<th>Prima mediana</th><th>VIX &gt; realizada</th><th>Peor 10 %</th></tr>'
            + "".join(filas) + "</table></div>")


def _tabla_regimen(t: pd.DataFrame, actual: str | None) -> str:
    filas = []
    for _, r in t.iterrows():
        cls = "open" if r["regimen"] == actual else ("lose" if r["siempre_media"] < 0 else "")
        aviso = "" if r["fiable"] else ' <span style="color:var(--gray-light)">· muestra corta</span>'
        filas.append(
            f'<tr class="{cls}"><td>{T.esc(r["regimen"])}{aviso}</td><td>{ne(r["peso"], 1)} %</td>'
            f'<td>{pe(r["siempre_media"], 1)}</td><td>{ne(r["siempre_gana"], 0)} %</td>'
            f'<td>{pe(r["siempre_p5"], 1)}</td><td>{pe(r["regla_media"], 1)}</td>'
            f'<td>{pe(r["regla_p5"], 1)}</td></tr>')
    return ('<div style="overflow-x:auto"><table class="stc-table"><tr><th>Entorno al cierre</th>'
            '<th>% del tiempo</th><th>Siempre corto · media</th><th>Siempre corto · % con ganancia</th>'
            '<th>Siempre corto · peor 5 %</th><th>Con la regla · media</th>'
            '<th>Con la regla · peor 5 %</th></tr>' + "".join(filas) + "</table></div>")


def _lectura(nombre: str, t: dict) -> tuple[str, str]:
    """(texto, clase css) con reglas simples y a la vista."""
    v, p = t["valor"], t["pct"]
    if nombre == "VVIX":
        if p >= 80:
            return "Alto: el mercado paga por calls del VIX. Riesgo de salto para el corto", "dn"
        return ("Tranquilo", "up") if p <= 20 else ("Normal", "")
    if nombre == "VIX3M/VIX":
        if v < 1:
            return "Invertida: el estrés es inmediato", "dn"
        return ("Plana: poco colchón", "am") if v < 1.05 else ("Contango normal", "up")
    if nombre == "VIX/VIX9D":
        return ("Los 9 días por encima del mes: tensión a muy corto plazo", "dn") if v < 1 \
            else ("Normal", "up")
    if nombre == "SKEW":
        return ("Demanda alta de protección de cola", "am") if p >= 80 else ("Normal", "")
    if nombre == "Crédito":
        if v <= -3:
            return "El crédito se deteriora: suele adelantarse al VIX", "dn"
        return ("Vigilar", "am") if v <= -1 else ("Sano", "up")
    return "", ""


def _tabla_termometros(filas: list[tuple[str, dict, int, str]]) -> str:
    html = []
    for nombre, t, dec, suf in filas:
        if not t:
            continue
        txt, cls = _lectura(nombre, t)
        html.append(f'<tr><td>{nombre}</td><td>{ne(t["valor"], dec)}{suf}</td>'
                    f'<td>{_pctl(t["pct"])}</td><td>{_pctl(t["pct_1a"])}</td>'
                    f'<td>{ne(t["cambio_mes"], dec, signo=True)}{suf}</td>'
                    f'<td class="{cls}" style="text-align:left">{txt}</td></tr>')
    return ('<div style="overflow-x:auto"><table class="stc-table"><tr><th>Indicador</th><th>Hoy</th>'
            '<th>Percentil 5 años</th><th>Percentil 1 año</th><th>Cambio 1 mes</th>'
            '<th style="text-align:left">Lectura</th></tr>' + "".join(html) + "</table></div>")


def _eventos(hoy: pd.Timestamp) -> str:
    chips = []
    for nombre, fechas in EVENTOS.items():
        for f in fechas:
            dias = (pd.Timestamp(f) - hoy).days
            if 0 <= dias <= 14:
                chips.append((dias, nombre, pd.Timestamp(f)))
    if not chips:
        return T.badge("Sin datos macro relevantes en los próximos 14 días")
    out = []
    for dias, nombre, f in sorted(chips):
        kind = "loss" if dias <= 2 else ("amber" if dias <= 5 else "")
        cuando = "hoy" if dias == 0 else ("mañana" if dias == 1 else f"en {dias} días")
        out.append(T.badge(f"{nombre} · {f.strftime('%d/%m')} · {cuando}", kind))
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────
# Página
# ──────────────────────────────────────────────────────────────────────
def render() -> None:
    faltan = [p for p in (BARO_PATH, CURVE_PATH, HIST_PATH) if not p.exists()]
    if faltan:
        st.markdown(f'<div class="stc-alert loss"><b>Faltan datos</b>No existe '
                    f'{", ".join(str(p) for p in faltan)}. Los generan las GitHub Actions diarias.'
                    f'</div>', unsafe_allow_html=True)
        return
    D = _datos(_mtime(BARO_PATH), _mtime(CURVE_PATH), _mtime(HIST_PATH))
    baro, prima = D["baro"], D["prima"]
    vix = baro["VIX"].dropna()
    hoy_v = float(vix.iloc[-1])
    f_dato = vix.index[-1]

    st.markdown(f"""{T.eyebrow("Volatilidad")}
        <div class="stc-h1">¿Está cara la volatilidad y cuánto riesgo hay detrás?</div>
        <p class="stc-lead">El vendedor de volatilidad cobra la diferencia entre lo que el mercado
        espera (VIX) y lo que luego ocurre. Aquí se mide esa prima con más de treinta años de datos,
        lo que ha hecho un corto en VXX en cada entorno y las señales de tensión que no salen en el
        VIX.</p>
        <div class="stc-badges">{T.badge(f"Cierre del {_fecha(f_dato)}")}
          {T.badge(f"VIX desde {vix.index[0].year} · SPY desde {prima.index[0].year}")}</div>""",
                unsafe_allow_html=True)

    # ── 1. Lectura de hoy ───────────────────────────────────────────
    ult = prima.dropna(subset=["rv"]).iloc[-1]
    tramo = str(vd.tramo_vix(pd.Series([hoy_v])).iloc[0])
    niv = D["nivel"].loc[tramo] if tramo in D["nivel"].index else None
    contango_hoy = bool(D["contango"].iloc[-1])
    reg_hoy = vd.regimen(pd.Series([hoy_v]), pd.Series([contango_hoy])).iloc[0]
    reg = D["regimen"]
    fila_reg = reg[reg["regimen"] == reg_hoy].iloc[0] if (reg["regimen"] == reg_hoy).any() else None

    st.markdown(T.section("Hoy", "Lectura de hoy"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.markdown(_card("VIX · implícita a 30 días", ne(hoy_v, 2), "amber",
                          [("Percentil desde 1990", _pctl(vd.percentil(vix))),
                           ("Percentil 1 año", _pctl(vd.percentil(vix.tail(vd.ANUAL)))),
                           ("Mediana histórica", ne(float(vix.median()), 1))]),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Realizada · SPY, 21 sesiones", ne(float(ult["rv"]), 1), "white",
                          [("VIX − realizada", f'{ne(float(ult["prima_hoy"]), 1, signo=True)} pts'),
                           ("Percentil de la realizada", _pctl(vd.percentil(prima["rv"])))],
                          "Lo que ya pasó. La prima que se cobra de verdad es contra la realizada "
                          "del mes que viene."), unsafe_allow_html=True)
    with c3:
        if niv is not None:
            st.markdown(_card(f"Prima histórica con VIX {tramo}", f'{ne(niv["prima"], 1, signo=True)}',
                              "amber" if niv["prima"] > 0 else "loss",
                              [("Veces con VIX > realizada", f'{ne(niv["gana"], 0)} %'),
                               ("Peor 10 % de los casos", f'{ne(niv["p10"], 1, signo=True)} pts'),
                               ("Días con este nivel", f'{int(niv["dias"]):,}'.replace(",", "."))],
                              "Puntos de volatilidad: VIX menos la realizada de las 21 sesiones "
                              "siguientes."), unsafe_allow_html=True)
    with c4:
        if fila_reg is not None:
            tramo_r, curva_r = reg_hoy.split(" · ")
            st.markdown(_card(f"Entorno de hoy · {curva_r}", T.esc(tramo_r), "white",
                [("Corto VXX · media 21 s", pe(fila_reg["siempre_media"], 1)),
                 ("Corto VXX · peor 5 %", pe(fila_reg["siempre_p5"], 1)),
                 ("Con la regla · peor 5 %", pe(fila_reg["regla_p5"], 1))],
                "Lo que hizo un corto en VXX en las 21 sesiones siguientes a días como hoy."),
                unsafe_allow_html=True)
    st.markdown(f'<div class="stc-badges" style="margin-top:0.9rem">'
                f'<span class="stc-label" style="margin-right:0.4rem">Datos macro</span>'
                f'{_eventos(pd.Timestamp.now().normalize())}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    v = st.segmented_control("Ventana de las gráficas", options=list(VENTANAS), default="5 años",
                             key="vol_ventana") or "5 años"
    desde = _desde(baro.index, VENTANAS[v])

    # ── 2. La prima ─────────────────────────────────────────────────
    st.markdown(T.section("La prima", "Implícita frente a lo que luego se realizó",
                          "Cada día se compara el VIX con la volatilidad que el SPY tuvo de verdad en "
                          "las 21 sesiones siguientes. Verde: el vendedor de volatilidad cobró. Rojo: "
                          "pagó. Las últimas 21 sesiones aún no tienen resultado (zona gris)."),
                unsafe_allow_html=True)
    st.plotly_chart(_chart_prima(_corta(prima, desde)), width="stretch",
                    config={"displayModeBar": False})
    x = prima.dropna(subset=["prima"])
    st.markdown(f'<p class="stc-note">Desde {x.index[0].year}: el VIX ha superado a la realizada '
                f'posterior el {ne((x["prima"] > 0).mean() * 100, 0)} % de los días, con una prima '
                f'mediana de {ne(float(x["prima"].median()), 1, signo=True)} puntos. Las pérdidas llegan '
                f'concentradas: cuando la prima es negativa, su mediana es '
                f'{ne(float(x.loc[x["prima"] < 0, "prima"].median()), 1, signo=True)} puntos.</p>',
                unsafe_allow_html=True)
    st.markdown(_tabla_nivel(D["nivel"], tramo), unsafe_allow_html=True)
    st.markdown('<p class="stc-note" style="margin-top:0.6rem">Con el VIX bajo la prima en puntos es '
                'pequeña y el colchón ante un salto, escaso. Fila resaltada: el nivel de hoy.</p>',
                unsafe_allow_html=True)

    # ── 3. Régimen ──────────────────────────────────────────────────
    st.markdown(T.section("Régimen", "Lo que hizo un corto en VXX después de cada entorno",
                          "El entorno se fija con el cierre de cada día: nivel del VIX y curva en "
                          "contango o backwardation según la regla de la estrategia (M2/M1 o VIX3M/VIX "
                          "por encima de 1,00). Después se mide el corto en las 21 sesiones "
                          "siguientes, siempre dentro y aplicando la regla."), unsafe_allow_html=True)
    st.markdown(_tabla_regimen(reg, reg_hoy), unsafe_allow_html=True)
    st.markdown(f'<p class="stc-note" style="margin-top:0.6rem">VXX reconstruido con la liquidación '
                f'diaria de CBOE (índice de futuros a un mes, la metodología que replica el VXX), '
                f'{_fecha(reg.attrs.get("desde"))} → {_fecha(reg.attrs.get("hasta"))}. Corto a '
                f'exposición constante y sin costes: mide el riesgo de cada entorno, no replica el '
                f'backtest. Fila resaltada: el entorno de hoy; en rojo, los entornos donde el corto '
                f'pierde de media.</p>', unsafe_allow_html=True)

    # ── 4. Termómetros ──────────────────────────────────────────────
    st.markdown(T.section("Tensión", "Termómetros de riesgo fuera del VIX",
                          "El VIX puede estar tranquilo mientras otros mercados ya pagan por "
                          "protección. Percentiles sobre los últimos cinco años: el SKEW y el VVIX "
                          "han subido de nivel con los años y la historia completa engaña."),
                unsafe_allow_html=True)
    b = baro
    term = [("VVIX", vd.termometro(b["VVIX"]), 1, ""),
            ("VIX3M/VIX", vd.termometro(b["VIX3M"] / b["VIX"]), 3, ""),
            ("VIX/VIX9D", vd.termometro(b["VIX"] / b["VIX9D"]), 3, ""),
            ("SKEW", vd.termometro(b["SKEW"]), 1, ""),
            ("Crédito", vd.termometro(D["credito"]), 1, " %")]
    st.markdown(_tabla_termometros(term), unsafe_allow_html=True)

    bc = _corta(b, desde)
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        st.markdown('<div class="stc-label" style="margin-top:1.2rem">VVIX · volatilidad del VIX</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_bandas(bc["VVIX"].dropna(), b["VVIX"].dropna(), "VVIX", T.AMBER),
                        width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="stc-note">Precio de las opciones sobre el VIX. Alto con el VIX bajo '
                    'significa que alguien está comprando calls del VIX: es el entorno de los saltos '
                    'que castigan al corto.</p>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="stc-label" style="margin-top:1.2rem">Estructura de los índices</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_estructura(bc[["VIX", "VIX3M", "VIX9D"]].dropna(how="any")),
                        width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="stc-note">Por encima de 1,00, normal. VIX/VIX9D se gira antes: avisa de '
                    'tensión de días; VIX3M/VIX es la segunda medida de la estrategia.</p>',
                    unsafe_allow_html=True)
    g3, g4 = st.columns(2, gap="medium")
    with g3:
        st.markdown('<div class="stc-label" style="margin-top:1.2rem">SKEW · precio de la cola</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_bandas(bc["SKEW"].dropna(), b["SKEW"].dropna(), "SKEW", T.WHITE),
                        width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="stc-note">Cuánto se paga por puts muy fuera del dinero. Contexto, no '
                    'señal: su capacidad para anticipar caídas es baja.</p>', unsafe_allow_html=True)
    with g4:
        st.markdown('<div class="stc-label" style="margin-top:1.2rem">Crédito · HYG frente a IEF</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_credito(_corta(D["credito"], desde), _corta(vix, desde)),
                        width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="stc-note">Bonos high yield frente a Tesoro a 7-10 años, en % bajo su '
                    'máximo de un año. Si cae mientras el VIX sigue quieto, el crédito va por '
                    'delante.</p>', unsafe_allow_html=True)
    g5, g6 = st.columns(2, gap="medium")
    with g5:
        st.markdown('<div class="stc-label" style="margin-top:1.2rem">Curva de índices de volatilidad'
                    '</div>', unsafe_allow_html=True)
        ci = vd.curva_indices(b, {"hoy": f_dato, "hace 1 semana": f_dato - pd.Timedelta(days=7),
                                  "hace 1 mes": f_dato - pd.DateOffset(months=1)})
        st.plotly_chart(_chart_curva_indices(ci), width="stretch", config={"displayModeBar": False})
    with g6:
        st.markdown(f'<div class="stc-label" style="margin-top:1.2rem">VIX · distribución desde '
                    f'{vix.index[0].year}</div>', unsafe_allow_html=True)
        st.plotly_chart(_chart_distribucion(vix, hoy_v), width="stretch",
                        config={"displayModeBar": False})
    st.markdown('<p class="stc-note">Fuentes: índices de CBOE (VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW) y '
                'ETFs (SPY, HYG, IEF) vía yfinance, actualizados cada día; liquidación de futuros de '
                'CBOE. Realizada = rendimientos diarios del SPY, anualizada.</p>',
                unsafe_allow_html=True)
