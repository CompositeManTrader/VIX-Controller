"""
strategy_page.py — Página "Estrategia": seguimiento del modelo VIX Inverse.

Responde, de arriba abajo, a las preguntas en el orden en que se hacen:
  1. ¿Estoy dentro o fuera, y qué hago en la próxima apertura?
  2. ¿A qué distancia está cada medida de 1,00? ¿Se está girando hoy?
  3. ¿Se puede seguir operando el modelo (condiciones de parada)?
  4. ¿Cómo va la cartera frente al SPY y al SPY a igual volatilidad?
  5. ¿Está derivando del backtest congelado?
  6. Operaciones, sustos y alertas enviadas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from vix_controller import alerts as al
from vix_controller.quant import vix_inverse as vi
from vix_controller.quant import vix_inverse_live as vl
from vix_controller.ui import theme as T

HIST_PATH = Path("data/vix_inverse_history.parquet")
CURVE_PATH = Path("data/vx_curve_history.parquet")
ALERTS_PATH = Path("data/vix_inverse_alerts.json")

ne, pe = vi.num_es, vi.pct_es

# Referencia: informe canónico (corto estático, ejecución en apertura), 2013-05 → 2026-07
REF_INFORME = {"cagr": 17.91, "vol": 19.01, "sharpe": 0.96, "dd": -30.93,
               "beta": 1.06, "alfa": 3.81, "ir": 0.60}
REF_SPY = {"cagr": 13.9, "vol": 17.0, "sharpe": 0.85, "dd": -33.7}
EPISODIOS_PUBLICADOS = [
    ("24/08/2015", "+74 %", "−22,9 %", "−11,7 %"),
    ("05/02/2018", "+110 %", "−25,5 %", "−9,9 %"),
    ("25/02/2020", "+339 %", "−26,2 %", "−30,9 %"),
    ("05/08/2024", "+112 %", "−13,2 %", "−8,2 %"),
    ("04/04/2025", "+67 %", "−21,6 %", "−14,5 %"),
]


# ──────────────────────────────────────────────────────────────────────
# Datos
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_history(mtime: float) -> pd.DataFrame:
    h = pd.read_parquet(HIST_PATH)
    h.index = pd.DatetimeIndex(h.index).normalize()
    return h


@st.cache_data(ttl=300, show_spinner=False)
def load_curve(mtime: float) -> pd.DataFrame:
    c = pd.read_parquet(CURVE_PATH)
    c.index = pd.DatetimeIndex(c.index).normalize()
    if "front_ok" in c.columns:
        c = c[c["front_ok"].astype(bool)]
    return c


def load_alerts() -> dict:
    try:
        return json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _fecha(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y") if d is not None and not pd.isna(d) else "—"


# ──────────────────────────────────────────────────────────────────────
# Bloques
# ──────────────────────────────────────────────────────────────────────
def _gauge(v: float, lo: float = 0.85, hi: float = 1.20) -> str:
    """Barra horizontal con la marca de 1,00. El tramo coloreado va de 1,00 al valor."""
    def x(z):
        return float(np.clip((z - lo) / (hi - lo), 0, 1) * 100)
    one, cur = x(1.0), x(v)
    a, b = sorted((one, cur))
    color = T.PROFIT if v > 1 else T.LOSS
    return (f'<div class="stc-gauge"><div class="fill" style="left:{a:.1f}%;width:{b - a:.1f}%;'
            f'background:{color}"></div><div class="one" style="left:{one:.1f}%"></div></div>'
            f'<div class="stc-gauge-scale"><span>{ne(lo, 2)}</span><span>1,00</span>'
            f'<span>{ne(hi, 2)}</span></div>')


def _medida(nombre: str, fuente: str, v: float) -> str:
    dist = v - 1.0
    estado = "contango" if v > 1 else "invertida"
    cls = "up" if v > 1 else "dn"
    return (f'<div style="margin-bottom:1.05rem">'
            f'<div class="stc-label">{nombre} · {fuente}</div>'
            f'<div style="display:flex;align-items:baseline;gap:0.75rem;margin-top:0.25rem">'
            f'<span class="stc-val">{ne(v, 4)}</span>'
            f'<span class="{cls}" style="font-family:var(--mono);font-size:0.82rem">'
            f'{T.arrow(dist)} {ne(dist * 100, 2, signo=True)} % a 1,00 · {estado}</span></div>'
            f'{_gauge(v)}</div>')


def _hero(e: dict, peso: float, alert_state: dict) -> None:
    dentro, prox = e["dentro"], e["prox_dentro"]
    c1, c2, c3 = st.columns([1.0, 1.25, 1.05], gap="medium")
    with c1:
        txt, cls = ("DENTRO", "amber") if dentro else ("FUERA", "white")
        sub = (f"Corto de VXX · sleeve del {ne(peso * 100, 0)} % de la cartera"
               if dentro else "Sin posición en VXX · sleeve en efectivo")
        st.markdown(f"""<div class="stc-card hero {'accent' if dentro else ''}">
            <div class="stc-label">Posición vigente</div>
            <div class="stc-big {cls}">{txt}</div>
            <div class="stc-sub">{sub}</div>
            <div class="stc-sub" style="margin-top:0.6rem;font-family:var(--mono);font-size:0.76rem">
              {e['racha']} sesiones en esta posición</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stc-card hero">
            <div class="stc-label" style="color:var(--amber);margin-bottom:0.8rem">
              Las dos medidas · cierre del {_fecha(e['fecha_dato'])}</div>
            {_medida("M2/M1", "futuros", e['ratio_m2m1'])}
            {_medida("VIX3M/VIX", "índices", e['ratio_vix3m'])}
            <div class="stc-sub" style="font-size:0.78rem">Dentro mientras cualquiera esté por
              encima de 1,00. Fuera solo cuando las dos cierran en 1,00 o menos.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        ejec = al.siguiente_sesion(pd.Timestamp(e["fecha_dato"]).date())
        if e["cambio_pendiente"] and not prox:
            accion, kind = f"Recomprar VXX en la apertura del {al.fmt(ejec)}", "loss"
        elif e["cambio_pendiente"] and prox:
            accion, kind = f"Vender VXX en la apertura del {al.fmt(ejec)}", "profit"
        else:
            accion, kind = ("Mantener el corto. Sin acción." if dentro
                            else "Seguir fuera. Sin acción."), ""
        correo = alert_state.get("correo", "sin configurar")
        ult = alert_state.get("ultima_pasada")
        ult_txt = pd.Timestamp(ult).tz_convert("America/Mexico_City").strftime("%d/%m %H:%M") if ult else "—"
        st.markdown(f"""<div class="stc-card hero {kind}">
            <div class="stc-label">Ejecución · apertura del {al.fmt(ejec)}</div>
            <div style="font-family:var(--display);font-weight:500;font-size:1.3rem;line-height:1.25;
                 color:var(--white);margin:0.55rem 0 0.9rem">{accion}</div>
            <div class="stc-row"><span class="k">Alerta por correo</span>
              <span class="v {'up' if correo == 'configurado' else 'am'}">{correo}</span></div>
            <div class="stc-row"><span class="k">Última evaluación automática</span>
              <span class="v">{ult_txt}</span></div>
            <div class="stc-row"><span class="k">Cierre evaluado</span>
              <span class="v">{_fecha(alert_state.get('evaluado'))}</span></div>
        </div>""", unsafe_allow_html=True)

    if e["cambio_pendiente"] and not prox:
        st.markdown(f'<div class="stc-alert loss"><b>Salida</b>Las dos medidas han cerrado en '
                    f'backwardation. La regla manda recomprar toda la posición en la apertura del '
                    f'{al.fmt(ejec)}.</div>', unsafe_allow_html=True)
    elif dentro and e["una_invertida"]:
        cual = "M2/M1" if e["ratio_m2m1"] <= 1 else "VIX3M/VIX"
        otra = e["ratio_vix3m"] if cual == "M2/M1" else e["ratio_m2m1"]
        st.markdown(f'<div class="stc-alert"><b>Aviso · una medida invertida</b>{cual} ha cerrado '
                    f'bajo 1,00 y la otra sigue en {ne(otra, 3)}. Aún no es salida, pero es el patrón '
                    f'del 2 de agosto de 2024: si la otra cae mañana, se sale en la apertura siguiente.'
                    f'</div>', unsafe_allow_html=True)


def _intradia(r1: float | None, r2: float | None, ts: str | None) -> None:
    if r1 is None or r2 is None:
        return
    dentro = (r1 > 1) or (r2 > 1)
    st.markdown(f"""<div class="stc-card" style="margin-top:0.9rem;padding:0.85rem 1.25rem">
        <div style="display:flex;flex-wrap:wrap;gap:1.6rem;align-items:center">
          <div class="stc-label" style="min-width:150px">Si cerrara ahora<br>
            <span style="color:var(--gray-light)">CBOE · ~15 min de retraso</span></div>
          <div><div class="stc-label">M2/M1</div><div class="stc-val" style="font-size:1.1rem">
            <span class="{'up' if r1 > 1 else 'dn'}">{T.arrow(r1, 1)}</span> {ne(r1, 4)}</div></div>
          <div><div class="stc-label">VIX3M/VIX</div><div class="stc-val" style="font-size:1.1rem">
            <span class="{'up' if r2 > 1 else 'dn'}">{T.arrow(r2, 1)}</span> {ne(r2, 4)}</div></div>
          <div><div class="stc-label">Señal provisional</div><div class="stc-val"
            style="font-size:1.1rem;color:{'var(--amber)' if dentro else 'var(--white)'}">
            {'DENTRO' if dentro else 'FUERA'}</div></div>
          <div class="stc-sub" style="flex:1;min-width:220px;font-size:0.78rem">Solo orientativo. La
            señal oficial se toma con la liquidación del cierre{(' · ' + ts) if ts else ''}.</div>
        </div></div>""", unsafe_allow_html=True)


def _protocolo(hist: pd.DataFrame) -> None:
    conds = vl.condiciones_parada(hist)
    estado, expl = vl.veredicto_protocolo(conds)
    kind = {"OPERABLE": "profit", "REVISIÓN": "", "APAGADO": "loss"}[estado]
    color = {"OPERABLE": "var(--profit)", "REVISIÓN": "var(--amber)", "APAGADO": "var(--loss)"}[estado]
    st.markdown(T.section("Protocolo de operación", "¿Se puede seguir operando?",
                          "Condiciones estructurales fijadas en frío antes de operar. Una dispara "
                          "revisión inmediata; dos, el apagado. Ninguna mira el resultado: miran lo "
                          "que se ve el día que pasa."), unsafe_allow_html=True)
    def _nombre(n, ref):
        return (f'<span style="font-family:var(--body);color:var(--white)">{T.esc(n)}</span>'
                f'<br><span class="mu" style="font-size:0.7rem">{T.esc(ref)}</span>')

    def _estado(c):
        if c.disparada:
            return '<span class="dn">▼ DISPARADA</span>'
        return '<span class="am">CERCA</span>' if c.cerca else '<span class="up">▲ OK</span>'
    filas = "".join(
        f'<tr><td>{_nombre(c.nombre, c.referencia)}</td><td>{T.esc(c.valor)}</td>'
        f'<td class="mu">{T.esc(c.umbral)}</td><td>{_estado(c)}</td></tr>'
        for c in conds)
    filas += (f'<tr><td>{_nombre("Riesgo de producto", "emisión suspendida, cambio de emisor, prima >2 % sobre NAV, corto restringido")}</td>'
              '<td>revisión manual</td><td class="mu">apagado inmediato</td>'
              '<td><span class="am">MANUAL</span></td></tr>')
    a, b = st.columns([0.9, 3.1], gap="medium")
    with a:
        st.markdown(f"""<div class="stc-card {kind}"><div class="stc-label">Veredicto</div>
            <div class="stc-big" style="font-size:2.1rem;color:{color}">{estado}</div>
            <div class="stc-sub">{T.esc(expl)}</div></div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""<div class="stc-card" style="padding:0.4rem 0.9rem"><table class="stc-table">
            <tr><th>Condición · referencia</th><th>Hoy</th><th>Umbral</th><th>Estado</th></tr>
            {filas}</table></div>""", unsafe_allow_html=True)


def _chart_capital(c: pd.DataFrame) -> go.Figure:
    k = c.attrs.get("k", np.nan)
    eq = (1 + c).cumprod()
    dd = eq / eq.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05)
    estilo = {"cartera": ("Cartera con sleeve", T.AMBER, 2.4, None),
              "spy_apalancado": (f"SPY {ne(k, 2)}x · misma volatilidad", T.WHITE, 1.5, None),
              "spy": ("SPY", T.GRAY, 1.2, "dot")}
    for col in ("spy", "spy_apalancado", "cartera"):
        nombre, color, w, dash = estilo[col]
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col].tolist(), name=nombre, mode="lines",
                                 line=dict(color=color, width=w, dash=dash),
                                 hovertemplate="%{x|%d/%m/%Y}  %{y:.2f}x<extra>" + nombre + "</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=(dd[col] * 100).tolist(), name=nombre, mode="lines",
                                 line=dict(color=color, width=1.1 if col != "cartera" else 1.6, dash=dash),
                                 showlegend=False,
                                 hovertemplate="%{x|%d/%m/%Y}  %{y:.1f} %<extra>" + nombre + "</extra>"),
                      row=2, col=1)
    fig.update_yaxes(type="log", title=dict(text="capital (log)", font=dict(size=10)), row=1, col=1)
    fig.update_yaxes(ticksuffix=" %", title=dict(text="caída", font=dict(size=10)), row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(l=56, r=16, t=10, b=30),
                      legend=dict(orientation="h", y=1.04, x=0, xanchor="left"))
    return fig


def _tramos(mask: pd.Series) -> list[tuple]:
    """(inicio, fin) de cada racha True consecutiva."""
    out, idx, i = [], mask.index, 0
    v = mask.to_numpy()
    while i < len(v):
        if v[i]:
            j = i
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            out.append((idx[i], idx[j]))
            i = j + 1
        else:
            i += 1
    return out


def _chart_medidas(m: pd.DataFrame, desde: pd.Timestamp | None) -> go.Figure:
    h = m[m.index >= desde] if desde is not None else m
    largo = len(h) > 800
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h["ratio_vix3m"].tolist(), name="VIX3M/VIX · índices",
                             mode="lines", line=dict(color=T.WHITE, width=0.8 if largo else 1.1),
                             opacity=0.55 if largo else 1.0,
                             hovertemplate="%{x|%d/%m/%Y}  VIX3M/VIX %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=h.index, y=h["ratio_m2m1"].tolist(), name="M2/M1 · futuros",
                             mode="lines", line=dict(color=T.AMBER, width=1.0 if largo else 1.5),
                             hovertemplate="%{x|%d/%m/%Y}  M2/M1 %{y:.3f}<extra></extra>"))
    # Días fuera como marcas en la base: con años en pantalla, un sombreado de
    # dos o tres sesiones no se ve
    fuera = h.index[~h["pos"].astype(bool)]
    fig.add_trace(go.Scatter(x=fuera, y=[0.63] * len(fuera), name="Fuera de mercado", mode="markers",
                             marker=dict(symbol="line-ns", size=9, line=dict(color=T.LOSS, width=1.2)),
                             hoverinfo="skip"))
    # Tramos fuera de mercado: sombreado DESPUÉS de los traces (plotly ≥ 6)
    for x0, x1 in _tramos(~h["pos"].astype(bool)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(234,57,67,0.13)", line_width=0, layer="below")
    fig.add_hline(y=1.0, line_color=T.WHITE, line_width=1, line_dash="dash",
                  annotation_text="1,00", annotation_position="right",
                  annotation_font=dict(family=T.FONT_MONO, size=10, color=T.GRAY))
    previo = h[~h["backtest"].astype(bool)]
    if not previo.empty and h["backtest"].any():
        x_bt = h.index[h["backtest"].astype(bool)][0]
        fig.add_vline(x=x_bt, line_color=T.GRAY, line_width=1, line_dash="dot")
        fig.add_annotation(x=x_bt, y=0.06, yref="paper", xanchor="left", yanchor="bottom",
                           showarrow=False,
                           text=" inicio del backtest →", font=dict(family=T.FONT_MONO, size=10,
                                                                    color=T.GRAY))
        fig.add_annotation(x=previo.index[0], y=0.06, yref="paper", xanchor="left", yanchor="bottom",
                           showarrow=False, text=" reconstruido con la regla · archivo CBOE",
                           font=dict(family=T.FONT_MONO, size=10, color=T.MUTED))
    fig.update_yaxes(range=[0.6, 1.45], tickformat=".2f")
    fig.update_layout(height=380, hovermode="x unified", margin=dict(l=48, r=40, t=10, b=30),
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


def _tabla_operaciones(ops: list[dict], peso: float, n: int | None) -> str:
    filas = []
    for o in (ops[::-1][:n] if n else ops[::-1]):
        cls = "open" if o["abierta"] else ("lose" if o["ret"] <= 0 else "")
        filas.append(
            f'<tr class="{cls}"><td>{_fecha(o["f_entrada"])}</td>'
            f'<td>{"abierta" if o["abierta"] else _fecha(o["f_salida"])}</td>'
            f'<td>{o["dias"]}</td>'
            f'<td>{ne(o["px_entrada"], 2) if o["px_entrada"] else "—"}</td>'
            f'<td>{ne(o["px_salida"], 2) if o["px_salida"] else "—"}</td>'
            f'<td>{T.arrow(o["ret"])} {pe(o["ret"] * 100, 1)}</td>'
            f'<td>{pe(o["ret"] * peso * 100, 2)}</td></tr>')
    return ('<table class="stc-table"><tr><th>Entrada</th><th>Salida</th><th>Sesiones</th>'
            '<th>VXX entrada</th><th>VXX salida</th><th>Sleeve</th><th>Cartera</th></tr>'
            + "".join(filas) + "</table>")


# ──────────────────────────────────────────────────────────────────────
# Página
# ──────────────────────────────────────────────────────────────────────
def render(live: dict | None = None) -> None:
    live = live or {}
    if not HIST_PATH.exists():
        st.markdown('<div class="stc-alert loss"><b>Sin datos de la estrategia</b>Falta '
                    'data/vix_inverse_history.parquet. Se genera con scripts/seed_vix_inverse.py '
                    'y la actualiza la GitHub Action diaria.</div>', unsafe_allow_html=True)
        return

    hist = load_history(_mtime(HIST_PATH))
    curva = load_curve(_mtime(CURVE_PATH)) if CURVE_PATH.exists() else None
    alert_state = load_alerts()
    e = vl.estado(hist, curva)

    # ── Cabecera de la página ───────────────────────────────────────
    hoy = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    retraso = int(np.busday_count(pd.Timestamp(e["fecha_dato"]).date(), hoy.date()))
    datos_badge = ("profit" if retraso <= 1 else "loss")
    st.markdown(f"""<div style="animation:enter var(--dur-enter) var(--ease)">
        {T.eyebrow("Estrategia · VIX Inverse")}
        <div class="stc-h1">Corto de VXX mientras la curva paga</div>
        <p class="stc-lead">Sleeve del 15–20 % de una cartera con SPY. Se cobra el carry de la curva de
        futuros del VIX y se sale cuando las dos medidas de pendiente se invierten. Señal con el cierre,
        ejecución en la apertura siguiente. Sin stop, sin objetivo, sin reajustar el tamaño.</p>
        <div class="stc-badges">{T.badge("Operable acotado · no validado", "amber")}
          {T.badge("Señal congelada el " + _fecha(vi.FECHA_CONGELACION))}
          {T.badge(f"Dato al {_fecha(e['fecha_dato'])}", datos_badge)}
          {T.badge(f"{len(hist):,} sesiones desde 2013".replace(",", "."))}</div>
    </div>""", unsafe_allow_html=True)
    if retraso > 1:
        st.markdown(f'<div class="stc-alert loss"><b>Dato atrasado</b>El último cierre evaluado es del '
                    f'{_fecha(e["fecha_dato"])} ({retraso} sesiones hábiles de retraso). No operes con '
                    f'esta pantalla hasta que se actualice.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
    peso = st.segmented_control("Peso del sleeve", options=[0.15, 0.16, 0.20], default=0.20,
                                format_func=lambda x: {0.15: "15 %", 0.16: "16 % · medio Kelly",
                                                       0.20: "20 % · cifras publicadas"}[x],
                                key="vinv_peso") or 0.20
    _hero(e, peso, alert_state)
    _intradia(live.get("r1"), live.get("r2"), live.get("ts"))

    # ── Protocolo ───────────────────────────────────────────────────
    _protocolo(hist)

    # ── Las dos medidas en el tiempo ────────────────────────────────
    st.markdown(T.section("La señal", "Las dos medidas frente a 1,00",
                          "Sombreado: tramos fuera de mercado (las dos invertidas). El aviso vive en "
                          "el frente de la curva: el 2/08/2024 el VIX3M/VIX se invirtió un día antes "
                          "que el M2/M1."), unsafe_allow_html=True)
    medidas = vl.medidas_extendidas(hist, curva)
    fin = medidas.index[-1]
    ventanas = {"1 año": fin - pd.DateOffset(years=1), "3 años": fin - pd.DateOffset(years=3),
                "5 años": fin - pd.DateOffset(years=5), "Backtest · 2013": hist.index[0],
                f"Todo · {medidas.index[0].year}": None}
    v = st.segmented_control("Ventana", options=list(ventanas), default=list(ventanas)[-1],
                             key="vinv_ventana_m") or list(ventanas)[-1]
    st.plotly_chart(_chart_medidas(medidas, ventanas[v]), width="stretch",
                    config={"displayModeBar": False})
    previo = medidas[~medidas["backtest"]]
    if not previo.empty:
        cont = vi.señal_contango(previo)
        n_prev = f"{len(previo):,}".replace(",", ".")
        st.markdown(f'<p class="stc-note">Antes del {_fecha(hist.index[0])} la serie se reconstruye con '
                    f'la liquidación diaria del archivo histórico de CBOE y la misma regla: '
                    f'{n_prev} sesiones, dentro el {cont.mean() * 100:.0f} % del tiempo. No forma '
                    f'parte del backtest del informe. Arrastra sobre la gráfica para ampliar; doble '
                    f'clic para volver.</p>', unsafe_allow_html=True)

    # ── Curva de capital ────────────────────────────────────────────
    c = vl.cartera(hist, peso)
    k = c.attrs.get("k", np.nan)
    m_c = vi.metricas(c["cartera"], c["spy"])
    m_s, m_l = vi.metricas(c["spy"]), vi.metricas(c["spy_apalancado"])
    st.markdown(T.section("Resultados", "La cartera frente al SPY a igual volatilidad",
                          f"La comparación honesta no es contra el SPY a secas: el sleeve añade riesgo. "
                          f"Es contra el SPY apalancado a la misma volatilidad ({ne(k, 2)}x). Ahí se ve "
                          f"la ventaja real: la volatilidad del sleeve es asimétrica y, a igual "
                          f"volatilidad, la cartera cae menos."), unsafe_allow_html=True)
    ventaja = m_c["dd"] - m_l["dd"]
    st.markdown(f"""<div class="mrow">
      <div class="mpill"><div class="ml">Cartera · CAGR</div><div class="mv am">{pe(m_c['cagr'])}</div></div>
      <div class="mpill"><div class="ml">SPY {ne(k, 2)}x · CAGR</div><div class="mv">{pe(m_l['cagr'])}</div></div>
      <div class="mpill"><div class="ml">Cartera · peor caída</div><div class="mv">{pe(m_c['dd'])}</div></div>
      <div class="mpill"><div class="ml">SPY {ne(k, 2)}x · peor caída</div><div class="mv">{pe(m_l['dd'])}</div></div>
      <div class="mpill"><div class="ml">Ventaja en caída</div><div class="mv {'up' if ventaja > 0 else 'dn'}">
        {T.arrow(ventaja)} {ne(ventaja, 1, signo=True)} pp</div></div>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(_chart_capital(c), width="stretch", config={"displayModeBar": False})

    # ── Deriva frente al backtest ───────────────────────────────────
    st.markdown(T.section("Deriva", "Serie viva frente al backtest congelado",
                          "La columna del informe es fija (corto estático con ejecución en apertura, "
                          "mayo 2013 – julio 2026). La serie viva se recalcula cada día desde el inicio "
                          "con el mismo motor. Si se separan, el modelo está derivando."),
                unsafe_allow_html=True)
    filas = [("CAGR", "cagr", "pct"), ("Volatilidad", "vol", "pct"), ("Sharpe", "sharpe", "num"),
             ("Peor caída", "dd", "pct"), ("Beta", "beta", "num"), ("Alfa anual", "alfa", "pct"),
             ("Information ratio", "ir", "num")]
    def f(v, t):
        return pe(v, 2) if t == "pct" else ne(v, 2)
    tabla = "".join(
        f'<tr><td>{n}</td><td class="am">{f(m_c.get(k_), t)}</td><td>{f(REF_INFORME.get(k_), t)}</td>'
        f'<td>{f(m_s.get(k_), t) if k_ in m_s else "—"}</td>'
        f'<td>{f(m_l.get(k_), t) if k_ in m_l else "—"}</td></tr>'
        for n, k_, t in filas)
    st.markdown(f"""<div class="stc-card" style="padding:0.4rem 0.9rem"><table class="stc-table">
        <tr><th>Métrica</th><th>Serie viva · {ne(peso * 100, 0)} %</th><th>Informe · 20 %</th>
        <th>SPY</th><th>SPY {ne(k, 2)}x</th></tr>{tabla}</table></div>""", unsafe_allow_html=True)
    st.caption("La serie viva cobra 5 pb de comisión + 2 pb de deslizamiento por lado y el préstamo del "
               "corto al 6 % anual; el informe, 5 pb (diferencia medida: −0,05 puntos de CAGR).")

    # ── Operaciones ─────────────────────────────────────────────────
    ops = vl.operaciones(hist)
    cerradas = [o for o in ops if not o["abierta"]]
    gan = sum(o["ret"] > 0 for o in cerradas)
    med = float(np.median([o["ret"] for o in cerradas])) if cerradas else np.nan
    st.markdown(T.section("Operaciones", "Cada entrada y cada salida",
                          f"{len(ops)} operaciones · {gan} de {len(cerradas)} cerradas ganadoras · mediana "
                          f"{pe(med * 100, 2)}. La mitad pierde: todo el dinero lo hacen unos pocos tramos "
                          f"largos, y por eso stops y objetivos de beneficio la destrozan. En rojo, las que "
                          f"perdieron; en ámbar, la abierta."), unsafe_allow_html=True)
    st.markdown(f'<div class="stc-card" style="padding:0.4rem 0.9rem">'
                f'{_tabla_operaciones(ops, peso, 12)}</div>', unsafe_allow_html=True)
    with st.expander(f"Ver las {len(ops)} operaciones"):
        st.markdown(_tabla_operaciones(ops, peso, None), unsafe_allow_html=True)

    # ── Episodios ───────────────────────────────────────────────────
    st.markdown(T.section("Riesgo", "Los sustos que lo deciden todo",
                          "Cinco semanas en trece años son la muestra real del modelo, no las sesiones. "
                          "Episodio = el VXX sube más de un 40 % en 5 sesiones."), unsafe_allow_html=True)
    pub = "".join(f"<tr><td>{a}</td><td>{b}</td><td class='dn'>▼ {c_}</td><td class='dn'>▼ {d}</td></tr>"
                  for a, b, c_, d in EPISODIOS_PUBLICADOS)
    nuevos = vl.episodios(hist, peso)
    nuevos = nuevos[nuevos["fecha"] > pd.Timestamp("2026-07-30")]
    extra = "".join(f"<tr class='open'><td>{_fecha(r.fecha)} · nuevo</td><td>—</td>"
                    f"<td>▼ {pe(r.coste_sleeve * 100, 1)}</td><td>▼ {pe(r.coste_cartera * 100, 1)}</td></tr>"
                    for r in nuevos.itertuples())
    st.markdown(f"""<div class="stc-card" style="padding:0.4rem 0.9rem"><table class="stc-table">
        <tr><th>Episodio</th><th>Subida del VXX</th><th>Coste al sleeve</th><th>Coste a la cartera (20 %)</th></tr>
        {pub}{extra}</table></div>""", unsafe_allow_html=True)
    st.caption("Los cinco primeros son los publicados en el informe. Un episodio nuevo aparece aquí en cuanto "
               "ocurre: el protocolo pide recalcular escenarios y Sharpe deflactado antes de volver a entrar.")

    # ── Alertas ─────────────────────────────────────────────────────
    st.markdown(T.section("Alertas", "Correo en cada cambio de régimen",
                          "Una GitHub Action evalúa cada cierre oficial y escribe en la salida (las dos "
                          "medidas en backwardation), en el aviso temprano (la primera se invierte) y en la "
                          "reentrada. Si a la mañana siguiente no hay dato, también avisa."),
                unsafe_allow_html=True)
    eventos = alert_state.get("eventos", [])[::-1][:12]
    if eventos:
        filas_ev = "".join(
            f'<tr class="{"lose" if ev["tipo"] == "SALIDA" else ""}"><td>{ev["tipo"]}</td>'
            f'<td>{_fecha(ev.get("fecha_dato"))}</td><td>{_fecha(ev.get("fecha_ejecucion"))}</td>'
            f'<td>{ne(ev.get("r1"), 4)}</td><td>{ne(ev.get("r2"), 4)}</td>'
            f'<td>{"▲ enviado" if ev.get("enviado") else "pendiente"}</td></tr>' for ev in eventos)
        st.markdown(f"""<div class="stc-card" style="padding:0.4rem 0.9rem"><table class="stc-table">
            <tr><th>Evento</th><th>Cierre</th><th>Ejecución</th><th>M2/M1</th><th>VIX3M/VIX</th><th>Correo</th></tr>
            {filas_ev}</table></div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p class="stc-note">Aún no se ha registrado ningún evento desde que se activó '
                    'el sistema de alertas.</p>', unsafe_allow_html=True)
    if alert_state.get("correo") != "configurado":
        with st.expander("Activar el correo (una sola vez)"):
            st.markdown("""
1. En la cuenta de Gmail que enviará los avisos, activa la verificación en dos pasos y crea una
   **contraseña de aplicación** en *myaccount.google.com/apppasswords*.
2. Guarda tres secretos en el repositorio de GitHub (*Settings → Secrets and variables → Actions*),
   o desde una terminal con `gh`:
""")
            st.code('gh secret set SMTP_USER --repo CompositeManTrader/VIX-Controller\n'
                    'gh secret set SMTP_PASSWORD --repo CompositeManTrader/VIX-Controller\n'
                    'gh secret set ALERT_TO --repo CompositeManTrader/VIX-Controller', language="bash")
            st.markdown("""
3. Lanza el workflow **VIX Inverse · curva, estrategia y alertas** a mano con *Enviar un correo de
   PRUEBA* marcado. Si llega, el sistema queda activo.

Los secretos nunca van en el código: el repositorio es público.
""")

    # ── Honestidad ──────────────────────────────────────────────────
    st.markdown(T.section("Límites", "Lo que este panel no demuestra"), unsafe_allow_html=True)
    st.markdown(f"""<div class="stc-card"><div class="stc-row"><span class="k">Validación</span><span class="v">
        No validado: Sharpe deflactado 77,0 % frente al 80 % exigido. La causa es el tamaño de muestra
        (cinco episodios en trece años), no el análisis.</span></div>
      <div class="stc-row"><span class="k">Probabilidad de batir al SPY</span><span class="v">86,8 %
        en 20.000 remuestreos por bloques. Un 13,6 % de caer peor del 40 %.</span></div>
      <div class="stc-row"><span class="k">Diversificación</span><span class="v">No es un diversificador:
        beta {ne(m_c.get('beta'), 2)} contra el SPY; protege en 2 de los 10 peores meses del índice.</span></div>
      <div class="stc-row"><span class="k">Tamaño</span><span class="v">15–20 %, deliberadamente por debajo
        del óptimo de la muestra (32,5 %, Kelly completo): el óptimo siempre está sobreajustado.</span></div>
      <div class="stc-row"><span class="k">Producto</span><span class="v">VXX es un ETN: riesgo de crédito
        del emisor. La pérdida de un corto no tiene tope teórico.</span></div></div>""",
                unsafe_allow_html=True)
