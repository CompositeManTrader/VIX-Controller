"""
alerts.py — Alertas del modelo VIX Inverse: detección de eventos y correo.

Eventos, evaluados con el CIERRE OFICIAL de cada sesión (la señal se ejecuta
en la apertura siguiente):

  SALIDA    la curva pasa de contango a backwardation en las DOS medidas
            (M2/M1 ≤ 1 y VIX3M/VIX ≤ 1) → recomprar VXX en la apertura.
  AVISO     estando dentro, la PRIMERA de las dos medidas cae a ≤ 1. Aún no
            es salida (la regla pide las dos), pero es el patrón del
            2/08/2024: M2/M1 en 1,006 con VIX3M/VIX ya en 0,998.
  ENTRADA   vuelve el contango → vender VXX en la apertura.
  DATO      la mañana siguiente aún no hay cierre oficial: no se puede
            evaluar la señal (nunca se interpreta un hueco como salida).

Funciones puras: no leen ficheros ni envían nada.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

SALIDA, AVISO, ENTRADA, DATO = "SALIDA", "AVISO", "ENTRADA", "DATO"


# ──────────────────────────────────────────────────────────────────────
# Calendario NYSE (festivos con cierre completo)
# ──────────────────────────────────────────────────────────────────────
def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - pd.Timedelta(days=1)
    if d.weekday() == 6:
        return d + pd.Timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d = d + pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return (pd.Timestamp(d) + pd.Timedelta(weeks=n - 1)).date()


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()
    return (pd.Timestamp(d) - pd.Timedelta(days=(d.weekday() - weekday) % 7)).date()


def nyse_holidays(year: int) -> set[date]:
    days = {
        _nth_weekday(year, 1, 0, 3),                       # Martin Luther King
        _nth_weekday(year, 2, 0, 3),                       # Presidents
        (pd.Timestamp(_easter(year)) - pd.Timedelta(days=2)).date(),   # Viernes Santo
        _last_weekday(year, 5, 0),                         # Memorial
        _nth_weekday(year, 9, 0, 1),                       # Labor
        _nth_weekday(year, 11, 3, 4),                      # Thanksgiving
    }
    for m, d in ((1, 1), (7, 4), (12, 25)):
        days.add(_observed(date(year, m, d)))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))             # Juneteenth
    # Año nuevo en sábado: NYSE no cierra el viernes 31 anterior
    days = {pd.Timestamp(d).date() for d in days}
    return {d for d in days if d.year == year}


def es_sesion(d: date) -> bool:
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def siguiente_sesion(d: date) -> date:
    x = pd.Timestamp(d).date()
    while True:
        x = (pd.Timestamp(x) + pd.Timedelta(days=1)).date()
        if es_sesion(x):
            return x


def sesion_anterior(d: date) -> date:
    x = pd.Timestamp(d).date()
    while True:
        x = (pd.Timestamp(x) - pd.Timedelta(days=1)).date()
        if es_sesion(x):
            return x


# ──────────────────────────────────────────────────────────────────────
# Eventos
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Evento:
    tipo: str
    fecha_dato: date               # cierre que lo dispara
    fecha_ejecucion: date | None   # apertura en la que se actúa
    r1: float | None               # M2/M1
    r2: float | None               # VIX3M/VIX
    r1_prev: float | None = None
    r2_prev: float | None = None
    detalle: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def clave(self) -> str:
        return f"{self.tipo}:{self.fecha_dato.isoformat()}"


def _invertidas(r1: float, r2: float) -> int:
    return int(not r1 > 1) + int(not r2 > 1)


def _contango(r1: float, r2: float) -> bool:
    if pd.isna(r1) or pd.isna(r2):
        return False
    return bool(r1 > 1 or r2 > 1)


def detectar_evento(curva: pd.DataFrame) -> Evento | None:
    """
    Compara el último cierre con el anterior. `curva` con índice fecha y
    columnas ratio_m2m1 (M2/M1) y ratio_vix3m (VIX3M/VIX).
    """
    cv = curva[["ratio_m2m1", "ratio_vix3m"]].dropna().sort_index()
    if len(cv) < 2:
        return None
    t = cv.index[-1]
    r1, r2 = float(cv.iloc[-1, 0]), float(cv.iloc[-1, 1])
    q1, q2 = float(cv.iloc[-2, 0]), float(cv.iloc[-2, 1])
    ahora, antes = _contango(r1, r2), _contango(q1, q2)
    ejec = siguiente_sesion(t.date())
    base = dict(fecha_dato=t.date(), fecha_ejecucion=ejec, r1=r1, r2=r2,
                r1_prev=q1, r2_prev=q2)
    if antes and not ahora:
        return Evento(SALIDA, **base,
                      detalle="Las dos medidas han cerrado en backwardation.")
    if not antes and ahora:
        return Evento(ENTRADA, **base, detalle="Vuelve el contango en al menos una medida.")
    if ahora and _invertidas(r1, r2) == 1 and _invertidas(q1, q2) == 0:
        cual = "M2/M1" if not r1 > 1 else "VIX3M/VIX"
        otra = r2 if cual == "M2/M1" else r1
        return Evento(AVISO, **base,
                      detalle=f"{cual} ha cerrado bajo 1,00; la otra sigue en {num(otra, 3)}.",
                      extra={"invertida": cual})
    return None


def evento_dato(ultima: date, esperada: date) -> Evento:
    return Evento(DATO, fecha_dato=esperada, fecha_ejecucion=siguiente_sesion(esperada),
                  r1=None, r2=None,
                  detalle=(f"No hay cierre oficial de la curva del {fmt(esperada)}. "
                           f"El último dato es del {fmt(ultima)}. La señal no se ha "
                           f"podido evaluar: revisa la fuente antes de la apertura."))


# ──────────────────────────────────────────────────────────────────────
# Formato y correo
# ──────────────────────────────────────────────────────────────────────
_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def num(v: float | None, dec: int = 4) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt(d: date | None) -> str:
    if d is None:
        return "—"
    d = pd.Timestamp(d).date()
    return f"{_DIAS[d.weekday()]} {d.strftime('%d/%m/%Y')}"


ASUNTO = {
    SALIDA: "VIX Inverse · SALIDA — recomprar VXX en la apertura del {ejec}",
    AVISO: "VIX Inverse · AVISO — {inv} bajo 1,00 (aún dentro)",
    ENTRADA: "VIX Inverse · ENTRADA — vender VXX en la apertura del {ejec}",
    DATO: "VIX Inverse · DATO PENDIENTE — sin cierre de la curva del {dato}",
}

ACCION = {
    SALIDA: "Recomprar toda la posición de VXX en la apertura del {ejec}. A efectivo.",
    AVISO: ("Sin acción: la regla solo sale con las dos medidas invertidas. "
            "Vigila el cierre de mañana: si la otra medida también cae bajo 1,00, "
            "la salida será en la apertura siguiente."),
    ENTRADA: "Vender VXX por el 100 % del capital del sleeve en la apertura del {ejec}.",
    DATO: "Comprobar el dato antes de la apertura del {ejec}. No operar a ciegas.",
}

COLOR = {SALIDA: "#EA3943", AVISO: "#F5A623", ENTRADA: "#16C784", DATO: "#9AA1A9"}


def asunto(ev: Evento) -> str:
    return ASUNTO[ev.tipo].format(ejec=fmt(ev.fecha_ejecucion), dato=fmt(ev.fecha_dato),
                                  inv=ev.extra.get("invertida", "una medida"))


def cuerpo_texto(ev: Evento, app_url: str = "") -> str:
    lineas = [asunto(ev), "",
              ev.detalle, "",
              f"Cierre evaluado:  {fmt(ev.fecha_dato)}",
              f"M2/M1 (futuros):  {num(ev.r1)}   antes {num(ev.r1_prev)}",
              f"VIX3M/VIX:        {num(ev.r2)}   antes {num(ev.r2_prev)}", "",
              "Acción: " + ACCION[ev.tipo].format(ejec=fmt(ev.fecha_ejecucion)), "",
              "Regla: dentro mientras M2/M1 > 1 o VIX3M/VIX > 1; fuera cuando las dos "
              "cierran ≤ 1. Ejecución en la apertura siguiente.", ""]
    if app_url:
        lineas += [f"Panel: {app_url}", ""]
    lineas += ["Spread Trading Club · VIX Controller",
               "Herramienta de seguimiento. No es asesoramiento de inversión."]
    return "\n".join(lineas)


def _fila(label: str, v: float | None, prev: float | None) -> str:
    color = "#F4F5F6"
    marca = ""
    if v is not None and not pd.isna(v):
        color = "#16C784" if v > 1 else "#EA3943"
        marca = "▲" if v > 1 else "▼"
    return (f'<tr><td style="padding:10px 0;color:#9AA1A9;font:500 12px \'JetBrains Mono\',monospace;'
            f'letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid #262A30">{label}</td>'
            f'<td style="padding:10px 0;text-align:right;color:{color};font:600 18px \'JetBrains Mono\','
            f'monospace;border-bottom:1px solid #262A30">{marca} {num(v)}</td>'
            f'<td style="padding:10px 0 10px 16px;text-align:right;color:#6B6B66;font:500 12px '
            f'\'JetBrains Mono\',monospace;border-bottom:1px solid #262A30">antes {num(prev)}</td></tr>')


def cuerpo_html(ev: Evento, app_url: str = "") -> str:
    c = COLOR[ev.tipo]
    accion = html.escape(ACCION[ev.tipo].format(ejec=fmt(ev.fecha_ejecucion)))
    boton = (f'<a href="{html.escape(app_url)}" style="display:inline-block;margin-top:20px;'
             f'background:#F5A623;color:#1A1303;text-decoration:none;font:600 14px Inter,sans-serif;'
             f'padding:12px 20px;border-radius:4px">Abrir el panel</a>') if app_url else ""
    tabla = ""
    if ev.r1 is not None:
        tabla = ('<table role="presentation" width="100%" cellspacing="0" cellpadding="0" '
                 'style="margin:20px 0 4px">'
                 + _fila("M2/M1 · futuros", ev.r1, ev.r1_prev)
                 + _fila("VIX3M/VIX · índices", ev.r2, ev.r2_prev) + "</table>")
    return f"""<!doctype html><html lang="es"><body style="margin:0;background:#0A0B0D">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#0A0B0D">
<tr><td align="center" style="padding:32px 16px">
<table role="presentation" width="560" cellspacing="0" cellpadding="0" style="max-width:560px;width:100%">
<tr><td style="padding-bottom:20px;border-bottom:1px solid #262A30">
  <span style="font:700 18px 'Space Grotesk',Inter,sans-serif;color:#F4F5F6">SPREAD</span>
  <span style="font:500 10px 'JetBrains Mono',monospace;letter-spacing:.4em;color:#9AA1A9;margin-left:6px">TRADING CLUB</span>
  <span style="float:right;font:500 11px 'JetBrains Mono',monospace;letter-spacing:.18em;color:#F5A623">VIX INVERSE</span>
</td></tr>
<tr><td style="padding:28px 0 0">
  <div style="font:500 11px 'JetBrains Mono',monospace;letter-spacing:.18em;color:{c};text-transform:uppercase">{ev.tipo} · cierre del {fmt(ev.fecha_dato)}</div>
  <div style="font:700 26px/1.2 'Space Grotesk',Inter,sans-serif;color:#F4F5F6;margin-top:10px">{html.escape(asunto(ev).split(' · ', 1)[1])}</div>
  <p style="font:400 15px/1.55 Inter,sans-serif;color:#C7CCD1;margin:14px 0 0">{html.escape(ev.detalle)}</p>
  {tabla}
  <div style="margin-top:18px;padding:14px 16px;background:#15171A;border:1px solid #262A30;border-left:3px solid {c};border-radius:4px">
    <div style="font:500 11px 'JetBrains Mono',monospace;letter-spacing:.18em;color:#9AA1A9;text-transform:uppercase">Acción</div>
    <div style="font:600 15px/1.5 Inter,sans-serif;color:#F4F5F6;margin-top:6px">{accion}</div>
  </div>
  {boton}
  <p style="font:400 12px/1.55 Inter,sans-serif;color:#6B6B66;margin:28px 0 0">
    Regla: dentro mientras M2/M1 &gt; 1 o VIX3M/VIX &gt; 1; fuera cuando las dos cierran ≤ 1.
    Ejecución en la apertura siguiente. Sin stop, sin objetivo, sin reajustar el tamaño.<br><br>
    Spread Trading Club · VIX Controller. Herramienta de seguimiento: no es asesoramiento de inversión.
  </p>
</td></tr></table></td></tr></table></body></html>"""
