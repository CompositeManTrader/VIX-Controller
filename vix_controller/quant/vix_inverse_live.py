"""
vix_inverse_live.py — Seguimiento EN VIVO del modelo VIX Inverse.

Por qué existe. El panel anterior leía los precios de la Plataforma en el
disco local: en Streamlit Cloud no cargaba. Además el repo es público, así
que no se pueden publicar los precios de Alpha Vantage.

Solución: una HISTORIA DERIVADA (`data/vix_inverse_history.parquet`).
  · Se siembra UNA vez desde la Plataforma (solo lectura) con el motor
    congelado → reproduce exactamente las cifras publicadas.
  · Guarda rendimientos y el ESTADO del motor (capital y exposición al
    cierre), no precios brutos. Solo conserva el precio de apertura de los
    ~150 días de ejecución (entradas/salidas) para la tabla de operaciones.
  · Se CONTINÚA cada día con precios públicos (yfinance, mismo ajuste que la
    Plataforma: ratio 1,000000 en el VXX) y la curva de CBOE. Continuar desde
    el estado guardado da exactamente lo mismo que simular de una vez.

Columnas de la historia:
  ratio_m2m1, ratio_vix3m   las dos medidas al cierre de la fila
  contango                  señal al cierre (decide la apertura SIGUIENTE)
  pos                       posición ejecutada en la apertura de la fila
  sleeve_ret, always_ret    rendimiento del sleeve (señal / siempre corto)
  spy_ret                   rendimiento del SPY
  cap, exp, cap_a, exp_a    estado del motor al cierre (señal / siempre corto)
  px_exec                   apertura del VXX en días de entrada o salida
  episodio                  VXX +40 % en 5 sesiones (inicio de episodio)

Funciones puras, sin Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import vix_inverse as vi

HIST_COLUMNS = ["ratio_m2m1", "ratio_vix3m", "contango", "pos", "sleeve_ret",
                "always_ret", "spy_ret", "cap", "exp", "cap_a", "exp_a",
                "px_exec", "episodio"]

EPISODIO_SUBIDA = 0.40         # VXX +40 % en 5 sesiones (protocolo §2.3)
EPISODIO_SESIONES = 5
EPISODIO_SEPARACION = 20       # sesiones entre dos episodios distintos
VENTANA_VENTAJA = 756          # 3 años, ventaja de Sharpe sobre siempre-corto

# Condiciones de parada (3_PROTOCOLO_OPERACION.md §2)
UMBRAL_CONTANGO_12M = 0.55
UMBRAL_EPISODIOS_24M = 3
UMBRAL_VENTAJA = -0.50
UMBRAL_VENTAJA_SESIONES = 126  # 6 meses seguidos
UMBRAL_CAIDA_SLEEVE = -0.45


class HistoriaError(RuntimeError):
    """La historia derivada no es coherente con los datos nuevos."""


# ──────────────────────────────────────────────────────────────────────
# Construcción y continuación
# ──────────────────────────────────────────────────────────────────────
def _episodios(close: pd.Series) -> pd.Series:
    """Marca el PRIMER día de cada episodio (VXX +40 % en 5 sesiones)."""
    sube = (close / close.shift(EPISODIO_SESIONES) - 1) > EPISODIO_SUBIDA
    out = pd.Series(False, index=close.index)
    ultimo = -10 ** 9
    for i, flag in enumerate(sube.to_numpy()):
        if flag:
            if i - ultimo > EPISODIO_SEPARACION:
                out.iloc[i] = True
            ultimo = i
    return out


def _px_exec(open_: pd.Series, pos: pd.Series, pos_prev: bool) -> pd.Series:
    cambio = pos.astype(bool) != pos.astype(bool).shift(fill_value=pos_prev)
    return open_.where(cambio)


def construir_historia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historia completa desde cero. `df` necesita: open, close (VXX),
    ratio_m2m1, ratio_vix3m, r_spy. Es el motor congelado tal cual.
    """
    contango = vi.señal_contango(df)
    pos = contango.shift().fillna(False).astype(bool)
    o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
    caps, exps = vi.motor_corto(o, c, pos.to_numpy())
    caps_a, exps_a = vi.motor_corto(o, c, np.ones(len(df), dtype=bool))

    sleeve = pd.Series(caps, index=df.index)
    always = pd.Series(caps_a, index=df.index)
    sleeve.iloc[0] = np.nan                    # igual que el motor original
    always.iloc[0] = np.nan
    h = pd.DataFrame({
        "ratio_m2m1": df["ratio_m2m1"].astype(float),
        "ratio_vix3m": df["ratio_vix3m"].astype(float),
        "contango": contango,
        "pos": pos,
        "sleeve_ret": sleeve.pct_change(),
        "always_ret": always.pct_change(),
        "spy_ret": df["r_spy"].astype(float),
        "cap": caps, "exp": exps, "cap_a": caps_a, "exp_a": exps_a,
        "px_exec": _px_exec(df["open"], pos, False),
        "episodio": _episodios(df["close"]),
    }, index=df.index)
    h.index.name = "fecha"
    return h[HIST_COLUMNS]


def continuar_historia(hist: pd.DataFrame, nuevos: pd.DataFrame,
                       vxx_close_lookback: pd.Series | None = None) -> pd.DataFrame:
    """
    Añade sesiones posteriores a la última fila de `hist`.

    `nuevos`: índice con la ÚLTIMA fecha de `hist` como primera fila (ancla
    del salto nocturno) y luego las sesiones nuevas. Columnas: open, close
    (VXX), spy_close, ratio_m2m1, ratio_vix3m. Las sesiones nuevas deben
    venir completas: una sesión sin precio o sin alguna medida no se añade
    (se espera al día siguiente en lugar de inventarla).

    `vxx_close_lookback`: cierres del VXX de al menos 5 sesiones antes del
    ancla, para detectar episodios que empiecen en las sesiones nuevas.
    """
    if hist.empty:
        raise HistoriaError("Historia vacía: hay que sembrarla primero")
    ancla = hist.index[-1]
    if nuevos.empty or nuevos.index[0] != ancla:
        raise HistoriaError(f"La primera fila de los datos nuevos debe ser el ancla {ancla.date()}")

    nuevos = nuevos.sort_index()
    cuerpo = nuevos.iloc[1:]
    completos = cuerpo[["open", "close", "spy_close", "ratio_m2m1", "ratio_vix3m"]].notna().all(axis=1)
    if not completos.all():
        primero_malo = completos[~completos].index[0]
        cuerpo = cuerpo.loc[cuerpo.index < primero_malo]
    if cuerpo.empty:
        return hist

    tramo = pd.concat([nuevos.loc[[ancla]], cuerpo])
    contango = vi.señal_contango(tramo)
    contango.iloc[0] = bool(hist["contango"].iloc[-1])          # la señal del ancla manda
    pos = contango.shift().fillna(False).astype(bool)
    pos.iloc[0] = bool(hist["pos"].iloc[-1])

    o, c = tramo["open"].to_numpy(float), tramo["close"].to_numpy(float)
    ult = hist.iloc[-1]
    caps, exps = vi.motor_corto(o, c, pos.to_numpy(), cap0=ult["cap"], exp0=ult["exp"])
    caps_a, exps_a = vi.motor_corto(o, c, np.ones(len(tramo), dtype=bool),
                                    cap0=ult["cap_a"], exp0=ult["exp_a"])

    if vxx_close_lookback is not None and len(vxx_close_lookback):
        serie = pd.concat([vxx_close_lookback[vxx_close_lookback.index < ancla],
                           tramo["close"]]).sort_index()
        serie = serie[~serie.index.duplicated(keep="last")]
        epis = _episodios(serie).reindex(tramo.index).fillna(False)
    else:
        epis = pd.Series(False, index=tramo.index)

    nuevo = pd.DataFrame({
        "ratio_m2m1": tramo["ratio_m2m1"].astype(float),
        "ratio_vix3m": tramo["ratio_vix3m"].astype(float),
        "contango": contango,
        "pos": pos,
        "sleeve_ret": pd.Series(caps, index=tramo.index).pct_change(),
        "always_ret": pd.Series(caps_a, index=tramo.index).pct_change(),
        "spy_ret": tramo["spy_close"].astype(float).pct_change(),
        "cap": caps, "exp": exps, "cap_a": caps_a, "exp_a": exps_a,
        "px_exec": _px_exec(tramo["open"], pos, bool(hist["pos"].iloc[-1])),
        "episodio": epis.astype(bool),
    }, index=tramo.index).iloc[1:]
    nuevo.index.name = "fecha"
    return pd.concat([hist, nuevo[HIST_COLUMNS]])


# ──────────────────────────────────────────────────────────────────────
# Lecturas para el panel
# ──────────────────────────────────────────────────────────────────────
def medidas_extendidas(hist: pd.DataFrame, curva: pd.DataFrame | None) -> pd.DataFrame:
    """
    Las dos medidas y la posición desde que existen las dos (VIX3M, 07/2006).

    Dentro del backtest se usa la historia tal cual. Antes de su inicio se
    reconstruyen desde la curva (archivo CFE de CBOE) con la MISMA regla
    (`vi.señal_posicion`). Ese tramo empieza después del último día con
    hueco M1→M2 anómalo: en 2004-2006 no se listaban todos los meses y el
    «M2» podía ser el de tres meses. Columna `backtest` = fila del informe.
    """
    out = hist[["ratio_m2m1", "ratio_vix3m", "pos"]].copy()
    out["pos"] = out["pos"].astype(bool)
    out["backtest"] = True
    if curva is None or curva.empty or hist.empty:
        return out
    c = curva[curva.index < hist.index[0]]
    if "gap_ok" in c.columns and (~c["gap_ok"].astype(bool)).any():
        c = c[c.index > c.index[~c["gap_ok"].astype(bool)].max()]
    c = c[["ratio_m2m1", "ratio_vix3m"]]
    completas = c.dropna()
    if completas.empty:
        return out
    c = c[c.index >= completas.index[0]]
    previo = c.copy()
    previo["pos"] = vi.señal_posicion(c)
    previo["backtest"] = False
    return pd.concat([previo, out])


def cartera(hist: pd.DataFrame, peso: float = vi.PESO_SLEEVE) -> pd.DataFrame:
    """Cartera, SPY y SPY apalancado a IGUAL volatilidad realizada."""
    cart = ((1 - peso) * hist["spy_ret"] + peso * hist["sleeve_ret"]).dropna()
    spy = hist["spy_ret"].reindex(cart.index)
    k = float(cart.std() / spy.std()) if spy.std() > 0 else np.nan
    out = pd.DataFrame({"cartera": cart, "spy": spy, "spy_apalancado": spy * k})
    out.attrs["k"] = k
    return out


def operaciones(hist: pd.DataFrame) -> list[dict]:
    """Un tramo corto completo por operación (mismo cálculo que el informe)."""
    s = hist["pos"].astype(bool).to_numpy()
    rs = hist["sleeve_ret"].fillna(0.0).to_numpy()
    px = hist["px_exec"].to_numpy()
    f = hist.index
    ops, i, n = [], 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1]:
                j += 1
            abierta = j == n - 1
            k = min(j + 1, n - 1)
            ops.append({
                "f_entrada": f[i], "f_salida": None if abierta else f[k],
                "px_entrada": float(px[i]) if np.isfinite(px[i]) else None,
                "px_salida": None if abierta or not np.isfinite(px[k]) else float(px[k]),
                "dias": int(k - i),
                "ret": float(np.prod(1.0 + rs[i:k + 1]) - 1.0),
                "abierta": abierta,
            })
            i = j + 1
        else:
            i += 1
    return ops


def episodios(hist: pd.DataFrame, peso: float = vi.PESO_SLEEVE) -> pd.DataFrame:
    """
    Los sustos: fecha de inicio y peor caída del sleeve y de la cartera en
    la ventana [−10, +20] sesiones. Esa ventana reproduce 4 de los 5 costes
    publicados en el informe (el de 08/2024 usa otra definición allí).
    """
    filas = []
    idx = hist.index
    cart = (1 - peso) * hist["spy_ret"] + peso * hist["sleeve_ret"]
    for d in idx[hist["episodio"].astype(bool)]:
        p = idx.get_loc(d)
        sl = slice(max(p - 10, 0), p + 20)
        eq = (1 + hist["sleeve_ret"].iloc[sl].fillna(0)).cumprod()
        ec = (1 + cart.iloc[sl].fillna(0)).cumprod()
        filas.append({"fecha": d,
                      "coste_sleeve": float((eq / eq.cummax() - 1).min()),
                      "coste_cartera": float((ec / ec.cummax() - 1).min())})
    return pd.DataFrame(filas, columns=["fecha", "coste_sleeve", "coste_cartera"])


def _sharpe_movil(r: pd.Series, w: int) -> pd.Series:
    mp = int(w * 0.9)
    return r.rolling(w, min_periods=mp).mean() / r.rolling(w, min_periods=mp).std() * np.sqrt(252)


def ventaja_sharpe(hist: pd.DataFrame, w: int = VENTANA_VENTAJA) -> pd.Series:
    """Sharpe móvil del sleeve menos el del siempre-corto (protocolo §2.4)."""
    return (_sharpe_movil(hist["sleeve_ret"], w) - _sharpe_movil(hist["always_ret"], w)).dropna()


@dataclass
class Condicion:
    nombre: str
    valor: str
    umbral: str
    referencia: str
    disparada: bool
    cerca: bool


def condiciones_parada(hist: pd.DataFrame) -> list[Condicion]:
    """Las cuatro condiciones estructurales medibles del protocolo, hoy."""
    out: list[Condicion] = []

    c12 = hist["contango"].astype(bool).tail(252).mean()
    out.append(Condicion(
        "La curva deja de pagar", f"{vi.num_es(c12 * 100, 0)} % de días en contango (12 meses)",
        f"< {vi.num_es(UMBRAL_CONTANGO_12M * 100, 0)} %", "85 % histórico · peor año 64 %",
        bool(c12 < UMBRAL_CONTANGO_12M), bool(c12 < UMBRAL_CONTANGO_12M + 0.10)))

    n_ep = int(hist["episodio"].astype(bool).tail(504).sum())
    out.append(Condicion(
        "Los sustos se vuelven frecuentes",
        f"{n_ep} episodio{'' if n_ep == 1 else 's'} en 24 meses",
        f"≥ {UMBRAL_EPISODIOS_24M}", "máximo visto: 2",
        n_ep >= UMBRAL_EPISODIOS_24M, n_ep >= UMBRAL_EPISODIOS_24M - 1))

    adv = ventaja_sharpe(hist)
    hoy = float(adv.iloc[-1]) if len(adv) else np.nan
    bajo = (adv < UMBRAL_VENTAJA).astype(int)
    racha = 0
    for v in bajo.to_numpy()[::-1]:
        if v:
            racha += 1
        else:
            break
    out.append(Condicion(
        "La señal pierde su ventaja",
        f"{vi.num_es(hoy, 2, signo=True)} Sharpe sobre siempre-corto (3 años) · {racha} sesiones bajo el umbral",
        f"< {vi.num_es(UMBRAL_VENTAJA, 2, signo=True)} durante 6 meses", "mediana +0,23 · mínimo −0,74",
        racha >= UMBRAL_VENTAJA_SESIONES, bool(np.isfinite(hoy) and hoy < UMBRAL_VENTAJA)))

    eq = (1 + hist["sleeve_ret"].fillna(0)).cumprod()
    dd = float((eq / eq.cummax() - 1).iloc[-1])
    out.append(Condicion(
        "Daño acumulado", f"{vi.num_es(dd * 100, 1)} % caída del sleeve desde máximos",
        f"> {vi.num_es(abs(UMBRAL_CAIDA_SLEEVE) * 100, 0)} %", "peor visto −26,2 %",
        dd < UMBRAL_CAIDA_SLEEVE, dd < UMBRAL_CAIDA_SLEEVE / 2))
    return out


def veredicto_protocolo(conds: list[Condicion]) -> tuple[str, str]:
    """(estado, explicación) según la regla: una → revisión, dos → apagado."""
    n = sum(c.disparada for c in conds)
    if n >= 2:
        return "APAGADO", "Dos condiciones disparadas: el protocolo manda apagar el modelo."
    if n == 1:
        return "REVISIÓN", "Una condición disparada: revisión inmediata antes de seguir operando."
    return "OPERABLE", "Ninguna condición estructural disparada."


def estado(hist: pd.DataFrame, curva: pd.DataFrame | None = None) -> dict:
    """
    Posición actual y lo que dice el último cierre conocido.

    `hist` manda sobre la posición (precios incluidos). `curva` (si trae
    fechas posteriores) manda sobre la PRÓXIMA apertura: puede ir un día por
    delante porque la curva oficial se publica antes que los precios.
    """
    ratios = hist[["ratio_m2m1", "ratio_vix3m"]]
    if curva is not None and not curva.empty:
        cv = curva[["ratio_m2m1", "ratio_vix3m"]].dropna()
        extra = cv[cv.index > hist.index[-1]]
        ratios = pd.concat([ratios, extra])
    contango = vi.señal_contango(ratios)
    # La historia manda en sus fechas (es la señal con la que se operó)
    contango.loc[hist.index] = hist["contango"].astype(bool)
    pos = contango.shift().fillna(False).astype(bool)
    pos.loc[hist.index] = hist["pos"].astype(bool)

    f_dato = ratios.index[-1]
    r1 = float(ratios["ratio_m2m1"].iloc[-1])
    r2 = float(ratios["ratio_vix3m"].iloc[-1])
    dentro = bool(pos.iloc[-1])          # posición vigente durante la sesión f_dato
    prox = bool(contango.iloc[-1])       # lo que se ejecuta en la próxima apertura

    v = pos.to_numpy()
    racha = 1
    for x in v[-2::-1]:
        if x == v[-1]:
            racha += 1
        else:
            break
    return {"fecha_hist": hist.index[-1], "fecha_dato": f_dato,
            "dentro": dentro, "prox_dentro": prox,
            "cambio_pendiente": prox != dentro,
            "ratio_m2m1": r1, "ratio_vix3m": r2,
            "dist_m2m1": r1 - 1.0, "dist_vix3m": r2 - 1.0,
            "una_invertida": (r1 > 1) != (r2 > 1),
            "racha": racha}
