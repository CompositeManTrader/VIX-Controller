"""
vix_inverse.py — Seguimiento del modelo VIX Inverse, CONGELADO el 2026-08-03.

Este módulo NO investiga ni mejora la señal: la reproduce exactamente y la
vigila. Cualquier cambio en la lógica de señal es un error, no una mejora.

────────────────────────────────────────────────────────────────────────────
LA ESPECIFICACIÓN CONGELADA
────────────────────────────────────────────────────────────────────────────
    Corto estático de VXX mientras haya contango en la curva del VIX.
    Fuera en cuanto las DOS medidas se inviertan.
    Ejecución en la APERTURA del día siguiente.
    Peso del sleeve: 15-20% de la cartera. El resto, SPY.

    señal(t)    = (ratio_m2m1 > 1) OR (ratio_vix3m > 1)   con datos de t-1
    posición(t) = señal(t), ejecutada en la APERTURA de t

────────────────────────────────────────────────────────────────────────────
AVISO DE SIGNO — LA COLUMNA ES M2/M1, NO M1/M2
────────────────────────────────────────────────────────────────────────────
El README del modelo dice "M1/M2 > 1". ESTÁ MAL ESCRITO. La columna real se
llama `ratio_m2m1` y vale **M2/M1**. Contango es M2 > M1, o sea
`ratio_m2m1 > 1`.

Fuente autoritativa — `analisis/02_datos/vix_indice_corto.py` línea 141:

    for etiqueta, m in (("contango  (M2 > M1)", v["ratio_m2m1"] > 1),

Verificado además aritméticamente contra los datos (2026-07-30):
    m1=18,7716 · m2=19,6787 · ratio_m2m1=1,048323 = 19,6787/18,7716 ✓
    VIX=17,09 · VIX3M=19,50 · ratio_vix3m=1,141018 = 19,50/17,09 ✓

Implementarlo como M1/M2 da la señal EXACTAMENTE INVERTIDA. `tests/
test_vix_inverse.py::TestConvencionDeSigno` falla si alguien lo invierte.

────────────────────────────────────────────────────────────────────────────
MÉTODO DE MEDICIÓN — POR QUÉ HAY DOS JUEGOS DE CIFRAS
────────────────────────────────────────────────────────────────────────────
El modelo tiene DOS mediciones distintas y sus números NO coinciden:

  · CANÓNICA (`analisis/`): corto ESTÁTICO con ejecución en apertura. La
    posición flota y se encoge sola según gana. Es lo que describe la
    especificación congelada, y es lo que implementa este módulo.
        CAGR +17,9% · vol ~19% · Sharpe 0,96 · caída −30,9%

  · MOTOR GENÉRICO (`modelo.py` + qplatform): reequilibra a peso fijo. El
    propio README lo llama «cota inferior» y dice que «no reproduce» la
    canónica.
        CAGR +18,5% · vol 22,8% · Sharpe 0,86 · caída −32,7%

Las cifras de referencia que circulan (18,50% / 22,75% / 0,862 / beta 1,220)
son las del MOTOR GENÉRICO. Comparar una serie viva calculada por el método
canónico contra esas referencias produce una «deriva» fantasma de ~3,7 puntos
de volatilidad que no es deriva: es diferencia de método. El panel muestra
las dos columnas etiquetadas para que no se confundan.

Funciones puras, sin Streamlit. Solo LECTURA de los parquet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# PARÁMETROS CONGELADOS — no tocar sin cambiar la fecha de congelación
# ──────────────────────────────────────────────────────────────────────
FECHA_CONGELACION = "2026-08-03"
PESO_SLEEVE = 0.20            # extremo alto de la banda 15-20%
BANDA_PESO = (0.15, 0.20)
PRESTAMO_ANUAL = 0.06         # coste del préstamo del corto
COMISION_BPS = 5.0
DESLIZAMIENTO_BPS = 2.0
COSTE_POR_LADO = (COMISION_BPS + DESLIZAMIENTO_BPS) / 10_000.0
INICIO = "2013-05-20"
CORTE_RESERVADO = pd.Timestamp("2020-01-01")

# Columnas mínimas que debe traer la curva
COLS_CURVA = ["ratio_m2m1", "ratio_vix3m", "VIX", "VIX3M", "m1", "m2",
              "dias_m1", "contango"]

# ──────────────────────────────────────────────────────────────────────
# CIFRAS DE REFERENCIA DEL BACKTEST — fijas, para detectar deriva
# ──────────────────────────────────────────────────────────────────────
# Motor genérico (peso fijo) — 2_RESULTADOS.md §3
REF_MOTOR_GENERICO = {
    "cagr": 18.50, "vol": 22.75, "sharpe": 0.862, "beta": 1.220,
    "alfa": 1.92, "ir": 0.500, "dd": -32.71,
    "cagr_spy": 13.91, "vol_spy": 16.9, "sharpe_spy": 0.85, "dd_spy": -33.70,
}
# Canónica (`analisis/`) — 2_RESULTADOS.md §7, tabla de conclusiones
REF_CANONICA = {
    "cagr": 17.9, "vol": 20.0, "sharpe": 0.96, "dd": -30.9,
}
# Comparación contra SPY apalancado a igual volatilidad (motor genérico)
REF_APALANCADO = {
    "k": 1.34,
    "sleeve_cagr": 19.33, "sleeve_dd": -32.6,
    "spy_lev_cagr": 18.81, "spy_lev_dd": -43.2,
    "ventaja_dd_pp": 10.6,
}
# Estado epistémico
ESTADO = {
    "validado": False,
    "dsr_modelo": 0.770, "dsr_umbral": 0.80,
    "prob_batir_spy": 0.868, "prob_umbral": 0.95,
    "episodios": 5, "anios": 13,
    "kelly_completo": 0.325, "kelly_medio": 0.162,
    "n_pruebas": 750,
}


# ──────────────────────────────────────────────────────────────────────
# ERRORES EXPLÍCITOS — nada de fallos silenciosos
# ──────────────────────────────────────────────────────────────────────
class DatosVixInverse(Exception):
    """Problema con los datos del modelo. Se muestra en pantalla, no se traga."""


@dataclass
class Diagnostico:
    """Lo que el panel debe contar al usuario sobre la salud del dato."""
    ultima_fecha: pd.Timestamp | None = None
    dias_habiles_retraso: int | None = None
    obsoleto: bool = False
    n_sesiones: int = 0
    avisos: list[str] = field(default_factory=list)
    errores: list[str] = field(default_factory=list)

    @property
    def hay_problemas(self) -> bool:
        return bool(self.errores) or self.obsoleto or bool(self.avisos)


# ──────────────────────────────────────────────────────────────────────
# CARGA (solo lectura)
# ──────────────────────────────────────────────────────────────────────
def _leer_equity(raiz: Path, simbolo: str) -> pd.DataFrame:
    ruta = raiz / "equities" / "daily" / f"symbol={simbolo}.parquet"
    if not ruta.exists():
        raise DatosVixInverse(f"No existe {ruta}")
    df = pd.read_parquet(ruta)
    if "date" not in df.columns:
        raise DatosVixInverse(f"{ruta.name} no trae columna 'date'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def cargar_datos(raiz_curated: str | Path,
                 hoy: pd.Timestamp | None = None) -> tuple[pd.DataFrame, Diagnostico]:
    """
    Lee curva del VIX + VXX + SPY. SOLO LECTURA.

    No rellena huecos por interpolación: si falta un día, se reporta.
    Devuelve (df, diagnostico). El diagnóstico nunca se traga: el panel lo pinta.
    """
    raiz = Path(raiz_curated)
    diag = Diagnostico()

    ruta_curva = raiz / "vix" / "curva.parquet"
    if not ruta_curva.exists():
        raise DatosVixInverse(f"No existe {ruta_curva}")
    cur = pd.read_parquet(ruta_curva)
    cur.index = pd.DatetimeIndex(cur.index).normalize()
    cur = cur.sort_index()

    faltan = [c for c in COLS_CURVA if c not in cur.columns]
    if faltan:
        raise DatosVixInverse(
            f"curva.parquet no trae las columnas {faltan}. "
            f"Tiene: {list(cur.columns)}")

    vxx = _leer_equity(raiz, "VXX")
    spy = _leer_equity(raiz, "SPY")["close"]

    d = vxx[["open", "close"]].join(cur[COLS_CURVA], how="inner")
    antes = len(d)
    d = d.dropna(subset=["open", "close", "ratio_m2m1", "VIX"]).sort_index()
    if antes - len(d):
        diag.avisos.append(
            f"{antes - len(d)} sesiones descartadas por faltar precio o "
            f"ratio_m2m1 (no se interpolan).")

    d["spy"] = spy
    if d["spy"].isna().any():
        n = int(d["spy"].isna().sum())
        diag.avisos.append(f"{n} sesiones sin cierre de SPY; quedan fuera de la cartera.")
    d["r_spy"] = d["spy"].pct_change()

    # ratio_vix3m puede faltar días sueltos: se informa, no se rellena.
    n_nan_r2 = int(d["ratio_vix3m"].isna().sum())
    if n_nan_r2:
        diag.avisos.append(
            f"{n_nan_r2} sesiones sin ratio_vix3m. Esos días la señal se apoya "
            f"solo en ratio_m2m1 (un OR con NaN no puede activarse por ese lado).")

    if d.empty:
        raise DatosVixInverse("El cruce de curva, VXX y SPY quedó vacío.")

    diag.n_sesiones = len(d)
    diag.ultima_fecha = d.index.max()

    ref = pd.Timestamp(hoy).normalize() if hoy is not None \
        else pd.Timestamp.today().normalize()
    retraso = int(np.busday_count(diag.ultima_fecha.date(), ref.date()))
    diag.dias_habiles_retraso = max(retraso, 0)
    diag.obsoleto = diag.dias_habiles_retraso > 1

    return d, diag


# ──────────────────────────────────────────────────────────────────────
# SEÑAL — el corazón congelado
# ──────────────────────────────────────────────────────────────────────
def señal_contango(df: pd.DataFrame) -> pd.Series:
    """
    Contango observado en el cierre de CADA fila (sin desplazar todavía).

    contango = (M2/M1 > 1) OR (VIX3M/VIX > 1)

    OJO: `ratio_m2m1` es M2/M1. Invertirlo da la señal del revés.
    """
    r1 = df["ratio_m2m1"] > 1
    r2 = df["ratio_vix3m"] > 1     # NaN > 1 → False, correcto para un OR
    return (r1 | r2).astype(bool)


def señal_posicion(df: pd.DataFrame) -> pd.Series:
    """
    La posición que se ejecuta en la APERTURA de cada día.

    Un ÚNICO desplazamiento: la decisión se toma con el cierre de t-1 y se
    ejecuta en la apertura de t. `simular_corto` aplica s[i] en la apertura
    del día i, así que aquí no debe haber un segundo shift.
    """
    return señal_contango(df).shift().fillna(False).astype(bool)


def verificar_sin_anticipacion(df: pd.DataFrame, pos: pd.Series) -> None:
    """
    Asegura que la posición de hoy NO usa información de hoy.

    Prueba: se altera la curva del último día de forma que el contango se dé
    la vuelta; la posición de ese mismo día debe quedar IDÉNTICA (sólo podría
    cambiar la del día siguiente, que aún no existe).

    Lanza AssertionError si detecta anticipación.
    """
    if len(df) < 10:
        return
    pos_orig = señal_posicion(df)
    trucado = df.copy()
    ult = trucado.index[-1]
    # Invertimos ambas medidas del último día
    trucado.loc[ult, "ratio_m2m1"] = 0.5 if trucado.loc[ult, "ratio_m2m1"] > 1 else 1.5
    trucado.loc[ult, "ratio_vix3m"] = 0.5 if trucado.loc[ult, "ratio_vix3m"] > 1 else 1.5
    pos_truc = señal_posicion(trucado)

    if bool(pos_orig.iloc[-1]) != bool(pos_truc.iloc[-1]):
        raise AssertionError(
            "ANTICIPACIÓN: cambiar la curva del último día alteró la posición "
            "de ese mismo día. La señal debe ejecutarse con el dato de ayer.")
    if not pos_orig.iloc[:-1].equals(pos_truc.iloc[:-1]):
        raise AssertionError(
            "ANTICIPACIÓN: alterar el último día cambió posiciones anteriores.")


# ──────────────────────────────────────────────────────────────────────
# SIMULACIÓN — corto estático, ejecución en apertura
# ──────────────────────────────────────────────────────────────────────
def simular_corto(px_o: pd.Series, px_c: pd.Series, pos: pd.Series,
                  prestamo: float = PRESTAMO_ANUAL,
                  coste: float = COSTE_POR_LADO) -> pd.Series:
    """
    Corto ESTÁTICO de VXX con ejecución en la apertura.

    Réplica exacta de `analisis/04_backtest/vix_corto_vxx_apertura.py::corto`
    con `en_apertura=True`. La posición flota: no se reequilibra, así que se
    encoge sola según gana. Esa flotación aporta ~0,10 de Sharpe frente al
    peso fijo del motor genérico.

    Mecánica de cada día i:
      1. tramo nocturno cierre[i-1] → apertura[i] con la posición que se traía
      2. en la apertura se ejecuta la decisión tomada ayer (pos[i])
      3. tramo apertura[i] → cierre[i] con la posición nueva, + préstamo

    Devuelve el rendimiento diario del capital del sleeve.

    NOTA sobre el coste, replicada tal cual del motor congelado: al abrir se
    hace `exp = cap` y DESPUÉS `cap -= cap·coste`, de modo que la exposición
    queda fijada sobre el capital previo al coste. El efecto es que un coste
    mayor eleva mínimamente el apalancamiento efectivo y, encadenando tramos
    ganadores, el resultado final no es monótono en el coste. Es una rareza
    del motor, no un error de transcripción: cambiarla alteraría las cifras
    publicadas del modelo congelado.
    """
    o = px_o.to_numpy(dtype=float)
    c = px_c.to_numpy(dtype=float)
    s = pos.to_numpy()
    cap, exp = 1.0, 0.0
    out = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if exp:                                  # 1. salto nocturno
            r = o[i] / c[i - 1] - 1
            cap -= exp * r
            exp *= (1 + r)
        if s[i] and not exp:                     # 2. ejecución en apertura
            exp = cap
            cap -= cap * coste
        elif not s[i] and exp:
            cap -= abs(exp) * coste
            exp = 0.0
        if exp:                                  # 3. sesión + préstamo
            r = c[i] / o[i] - 1
            cap -= exp * r
            exp *= (1 + r)
            cap -= exp * prestamo / 252
        out[i] = cap
    return pd.Series(out, index=px_c.index).pct_change()


def construir_cartera(df: pd.DataFrame, sleeve: pd.Series,
                      peso: float = PESO_SLEEVE) -> pd.DataFrame:
    """
    Cartera = (1-peso)·SPY + peso·sleeve, más el SPY apalancado a IGUAL
    volatilidad realizada.

    El apalancamiento k se calcula EN VIVO como σ(cartera)/σ(SPY). Fijarlo
    a la constante publicada (1,34x, que salió de la volatilidad del motor
    genérico) contra una serie canónica de ~19% de volatilidad compararía
    contra un SPY más volátil que la cartera, lo que favorecería
    artificialmente al modelo. La comparación honesta exige igualar la
    volatilidad de las series que realmente se están mostrando.
    """
    cart = ((1 - peso) * df["r_spy"] + peso * sleeve).dropna()
    spy = df["r_spy"].reindex(cart.index)
    sd_spy = float(spy.std())
    k = float(cart.std() / sd_spy) if sd_spy > 0 else np.nan
    return pd.DataFrame({
        "cartera": cart,
        "spy": spy,
        "spy_apalancado": spy * k,
    }).assign(**{"_k": k})


def curva_capital(r: pd.Series) -> pd.Series:
    return (1 + r.dropna()).cumprod()


def underwater(r: pd.Series) -> pd.Series:
    eq = curva_capital(r)
    return eq / eq.cummax() - 1


# ──────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────
def metricas(r: pd.Series, bench: pd.Series | None = None) -> dict:
    """Métricas de periodo sobre rendimientos diarios. Porcentajes en %."""
    r = r.dropna()
    if len(r) < 30:
        return {}
    eq = (1 + r).cumprod()
    anios = len(r) / 252
    dd = eq / eq.cummax() - 1
    sd = float(r.std())
    m = {
        "cagr": (eq.iloc[-1] ** (1 / anios) - 1) * 100 if eq.iloc[-1] > 0 else -100.0,
        "vol": sd * np.sqrt(252) * 100,
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "dd": float(dd.min()) * 100,
        "total": (eq.iloc[-1] - 1) * 100,
        "n": len(r),
    }
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        c = r.reindex(b.index)
        act = c - b
        sa = float(act.std())
        vb = float(b.var())
        m["alfa"] = float(act.mean()) * 252 * 100
        m["ir"] = float(act.mean() / sa * np.sqrt(252)) if sa > 0 else np.nan
        m["beta"] = float(c.cov(b) / vb) if vb > 0 else np.nan
    return m


# ──────────────────────────────────────────────────────────────────────
# OPERACIONES
# ──────────────────────────────────────────────────────────────────────
def extraer_operaciones(df: pd.DataFrame, pos: pd.Series,
                        sleeve: pd.Series) -> list[dict]:
    """
    Un tramo corto completo por operación, de apertura a apertura.

    El retorno es el NETO del sleeve (compuesto de los rendimientos diarios,
    ya con préstamo y comisión), no `entrada/salida - 1`. El bruto del precio
    exagera cada operación y más cuanto más dure.

    La última operación puede estar ABIERTA: se marca con `abierta=True` y su
    retorno es el acumulado hasta hoy.
    """
    s = pos.reindex(df.index).fillna(False).to_numpy().astype(int)
    op = df["open"].to_numpy(dtype=float)
    rs = sleeve.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    fechas = df.index

    ops: list[dict] = []
    i, n = 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1]:
                j += 1
            abierta = (j == n - 1)
            k = min(j + 1, n - 1)          # día de recompra
            entrada, salida = float(op[i]), float(op[k])
            neto = float(np.prod(1.0 + rs[i:k + 1]) - 1.0)
            ops.append({
                "f_entrada": fechas[i],
                "f_salida": None if abierta else fechas[k],
                "px_entrada": round(entrada, 2),
                "px_salida": None if abierta else round(salida, 2),
                "dias": int(k - i),
                "ret": neto,
                "abierta": abierta,
            })
            i = j + 1
        else:
            i += 1
    return ops


def racha_actual(pos: pd.Series) -> int:
    """Días consecutivos en la posición actual (dentro o fuera)."""
    v = pos.to_numpy()
    if len(v) == 0:
        return 0
    ultimo, n = v[-1], 1
    for x in v[-2::-1]:
        if bool(x) == bool(ultimo):
            n += 1
        else:
            break
    return n


# ──────────────────────────────────────────────────────────────────────
# ESTADO DE HOY
# ──────────────────────────────────────────────────────────────────────
def estado_actual(df: pd.DataFrame, pos: pd.Series) -> dict:
    """
    Qué hay que hacer hoy y con qué dato se sustenta.

    `fecha_dato` es el cierre que decide la posición (t-1), y `fecha_ejecucion`
    la sesión en cuya apertura se ejecuta.
    """
    if len(df) < 2:
        raise DatosVixInverse("Hacen falta al menos 2 sesiones.")

    fila_dato = df.iloc[-2]           # el cierre que manda sobre la última fila
    r1 = float(fila_dato["ratio_m2m1"])
    r2 = float(fila_dato["ratio_vix3m"]) if pd.notna(fila_dato["ratio_vix3m"]) else np.nan

    # Y además el dato MÁS RECIENTE, que decidirá la próxima sesión
    fila_hoy = df.iloc[-1]
    r1_hoy = float(fila_hoy["ratio_m2m1"])
    r2_hoy = float(fila_hoy["ratio_vix3m"]) if pd.notna(fila_hoy["ratio_vix3m"]) else np.nan
    prox = bool((r1_hoy > 1) or (pd.notna(r2_hoy) and r2_hoy > 1))

    return {
        "dentro": bool(pos.iloc[-1]),
        "fecha_dato": df.index[-2],
        "fecha_ejecucion": df.index[-1],
        "ratio_m2m1": r1,
        "ratio_vix3m": r2,
        "dist_m2m1": r1 - 1.0,
        "dist_vix3m": (r2 - 1.0) if pd.notna(r2) else np.nan,
        "racha": racha_actual(pos),
        # Lo que dice el cierre más reciente para la PRÓXIMA apertura
        "prox_dentro": prox,
        "prox_fecha_dato": df.index[-1],
        "prox_ratio_m2m1": r1_hoy,
        "prox_ratio_vix3m": r2_hoy,
        "cambio_pendiente": prox != bool(pos.iloc[-1]),
    }


# ──────────────────────────────────────────────────────────────────────
# ORQUESTADOR
# ──────────────────────────────────────────────────────────────────────
def ejecutar(raiz_curated: str | Path, peso: float = PESO_SLEEVE,
             hoy: pd.Timestamp | None = None,
             verificar: bool = True) -> dict:
    """
    Todo el seguimiento de una pasada. Devuelve un dict con:
      df, diagnostico, pos, sleeve, series (cartera/spy/spy_apalancado),
      k, estado, operaciones, metricas_vivas, metricas_spy
    """
    df, diag = cargar_datos(raiz_curated, hoy=hoy)
    pos = señal_posicion(df)

    if verificar:
        verificar_sin_anticipacion(df, pos)

    sleeve = simular_corto(df["open"], df["close"], pos,
                           prestamo=PRESTAMO_ANUAL, coste=COSTE_POR_LADO)
    series = construir_cartera(df, sleeve, peso=peso)
    k = float(series["_k"].iloc[0]) if len(series) else np.nan

    return {
        "df": df,
        "diagnostico": diag,
        "pos": pos,
        "sleeve": sleeve,
        "series": series[["cartera", "spy", "spy_apalancado"]],
        "k": k,
        "estado": estado_actual(df, pos),
        "operaciones": extraer_operaciones(df, pos, sleeve),
        "metricas_vivas": metricas(series["cartera"], series["spy"]),
        "metricas_spy": metricas(series["spy"]),
        "metricas_sleeve": metricas(sleeve.dropna()),
        "metricas_apalancado": metricas(series["spy_apalancado"]),
        "peso": peso,
    }


# ──────────────────────────────────────────────────────────────────────
# FORMATO ESPAÑOL — coma decimal, punto de millar
# ──────────────────────────────────────────────────────────────────────
def num_es(v: float | None, dec: int = 2, signo: bool = False) -> str:
    """1234.5 → '1.234,50'. None/NaN → '—'."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = f"{v:{'+' if signo else ''},.{dec}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def pct_es(v: float | None, dec: int = 2, signo: bool = True) -> str:
    """Porcentaje ya en unidades de % (18.5 → '+18,50 %')."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return num_es(v, dec, signo) + " %"
