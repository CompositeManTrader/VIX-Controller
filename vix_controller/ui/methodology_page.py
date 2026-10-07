"""methodology_page.py — Página "Metodología": reglas, protocolo, fuentes y límites."""
from __future__ import annotations

import streamlit as st

from vix_controller.ui import theme as T


def _tabla(cabecera: list[str], filas: list[list[str]]) -> str:
    th = "".join(f"<th>{T.esc(c)}</th>" for c in cabecera)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
    return (f'<div class="stc-card" style="padding:0.4rem 0.9rem"><table class="stc-table text">'
            f'<tr>{th}</tr>{tr}</table></div>')


def render() -> None:
    st.markdown(f"""{T.eyebrow("Metodología")}
        <div class="stc-h1">Cómo funciona y qué no promete</div>
        <p class="stc-lead">Todo lo que muestra la app sale de datos públicos de CBOE y de precios
        diarios de mercado. La estrategia está congelada: el panel la vigila, no la reoptimiza.</p>""",
                unsafe_allow_html=True)

    st.markdown(T.section("La regla", "VIX Inverse, sin ambigüedad",
                          "Dos series de la curva de futuros del VIX, una comparación cada una y nada "
                          "más. Los dos umbrales valen exactamente 1,00: la frontera entre contango y "
                          "backwardation."), unsafe_allow_html=True)
    st.markdown(_tabla(["Paso", "Cuándo", "Qué exactamente"], [
        ["1 · Medir", "Al cierre de cada sesión t",
         "a = liquidación M2 / liquidación M1 · b = índice VIX3M / índice VIX"],
        ["2 · Decidir", "Con el cierre de t, nunca con el de t+1",
         "dentro = (a &gt; 1,00) o (b &gt; 1,00). Si falta cualquiera de las dos: dentro = falso"],
        ["3 · Entrar", "Apertura de t+1, si ayer estaba fuera y hoy dentro",
         "Vender VXX por el 100 % del capital del sleeve. No se toca hasta salir"],
        ["4 · Salir", "Apertura de t+1, si ayer estaba dentro y hoy no",
         "Recomprar toda la posición. A efectivo"],
        ["5 · Mantener", "Todos los demás días",
         "No hacer nada. Sin stop, sin objetivo, sin reajustar el tamaño"],
    ]), unsafe_allow_html=True)
    st.markdown('<p class="stc-note" style="margin-top:0.8rem">La posición flota: si el VXX cae, el '
                'corto encoge solo respecto a la cuenta y llega pequeño al susto. Vale +0,10 de Sharpe '
                'frente a mantener el tamaño constante. Peso del sleeve: 15–20 % de la cartera; el '
                'resto, SPY.</p>', unsafe_allow_html=True)

    st.markdown(T.section("Protocolo", "Cuándo se apaga, decidido en frío",
                          "Umbrales fuera de lo visto en trece años para que no salten por ruido. "
                          "Una condición dispara revisión; dos, el apagado."), unsafe_allow_html=True)
    st.markdown(_tabla(["Condición", "Umbral", "Referencia histórica"], [
        ["La curva deja de pagar", "&lt; 55 % de días en contango en 12 meses", "85 % · peor año 64 %"],
        ["Los sustos se vuelven frecuentes", "≥ 3 episodios en 24 meses", "máximo visto: 2"],
        ["La señal pierde su ventaja", "Sharpe sobre siempre-corto &lt; −0,50 durante 6 meses",
         "mediana +0,23 · mínimo −0,74"],
        ["Daño acumulado", "caída del sleeve &gt; 45 %", "peor visto −26,2 %"],
        ["Riesgo de producto", "apagado inmediato",
         "VXX suspende emisiones, cambia de emisor, prima &gt;2 % sobre NAV, o el bróker restringe el corto"],
    ]), unsafe_allow_html=True)

    st.markdown(T.section("Alertas", "Qué dispara un correo"), unsafe_allow_html=True)
    st.markdown(_tabla(["Evento", "Cuándo", "Acción"], [
        ["SALIDA", "Las dos medidas cierran en 1,00 o menos", "Recomprar VXX en la apertura siguiente"],
        ["AVISO", "La primera medida cierra bajo 1,00 (la otra sigue en contango)",
         "Ninguna: vigilar el cierre siguiente"],
        ["ENTRADA", "Vuelve el contango en al menos una medida", "Vender VXX en la apertura siguiente"],
        ["DATO PENDIENTE", "Antes de la apertura aún no hay cierre oficial",
         "Comprobar la fuente; nunca interpretar un hueco como salida"],
    ]), unsafe_allow_html=True)
    st.markdown('<p class="stc-note" style="margin-top:0.8rem">Evaluación automática con GitHub '
                'Actions a las 22:15, 01:15 y 11:30 UTC de cada día hábil. La primera pasada que ve el '
                'cierre oficial envía el correo; las siguientes no lo repiten.</p>',
                unsafe_allow_html=True)

    st.markdown(T.section("Datos", "Fuentes y calidad"), unsafe_allow_html=True)
    st.markdown(_tabla(["Dato", "Fuente", "Uso"], [
        ["Curva de futuros en vivo", "CBOE · get_quotes_combined (JSON, ~15 min de retraso)",
         "Curva, contango, señal provisional intradía"],
        ["Liquidación oficial diaria", "CBOE · settlement por fecha + histórico por contrato",
         "Señal de la estrategia (M2/M1)"],
        ["Curva 2004-2013", "CBOE · archivo histórico de CFE, un fichero por contrato",
         "Historia larga de la curva y de las dos medidas"],
        ["VIX, VIX3M y familia", "CBOE · índices diarios y cotización retrasada",
         "Señal (VIX3M/VIX), página Volatilidad"],
        ["VXX, SPY, HYG, IEF", "yfinance (precios ajustados)",
         "Curva de capital, volatilidad realizada, crédito"],
        ["Opciones SPY", "Yahoo Finance", "Skew, superficie, GEX"],
    ]), unsafe_allow_html=True)
    st.markdown("""<p class="stc-note" style="margin-top:0.8rem">
        <b style="color:var(--white)">Vencimientos desplazados por festivos.</b> El vencimiento del VIX es el
        miércoles 30 días antes del tercer viernes del mes siguiente; si ese viernes es festivo (Viernes
        Santo, Juneteenth) CBOE lo adelanta un día. La curva de esta app lo corrige: cada fecha comprueba
        que su M1 es el vencimiento mensual más cercano. Sin esa corrección, el contrato front desaparece
        un mes entero y M2/M1 se calcula con M2 y M3.</p>
        <p class="stc-note"><b style="color:var(--white)">Historia de la estrategia.</b> Sembrada una vez
        desde la plataforma de backtesting con el motor congelado (reproduce el informe: Sharpe 0,96,
        caída −30,93 %, 76 operaciones) y continuada cada día con precios públicos. Continuar desde el
        estado guardado da exactamente el mismo resultado que simular de una vez.</p>
        <p class="stc-note"><b style="color:var(--white)">Dos mediciones del mismo modelo.</b> El informe
        mide con el corto estático (canónico): CAGR 17,9 %, Sharpe 0,96, caída −30,9 %. El motor genérico
        reequilibra a peso fijo y da una cota inferior: 18,5 %, 0,86, −32,7 %. No son intercambiables;
        el panel compara siempre contra la canónica.</p>
        <p class="stc-note"><b style="color:var(--white)">Curva antes de 2013.</b> El CDN actual de CBOE
        no trae liquidaciones antes de mayo de 2013; el tramo 2004-2013 sale del archivo histórico de
        CFE. Hasta el 26/03/2007 los futuros cotizaban a 10 veces el VIX y se dividen entre 10. En
        2004-2006 CBOE no listaba todos los meses: esas fechas quedan marcadas y las dos medidas
        reconstruidas empiezan después, en agosto de 2006.</p>
        <p class="stc-note"><b style="color:var(--white)">Página Volatilidad.</b> Realizada = raíz de la
        media de los rendimientos logarítmicos diarios del SPY al cuadrado, anualizada (×√252). Prima
        = VIX de cada día menos la realizada de las 21 sesiones siguientes: solo se conoce 21 sesiones
        después. El régimen se fija con el cierre del día y lo que se mide después usa solo días
        posteriores. El VXX se reconstruye con la liquidación de CBOE siguiendo la metodología del
        índice de futuros a un mes (reparto diario entre el primer y el segundo vencimiento); cuadra
        con el VXX real año a año (2019: −68,3 % frente a −67,8 %).</p>""", unsafe_allow_html=True)

    st.markdown(T.section("Aviso", "Herramienta de seguimiento"), unsafe_allow_html=True)
    st.markdown('<p class="stc-note">Contenido educativo y de seguimiento de una estrategia propia. No es '
                'asesoramiento de inversión ni una recomendación. Los resultados pasados, y más aún los '
                'simulados, no garantizan resultados futuros. Vender volatilidad tiene pérdidas '
                'potencialmente ilimitadas.</p>', unsafe_allow_html=True)
