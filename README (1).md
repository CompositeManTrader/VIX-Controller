# 📈 VIX Central — Term Structure Dashboard

Réplica de [VIXCentral.com](https://vixcentral.com) construida con **Streamlit + Plotly + yfinance**.

## Features

- **VIX Futures Term Structure** en tiempo real (hasta 12 meses)
- **VIX Spot** con cambio diario
- **Contango / Backwardation** entre cada par de meses
- **Previous Day overlay** para comparar vs ayer
- **Historical Prices** — selecciona cualquier fecha desde 2010
- **Multi-date comparison** — hasta 20 fechas en un solo gráfico
- **Data table** con DTE, precios, tickers, y contango por mes
- **Dark theme** profesional estilo VIXCentral

## Setup

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la app
streamlit run app.py
```

## Notas

- Los datos de futuros VIX provienen de **Yahoo Finance** (delay ~15 min)
- Los tickers siguen el formato `VX{MonthCode}{Year}.CBE`
- Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
- La expiración de futuros VIX es el miércoles 30 días antes del tercer viernes del mes siguiente

## Author

Alberto Alarcón González — Estrategia de Volatilidad Inversa
