# Tests — VIX Controller

Golden tests para los engines quant críticos. Se ejecutan sin Streamlit.

## Setup

```bash
# 1. Instalar Python 3.11+ si aún no lo tienes
#    (Microsoft Store / python.org / conda)

# 2. Dependencias del proyecto
pip install -r requirements.txt

# 3. pytest
pip install pytest
```

## Ejecutar

Desde la raíz del proyecto:

```bash
pytest tests/ -v
```

Los tests que requieren red (`test_rates.py` → yfinance) harán fallback
al valor de `config.FALLBACK_*` si no hay conexión, así que pasan offline.

## Cobertura

| Test | Qué verifica |
|---|---|
| `test_bs.py` | Pricing BS (golden values Hull), put-call parity, IV roundtrip |
| `test_svi.py` | Fit SVI sobre smile sintético, constraint Durrleman butterfly |
| `test_percentile.py` | Rolling percentile (constante, monótono, NaN, min_obs) |
| `test_signals.py` | BB×Contango: output binario, no look-ahead verificado |
| `test_backtest.py` | Sharpe/Sortino/DD/CAGR + no look-ahead en `apply_signal_returns` |
| `test_rates.py` | Fallback cuando yfinance falla o ticker inválido |

## Golden values (referencia Hull 10th ed.)

```
bs_call(S=100, K=100, r=0.05, T=1, σ=0.20, q=0) ≈ 10.4506
put-call parity: C - P = S·exp(-qT) - K·exp(-rT)
bs_gamma max at ATM (para T y σ dados)
```
