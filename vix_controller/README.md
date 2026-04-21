# vix_controller — paquete modular

Extracción de `app.py` hacia módulos testeables.

## Layout

```
vix_controller/
├── config.py           # constantes, TTLs, pesos VTS (single source of truth)
├── rates.py            # get_risk_free_rate(), get_dividend_yield() dinámicos
└── quant/
    ├── bs.py           # Black-Scholes pricing + IV (Brent) con clamp
    ├── svi.py          # SVI Gatheral 2004 (fix T_proxy)
    ├── percentile.py   # Rolling percentile (VTS)
    ├── signals.py      # BB × Contango (sin look-ahead)
    └── backtest.py     # Sharpe, Sortino, DD, CAGR, apply_signal_returns
```

## Principios

1. **Zero-Streamlit**: ninguna función aquí usa `st.cache_data`, `st.session_state`
   ni levanta browsers. Esto las hace testeables offline.
2. **No magic numbers**: todo valor ajustable vive en `config.py`.
3. **Fallback seguro**: `rates.py` siempre devuelve un decimal válido
   aunque la red falle (lee `config.FALLBACK_*`).
4. **Sin look-ahead**: las señales shift(1) antes de multiplicar por retorno;
   verificado por `tests/test_signals.py::test_no_lookahead`.

## Cambios vs. versión anterior de `app.py`

| Fix | Qué cambió | Archivo |
|---|---|---|
| **B1** | `except: pass/continue` → excepciones tipadas con log | `app.py` |
| **B2** | `r=0.043, q=0.013` hardcoded → `get_risk_free_rate()` / `get_dividend_yield()` | `rates.py` + `app.py` |
| **B3** | SVI `T_proxy=1.0` → `T = dte/365` real por slice | `quant/svi.py` |
| **B7** | `min_obs = max(30, W//5)` → `max(60, W//3)` | `quant/percentile.py` |
| **B10** | IV Brent rango `[1e-6, 5.0]` → `[0.01, 3.0]` + verificación de cambio de signo | `quant/bs.py` |
| **E8** | Magic numbers dispersos → `config.py` | todos |
```
