"""
regime.py — Detección de régimen de volatilidad con HMM gaussiano.

Modelo: Hidden Markov Model univariado sobre retornos diarios del SPY,
K=3 estados gaussianos que se diferencian por su varianza:

    calm        (σ baja)   — cosecha de prima, short vol funciona
    transition  (σ media)  — drift débil, whipsaws, reducir tamaño
    panic       (σ alta)   — clusters de -3%/+4%, short vol muere aquí

Implementación Baum-Welch (EM) en numpy puro con scaling de Rabiner —
sin dependencias nuevas (no hmmlearn/sklearn). Para T~5000 y K=3 el fit
toma <1s.

Uso honesto en producción: las probabilidades FILTRADAS (forward only,
sin look-ahead) son las que valen para decidir HOY. Las suavizadas
(forward-backward) solo para visualizar historia.

Referencias: Rabiner (1989); Hamilton (1989) regime switching;
Ang & Bekaert (2002) regímenes en equity vol.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_SQRT_2PI = np.sqrt(2.0 * np.pi)
REGIME_LABELS = ("calm", "transition", "panic")


def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * _SQRT_2PI)


class GaussianHMM1D:
    """HMM gaussiano univariado con K estados. fit() via Baum-Welch."""

    def __init__(self, n_states: int = 3, n_iter: int = 60,
                 tol: float = 1e-6, seed: int = 7):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.mu = None        # (K,)
        self.sigma = None     # (K,)
        self.pi = None        # (K,) initial probs
        self.A = None         # (K,K) transition matrix
        self.loglik_ = None

    # ── init: cuantiles de |x| para separar por nivel de vol ─────
    def _init_params(self, x: np.ndarray) -> None:
        K = self.K
        absx = np.abs(x - x.mean())
        qs = np.quantile(absx, np.linspace(0, 1, K + 1))
        mu, sg = np.empty(K), np.empty(K)
        for k in range(K):
            mask = (absx >= qs[k]) & (absx <= qs[k + 1])
            chunk = x[mask] if mask.sum() > 10 else x
            mu[k] = chunk.mean()
            sg[k] = max(chunk.std(), 1e-5)
        order = np.argsort(sg)
        self.mu, self.sigma = mu[order], sg[order]
        self.pi = np.full(K, 1.0 / K)
        # Transiciones persistentes (diagonal dominante) como prior razonable
        self.A = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(self.A, 0.95)

    def _emissions(self, x: np.ndarray) -> np.ndarray:
        B = np.column_stack([_gauss_pdf(x, self.mu[k], self.sigma[k])
                             for k in range(self.K)])
        return np.maximum(B, 1e-300)

    def fit(self, x: np.ndarray) -> "GaussianHMM1D":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        T, K = len(x), self.K
        if T < 50 * K:
            raise ValueError(f"Serie muy corta para HMM: {T} obs, K={K}")
        self._init_params(x)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = self._emissions(x)

            # ── Forward con scaling (Rabiner) ────────────────────
            alpha = np.empty((T, K)); c = np.empty(T)
            alpha[0] = self.pi * B[0]
            c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.A) * B[t]
                c[t] = alpha[t].sum()
                if c[t] <= 0:
                    c[t] = 1e-300
                alpha[t] /= c[t]
            ll = float(np.log(c).sum())

            # ── Backward (mismo scaling) ─────────────────────────
            beta = np.empty((T, K))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            # xi sumado sobre t (para re-estimar A)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                num = (alpha[t][:, None] * self.A
                       * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
                xi_sum += num

            # ── M-step ───────────────────────────────────────────
            self.pi = gamma[0]
            self.A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
            gsum = np.maximum(gamma.sum(axis=0), 1e-300)
            self.mu = (gamma * x[:, None]).sum(axis=0) / gsum
            var = (gamma * (x[:, None] - self.mu[None, :]) ** 2).sum(axis=0) / gsum
            self.sigma = np.sqrt(np.maximum(var, 1e-10))

            if abs(ll - prev_ll) < self.tol * abs(prev_ll):
                prev_ll = ll
                break
            prev_ll = ll

        self.loglik_ = prev_ll
        self._sort_states()
        return self

    def _sort_states(self) -> None:
        """Reordena estados por σ ascendente → 0=calm, K-1=panic."""
        order = np.argsort(self.sigma)
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]
        self.pi = self.pi[order]
        self.A = self.A[np.ix_(order, order)]

    def filtered_probs(self, x: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..t) — forward only, SIN look-ahead. (T,K)."""
        x = np.asarray(x, dtype=float)
        B = self._emissions(x)
        T, K = len(x), self.K
        alpha = np.empty((T, K))
        alpha[0] = self.pi * B[0]
        alpha[0] /= max(alpha[0].sum(), 1e-300)
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            alpha[t] /= max(alpha[t].sum(), 1e-300)
        return alpha

    def expected_durations(self) -> np.ndarray:
        """Duración esperada de cada estado en días: 1/(1-A_ii)."""
        return 1.0 / np.maximum(1.0 - np.diag(self.A), 1e-6)


def fit_volatility_regime(returns: pd.Series, n_states: int = 3,
                          seed: int = 7) -> dict:
    """
    Ajusta el HMM a retornos diarios (decimal, p.ej. 0.012 = +1.2%).

    Returns dict:
      probs        : DataFrame (index=fechas) con P filtrada de calm/transition/panic
      current      : dict {label: prob} del último día
      state        : label del estado más probable hoy
      sigmas_ann   : vol anualizada de cada estado (%)
      transmat     : DataFrame KxK
      durations    : duración esperada por estado (días)
    o dict vacío si la serie es insuficiente.
    """
    r = returns.dropna().astype(float)
    if len(r) < 500:
        return {}

    try:
        hmm = GaussianHMM1D(n_states=n_states, seed=seed).fit(r.values)
    except (ValueError, np.linalg.LinAlgError):
        return {}

    labels = list(REGIME_LABELS[:n_states])
    probs = pd.DataFrame(hmm.filtered_probs(r.values), index=r.index,
                         columns=labels)
    cur = probs.iloc[-1]
    return {
        "probs": probs,
        "current": cur.to_dict(),
        "state": str(cur.idxmax()),
        "sigmas_ann": (hmm.sigma * np.sqrt(252) * 100).tolist(),
        "transmat": pd.DataFrame(hmm.A, index=labels, columns=labels),
        "durations": hmm.expected_durations().tolist(),
    }
