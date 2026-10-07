"""
theme.py — Sistema visual de Spread Trading Club para el VIX Controller.

Fuente: Spread Trading Club · design-system.md v4 (2026-06-28).

  Terminal financiero (registro Bloomberg) + la pertenencia de un Club.
  Casi monocromo, UN solo acento ámbar. Datos siempre en monoespaciada.
  Ganancia/pérdida nunca solo por color: siempre ▲/▼.
  Plano: sin gradientes ni sombras pesadas; separación por línea o por
  elevación de superficie. Sin emojis, sin neón, sin promesas.

Todo lo visual de la app sale de aquí: tokens, CSS, plantilla de Plotly y
logotipo. Para cambiar la marca se toca este fichero y nada más.
"""
from __future__ import annotations

import html as _html

import plotly.graph_objects as go
import plotly.io as pio

# ──────────────────────────────────────────────────────────────────────
# Tokens — "Terminal Pro"
# ──────────────────────────────────────────────────────────────────────
BLACK = "#0A0B0D"
SURFACE = "#15171A"
SURFACE_2 = "#1F2630"
LINE = "#262A30"
GRID = "#1C1F24"
ZERO = "#3A3F47"
AMBER = "#F5A623"
AMBER_DIM = "#C9821A"
WHITE = "#F4F5F6"
GRAY = "#9AA1A9"
GRAY_LIGHT = "#6B6B66"
MUTED = "#5E656E"
PROFIT = "#16C784"
LOSS = "#EA3943"
AMBER_FILL = "rgba(245,166,35,0.14)"

# Series de gráfico: ámbar marca lo que importa; el resto, escala de grises.
SERIES = [AMBER, WHITE, GRAY, AMBER_DIM, "#C7CCD1", MUTED, PROFIT, LOSS]

FONT_DISPLAY = "'Space Grotesk', Inter, system-ui, sans-serif"
FONT_BODY = "Inter, system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"


# ──────────────────────────────────────────────────────────────────────
# Logotipo (SVG oficial, brand-assets/icon-color.svg)
# ──────────────────────────────────────────────────────────────────────
ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 96" width="{w}" height="{h}" '
    'role="img" aria-label="Spread Trading Club">'
    '<line x1="12" y1="78" x2="108" y2="78" stroke="#262A30" stroke-width="1.2"/>'
    '<path d="M48 78 L48 36 Q60 20 72 36 L72 78 Z" fill="#F5A623" fill-opacity="0.14"/>'
    '<path d="M12 78 C 34 78, 44 26, 60 24 C 76 26, 86 78, 108 78" fill="none" '
    'stroke="#F5A623" stroke-width="4.2" stroke-linecap="round"/>'
    '<line x1="48" y1="36" x2="48" y2="82" stroke="#F4F5F6" stroke-width="2"/>'
    '<line x1="72" y1="36" x2="72" y2="82" stroke="#F4F5F6" stroke-width="2"/>'
    '<circle cx="48" cy="36" r="2.8" fill="#F4F5F6"/>'
    '<circle cx="72" cy="36" r="2.8" fill="#F4F5F6"/></svg>'
)


def icon(width: int = 46) -> str:
    return ICON_SVG.format(w=width, h=int(width * 0.8))


# ──────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');
:root{
  --black:#0A0B0D;--surface:#15171A;--surface-2:#1F2630;--line:#262A30;
  --amber:#F5A623;--amber-dim:#C9821A;--white:#F4F5F6;--gray:#9AA1A9;--gray-light:#6B6B66;
  --profit:#16C784;--loss:#EA3943;--profit-fill:rgba(245,166,35,0.14);
  --radius-card:10px;--radius-container:14px;--radius-pill:4px;
  --dur-micro:120ms;--dur-base:200ms;--dur-enter:320ms;--ease:cubic-bezier(0.4,0,0.2,1);
  --display:'Space Grotesk',Inter,system-ui,sans-serif;--body:Inter,system-ui,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,monospace;
  /* alias de compatibilidad para el marcado existente */
  --bg:var(--black);--card:var(--surface);--card2:var(--surface-2);--border:var(--line);
  --g:var(--profit);--r:var(--loss);--y:var(--amber);--b:var(--white);--c:var(--gray);
  --t:var(--white);--dim:var(--gray);--w:var(--white);--accent:var(--amber);--accent2:var(--amber-dim);
  --purple:var(--amber-dim);--gbg:rgba(22,199,132,0.06);--rbg:rgba(234,57,67,0.07);
  --thead:var(--surface-2);--grid:#1C1F24;--zero:#3A3F47;
}

/* ── Base ─────────────────────────────────────────────────────── */
.stApp{background:var(--black);color:var(--white);font-family:var(--body);}
.block-container{padding:4.6rem 2rem 3rem;max-width:1360px;}
/* La navegación superior vive DENTRO de stToolbar: solo se ocultan sus acciones. */
[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],
[data-testid="stDecoration"],footer,#MainMenu,[data-testid="stStatusWidget"]{display:none!important;}
h1,h2,h3,h4{font-family:var(--display);color:var(--white);letter-spacing:-0.01em;}
p,li{font-family:var(--body);line-height:1.55;}
code{font-family:var(--mono);color:var(--amber);background:var(--surface);}
a{color:var(--amber);}
@keyframes enter{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}

/* ── Navegación superior (st.navigation, position=top) ────────── */
header[data-testid="stHeader"]{background:var(--black);border-bottom:1px solid var(--line);height:3.25rem;}
[data-testid="stTopNavLink"],[data-testid="stTopNavLink"] span{font-family:var(--mono)!important;
  font-size:0.72rem!important;letter-spacing:0.14em;text-transform:uppercase;color:var(--gray)!important;}
[data-testid="stTopNavLink"]{border-radius:var(--radius-pill);padding:0.35rem 0.8rem!important;
  transition:color var(--dur-micro) var(--ease),background var(--dur-micro) var(--ease);}
[data-testid="stTopNavLink"]:hover,[data-testid="stTopNavLink"]:hover span{color:var(--white)!important;
  background:var(--surface)!important;}
[data-testid="stTopNavLink"][aria-current="page"],[data-testid="stTopNavLink"][aria-current="page"] span{
  color:var(--amber)!important;background:var(--surface)!important;}

/* ── Cabecera de marca ────────────────────────────────────────── */
.stc-head{display:flex;align-items:center;gap:1.25rem;padding:0.25rem 0 1rem;
  border-bottom:1px solid var(--line);margin-bottom:1.25rem;animation:enter var(--dur-enter) var(--ease);}
.stc-lockup{display:flex;align-items:center;gap:0.85rem;}
.stc-word{display:flex;flex-direction:column;line-height:1;}
.stc-word b{font-family:var(--display);font-weight:700;font-size:1.45rem;color:var(--white);letter-spacing:-0.01em;}
.stc-word span{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.42em;color:var(--gray);margin-top:5px;}
.stc-product{border-left:1px solid var(--line);padding-left:1.25rem;}
.stc-product b{display:block;font-family:var(--mono);font-size:0.72rem;letter-spacing:0.18em;color:var(--amber);}
.stc-product span{font-family:var(--body);font-size:0.82rem;color:var(--gray);}
.stc-meta{margin-left:auto;text-align:right;font-family:var(--mono);font-size:0.7rem;color:var(--gray);line-height:1.7;}
.stc-meta .on{color:var(--profit);} .stc-meta .off{color:var(--gray);}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:1px;}
.dot.on{background:var(--profit);} .dot.off{background:#3A3F47;}

/* ── Tira de estado ───────────────────────────────────────────── */
.statusbar{display:flex;flex-wrap:wrap;margin:0 0 1.5rem;border:1px solid var(--line);
  border-radius:var(--radius-card);background:var(--surface);overflow:hidden;}
.statusbar .sb-item{flex:1;min-width:118px;padding:0.7rem 1rem;border-right:1px solid var(--line);}
.statusbar .sb-item:last-child{border-right:none;}
.statusbar .sb-l{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--gray);}
.statusbar .sb-v{font-family:var(--mono);font-weight:600;font-size:1.02rem;color:var(--white);
  margin-top:3px;font-variant-numeric:tabular-nums;}
.statusbar .sb-item.hl{background:rgba(245,166,35,0.06);}
.statusbar .sb-item.hl .sb-l{color:var(--amber);}

/* ── Tipografía de sección ────────────────────────────────────── */
.stc-eyebrow{font-family:var(--mono);font-size:0.7rem;font-weight:500;letter-spacing:0.18em;
  text-transform:uppercase;color:var(--amber);margin:0 0 0.5rem;}
.stc-h1{font-family:var(--display);font-weight:700;font-size:2.1rem;line-height:1.12;color:var(--white);margin:0;}
.stc-h2{font-family:var(--display);font-weight:500;font-size:1.35rem;line-height:1.15;color:var(--white);margin:0 0 0.35rem;}
.stc-lead{font-family:var(--body);font-size:1rem;line-height:1.55;color:var(--gray);max-width:70ch;margin:0.6rem 0 0;}
.stc-section{margin:2.25rem 0 0.9rem;padding-top:1.5rem;border-top:1px solid var(--line);animation:enter var(--dur-enter) var(--ease);}
.stc-note{font-family:var(--body);font-size:0.82rem;line-height:1.55;color:var(--gray);max-width:80ch;}
.stc-badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:0.9rem;}
.stc-badge{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;
  color:var(--gray);border:1px solid var(--line);border-radius:var(--radius-pill);padding:0.3rem 0.55rem;}
.stc-badge.amber{color:var(--amber);border-color:rgba(245,166,35,0.45);}
.stc-badge.loss{color:var(--loss);border-color:rgba(234,57,67,0.45);}
.stc-badge.profit{color:var(--profit);border-color:rgba(22,199,132,0.45);}

/* ── Tarjetas ─────────────────────────────────────────────────── */
.stc-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-card);
  padding:1.1rem 1.25rem;height:100%;animation:enter var(--dur-enter) var(--ease);}
.stc-card.hero{min-height:19.5rem;}
.stc-card.accent{border-left:3px solid var(--amber);}
.stc-card.loss{border-left:3px solid var(--loss);}
.stc-card.profit{border-left:3px solid var(--profit);}
.stc-label{font-family:var(--mono);font-size:0.64rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--gray);}
.stc-big{font-family:var(--display);font-weight:700;font-size:3.2rem;line-height:1;letter-spacing:-0.02em;margin:0.45rem 0 0.35rem;}
.stc-big.amber{color:var(--amber);} .stc-big.white{color:var(--white);} .stc-big.loss{color:var(--loss);}
.stc-sub{font-family:var(--body);font-size:0.86rem;color:var(--gray);line-height:1.5;}
.stc-val{font-family:var(--mono);font-weight:600;font-size:1.55rem;color:var(--white);font-variant-numeric:tabular-nums;}
.stc-row{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;padding:0.5rem 0;
  border-bottom:1px solid var(--line);font-family:var(--mono);font-size:0.8rem;}
.stc-row:last-child{border-bottom:none;}
.stc-row .k{color:var(--gray);font-family:var(--body);font-size:0.84rem;}
.stc-row .v{color:var(--white);font-variant-numeric:tabular-nums;text-align:right;}
.up{color:var(--profit)!important;} .dn{color:var(--loss)!important;} .am{color:var(--amber)!important;} .mu{color:var(--gray)!important;}

/* Medida contra 1,00 */
.stc-gauge{position:relative;height:6px;background:var(--surface-2);border-radius:3px;margin:0.65rem 0 0.3rem;}
.stc-gauge .fill{position:absolute;top:0;bottom:0;border-radius:3px;}
.stc-gauge .one{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--white);}
.stc-gauge-scale{display:flex;justify-content:space-between;font-family:var(--mono);font-size:0.6rem;color:var(--gray-light);}

/* Avisos de marca */
.stc-alert{border:1px solid var(--line);border-left:3px solid var(--amber);background:var(--surface);
  border-radius:var(--radius-pill);padding:0.85rem 1rem;margin:0.75rem 0;font-family:var(--body);
  font-size:0.92rem;color:var(--white);line-height:1.5;}
.stc-alert.loss{border-left-color:var(--loss);} .stc-alert.profit{border-left-color:var(--profit);}
.stc-alert b{font-family:var(--mono);font-size:0.72rem;letter-spacing:0.16em;text-transform:uppercase;
  color:var(--amber);display:block;margin-bottom:0.25rem;}
.stc-alert.loss b{color:var(--loss);} .stc-alert.profit b{color:var(--profit);}

/* Tablas */
.stc-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.78rem;}
.stc-table th{font-weight:500;font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--gray);
  text-align:right;padding:0.55rem 0.6rem;border-bottom:1px solid var(--line);}
.stc-table th:first-child,.stc-table td:first-child{text-align:left;}
.stc-table td{padding:0.5rem 0.6rem;text-align:right;color:var(--white);border-bottom:1px solid #1C1F24;
  font-variant-numeric:tabular-nums;}
.stc-table tr:hover td{background:var(--surface);}
.stc-table tr.lose td{color:var(--loss);} .stc-table tr.open td{color:var(--amber);}
.stc-table.text td,.stc-table.text th{text-align:left;font-family:var(--body);font-size:0.86rem;line-height:1.45;vertical-align:top;}
.stc-table.text th{font-family:var(--mono);font-size:0.62rem;}
.stc-table.text td:first-child{font-family:var(--mono);font-size:0.78rem;color:var(--amber);white-space:nowrap;}

/* ── Componentes heredados, re-tematizados ───────────────────── */
.hr{border:none;border-top:1px solid var(--line);margin:1.4rem 0;}
.mrow{display:flex;gap:8px;margin-bottom:1rem;flex-wrap:wrap;}
.mpill{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-card);
  padding:0.7rem 0.9rem;flex:1;min-width:128px;text-align:left;}
.mpill .ml{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--gray);}
.mpill .mv{font-family:var(--mono);font-weight:600;font-size:1.25rem;color:var(--white);margin-top:4px;font-variant-numeric:tabular-nums;}
.mv.up{color:var(--profit);} .mv.dn{color:var(--loss);} .mv.nt{color:var(--white);}
.icard{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-card);
  padding:1rem 1.15rem;margin-bottom:0.8rem;}
.icard .ic-title{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.16em;text-transform:uppercase;
  color:var(--amber);margin-bottom:0.6rem;border-bottom:1px solid var(--line);padding-bottom:0.5rem;}
.icard .ic-row{display:flex;justify-content:space-between;gap:1rem;padding:0.28rem 0;font-family:var(--mono);font-size:0.78rem;}
.icard .ic-label{color:var(--gray);font-family:var(--body);font-size:0.82rem;}
.icard .ic-val{color:var(--white);font-variant-numeric:tabular-nums;text-align:right;}
.chk{display:flex;align-items:center;gap:0.55rem;padding:0.35rem 0;font-family:var(--body);font-size:0.85rem;color:var(--white);}
.chk .ok{color:var(--profit);font-family:var(--mono);font-weight:600;} .chk .no{color:var(--loss);font-family:var(--mono);font-weight:600;}
.sig-box{border-radius:var(--radius-card);padding:1.2rem;text-align:left;border:1px solid var(--line);background:var(--surface);}
.sig-long{border-left:3px solid var(--profit);} .sig-cash{border-left:3px solid var(--loss);}
.sig-box .sl{font-family:var(--display);font-weight:700;font-size:2rem;}
.sig-box .sd{font-family:var(--mono);font-size:0.7rem;color:var(--gray);margin-top:4px;}
.ctx{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.76rem;margin:0.5rem 0;}
.ctx td,.ctx th{padding:0.45rem 0.55rem;text-align:center;border-bottom:1px solid var(--line);}
.ctx th{color:var(--gray);font-weight:500;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;}
.ctx .pos{color:var(--profit);} .ctx .neg{color:var(--loss);}
.ctx .hdr-cell{color:var(--gray);font-family:var(--body);text-align:left;width:150px;}
.dtbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.76rem;margin-top:0.6rem;}
.dtbl th{color:var(--gray);font-weight:500;padding:0.55rem 0.6rem;border-bottom:1px solid var(--line);
  font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;text-align:center;}
.dtbl td{padding:0.45rem 0.6rem;text-align:center;color:var(--white);border-bottom:1px solid #1C1F24;font-variant-numeric:tabular-nums;}
.dtbl tr:hover td{background:var(--surface);}

/* ── Widgets nativos ──────────────────────────────────────────── */
.stButton>button,.stDownloadButton>button{min-height:44px;background:transparent;border:1px solid var(--line);
  color:var(--white);font-family:var(--body);font-weight:600;font-size:0.85rem;border-radius:var(--radius-pill);
  transition:all var(--dur-micro) var(--ease);}
.stButton>button:hover{border-color:var(--amber);color:var(--amber);}
.stButton>button[kind="primary"]{background:var(--amber);border-color:var(--amber);color:#1A1303;}
.stButton>button[kind="primary"]:hover{background:var(--amber-dim);border-color:var(--amber-dim);color:#1A1303;}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--line)!important;border-radius:var(--radius-card);}
[data-testid="stExpander"] summary p{font-family:var(--body);font-weight:500;color:var(--white);}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-card);padding:0.8rem 1rem;}
[data-testid="stMetricLabel"] p{font-family:var(--mono)!important;font-size:0.62rem!important;letter-spacing:0.16em;text-transform:uppercase;color:var(--gray)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono);font-weight:600;color:var(--white);}
[data-testid="stAlertContainer"]{background:var(--surface)!important;border:1px solid var(--line);
  border-left:3px solid var(--gray);border-radius:var(--radius-pill);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){border-left-color:var(--amber);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){border-left-color:var(--loss);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){border-left-color:var(--profit);}
[data-testid="stAlertContainer"] p,[data-testid="stAlertContainer"] li{color:var(--white)!important;font-family:var(--body);}
[data-testid="stWidgetLabel"] p{font-family:var(--mono)!important;font-size:0.66rem!important;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--gray)!important;}
[data-testid="stCaptionContainer"] p{font-family:var(--body);color:var(--gray-light);font-size:0.8rem;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.14em;text-transform:uppercase;
  color:var(--gray);padding:0.6rem 1rem;}
.stTabs [aria-selected="true"]{color:var(--amber)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--amber)!important;}
.stTabs [data-baseweb="tab-border"]{display:none;}
::-webkit-scrollbar{width:10px;height:10px;} ::-webkit-scrollbar-track{background:var(--black);}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px;}

/* ── Pie ──────────────────────────────────────────────────────── */
.stc-foot{margin-top:3rem;padding-top:1.25rem;border-top:1px solid var(--line);display:flex;flex-wrap:wrap;
  gap:1rem;justify-content:space-between;font-family:var(--mono);font-size:0.64rem;letter-spacing:0.12em;color:var(--gray-light);}
@media (max-width:760px){.block-container{padding:4.4rem 1rem 2rem;} .stc-meta{display:none;} .stc-big{font-size:2.4rem;}
  .stc-h1{font-size:1.6rem;}}
</style>
"""


# ──────────────────────────────────────────────────────────────────────
# Plotly
# ──────────────────────────────────────────────────────────────────────
def register_plotly_template() -> None:
    """Plantilla 'stc' sobre plotly_dark y la deja por defecto."""
    if "stc" in pio.templates:
        pio.templates.default = "stc"
        return
    base = go.layout.Template(pio.templates["plotly_dark"])
    base.layout.update(
        font=dict(family=FONT_BODY, color=GRAY, size=12),
        title=dict(font=dict(family=FONT_DISPLAY, color=WHITE, size=15), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=ZERO,
                   tickfont=dict(family=FONT_MONO, size=10, color=GRAY)),
        yaxis=dict(gridcolor=GRID, linecolor=LINE, zerolinecolor=ZERO,
                   tickfont=dict(family=FONT_MONO, size=10, color=GRAY)),
        legend=dict(font=dict(family=FONT_MONO, size=10, color=GRAY), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE,
                        font=dict(family=FONT_MONO, size=11, color=WHITE)),
        margin=dict(l=48, r=20, t=36, b=36),
        separators=",.",                       # convención española: 1.234,56
    )
    pio.templates["stc"] = base
    pio.templates.default = "stc"


# ──────────────────────────────────────────────────────────────────────
# Fragmentos HTML
# ──────────────────────────────────────────────────────────────────────
def esc(s) -> str:
    return _html.escape(str(s))


def eyebrow(text: str) -> str:
    return f'<div class="stc-eyebrow">{esc(text)}</div>'


def section(label: str, title: str, lead: str = "") -> str:
    lead_html = f'<p class="stc-lead">{lead}</p>' if lead else ""
    return (f'<div class="stc-section">{eyebrow(label)}'
            f'<div class="stc-h2">{esc(title)}</div>{lead_html}</div>')


def badge(text: str, kind: str = "") -> str:
    return f'<span class="stc-badge {kind}">{esc(text)}</span>'


def arrow(v: float | None, ref: float = 0.0) -> str:
    """▲/▼ según el signo respecto a `ref` (regla daltónica de la marca)."""
    if v is None or v != v:
        return ""
    return "▲" if v > ref else "▼"
