"""
VIX Controller — Bloomberg-Style Term Structure + Operational Monitor
Tab 1: CBOE live (Playwright)
Tab 2: Monitor operativo (CSV Drive + yfinance live para señal)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
import re, time, warnings, logging, os
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

st.set_page_config(page_title="VIX Controller", page_icon="🔴", layout="wide",
                   initial_sidebar_state="collapsed")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLAYWRIGHT — verifica instalación UNA vez
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def check_playwright_installed() -> bool:
    log = logging.getLogger("vix_controller")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
            b.close()
        log.info("Playwright check: Chromium OK")
        return True
    except Exception as e:
        log.error(f"Playwright check failed: {e}")
        return False

pw_ready = check_playwright_installed()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS BLOOMBERG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0D1117;--card:#161B22;--border:#30363D;--g:#3FB950;--r:#F85149;--y:#D29922;--b:#58A6FF;--c:#39D2C0;--t:#C9D1D9;--dim:#8B949E;--w:#F0F6FC;--gbg:#0B2E13;--rbg:#3B1218;}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0.5rem 1.5rem;max-width:1400px;}
.hdr{display:flex;align-items:center;padding:0.5rem 0;border-bottom:2px solid #F7931A;margin-bottom:0.8rem;}
.hdr .logo{font-family:'Inter',sans-serif;font-weight:800;font-size:1.3rem;color:#F7931A;letter-spacing:1px;}
.hdr .sub{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-left:auto;}
.mrow{display:flex;gap:4px;margin-bottom:0.6rem;flex-wrap:wrap;}
.mpill{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:0.4rem 0.7rem;flex:1;min-width:120px;text-align:center;}
.mpill .ml{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.6px;}
.mpill .mv{font-family:'Inter',sans-serif;font-weight:700;font-size:1.15rem;}
.mv.up{color:var(--g);}.mv.dn{color:var(--r);}.mv.nt{color:var(--b);}
.ctx{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.78rem;margin:0.4rem 0;}
.ctx td,.ctx th{padding:0.35rem 0.5rem;text-align:center;border:1px solid var(--border);}
.ctx th{background:#1C2128;color:var(--dim);font-weight:500;font-size:0.65rem;text-transform:uppercase;}
.ctx .pos{color:var(--g);}.ctx .neg{color:var(--r);}
.ctx .hdr-cell{background:var(--card);color:var(--t);font-weight:600;text-align:left;width:120px;}
.dtbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.75rem;margin-top:0.5rem;}
.dtbl th{color:var(--b);font-weight:500;padding:0.4rem 0.6rem;border-bottom:1px solid var(--border);font-size:0.62rem;text-transform:uppercase;letter-spacing:0.5px;text-align:center;background:#1C2128;}
.dtbl td{padding:0.35rem 0.6rem;text-align:center;color:var(--t);border-bottom:1px solid rgba(255,255,255,0.03);}
.dtbl tr:hover td{background:rgba(88,166,255,0.04);}
.sig-box{border-radius:6px;padding:1rem;text-align:center;border-width:2px;border-style:solid;}
.sig-long{background:var(--gbg);border-color:var(--g);}
.sig-cash{background:var(--rbg);border-color:var(--r);}
.sig-box .sl{font-family:'Inter',sans-serif;font-weight:800;font-size:2rem;}
.sig-box .sd{font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:var(--dim);margin-top:2px;}
.chk{display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--t);}
.chk .ok{color:var(--g);font-weight:700;}.chk .no{color:var(--r);font-weight:700;}
.icard{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;}
.icard .ic-title{font-family:'Inter',sans-serif;font-weight:700;font-size:0.85rem;color:var(--w);margin-bottom:0.5rem;border-bottom:1px solid var(--border);padding-bottom:0.3rem;}
.icard .ic-row{display:flex;justify-content:space-between;padding:0.2rem 0;font-family:'JetBrains Mono',monospace;font-size:0.8rem;}
.icard .ic-label{color:var(--dim);}.icard .ic-val{color:var(--t);font-weight:500;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-weight:600;font-size:0.82rem;color:var(--dim);padding:0.5rem 1.5rem;}
.stTabs [aria-selected="true"]{color:#F7931A !important;border-bottom:2px solid #F7931A !important;}
[data-testid="stSidebar"]{background:var(--card);}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CBOE_URL      = 'https://www.cboe.com/delayed_quotes/futures/future_quotes'
DRIVE_FILE_ID = "12fzSq4BgkppRjoupeMjM67jCB8Qwo8Yz"
DRIVE_URL     = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
MN = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
      7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

def cpct(p1, p2):
    if p1 and p2 and p1 > 0:
        return round((p2 - p1) / p1 * 100, 2)
    return None

def fv(v):
    return f"{v:.2f}" if v is not None and pd.notna(v) and v != 0 else "—"

def vc(v):
    if v is None: return "nt"
    return "up" if v >= 0 else "dn"

def fp(v):
    if v is None: return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}%"

def mcard(label, val, clr="nt"):
    return f'<div class="mpill"><div class="ml">{label}</div><div class="mv {clr}">{val}</div></div>'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 DATA — CBOE (Playwright, TTL 55s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=55)
def scrape_cboe_futures() -> pd.DataFrame:
    log = logging.getLogger("vix_controller")
    if not pw_ready:
        return pd.DataFrame()
    from playwright.sync_api import sync_playwright
    html = ""
    try:
        log.info("CBOE_SCRAPE: lanzando Chromium...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True,
                args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-first-run'])
            page = browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width':1920,'height':1080}, locale='en-US')
            page.route("**/googletagmanager**", lambda r: r.abort())
            page.route("**/google-analytics**", lambda r: r.abort())
            page.goto(CBOE_URL, wait_until='networkidle', timeout=45000)
            try:
                page.wait_for_function("() => document.body.innerText.includes('VX/')", timeout=25000)
            except:
                pass
            html = page.content()
            browser.close()
        log.info(f"CBOE_SCRAPE: HTML {len(html):,} chars · VX/ hits: {html.count('VX/')}")
    except Exception as e:
        log.error(f"CBOE_SCRAPE: error — {e}")
        st.session_state["scrape_debug"] = f"❌ Error: {e}"
        return pd.DataFrame()

    try:
        all_tables = pd.read_html(StringIO(html))
        log.info(f"CBOE_SCRAPE: {len(all_tables)} tablas en HTML")
    except:
        return pd.DataFrame()

    df_vx = pd.DataFrame()
    for i, df in enumerate(all_tables):
        cols_upper = [str(c).upper().strip() for c in df.columns]
        if 'SYMBOL' in cols_upper and 'EXPIRATION' in cols_upper:
            sym_col = df.columns[cols_upper.index('SYMBOL')]
            if df[sym_col].astype(str).str.startswith('VX').any():
                df_vx = df.copy()
                log.info(f"CBOE_SCRAPE: tabla VX en índice {i} ✅")
                break

    if df_vx.empty:
        return pd.DataFrame()

    df_vx.columns = [str(c).strip().upper() for c in df_vx.columns]
    rename = {'SYMBOL':'Symbol','EXPIRATION':'Expiration','LAST':'Last','CHANGE':'Change',
               'HIGH':'High','LOW':'Low','SETTLEMENT':'Settlement','VOLUME':'Volume'}
    df_vx.rename(columns={k:v for k,v in rename.items() if k in df_vx.columns}, inplace=True)
    mask = df_vx['Symbol'].astype(str).str.match(r'^VX/[A-Z]\d+$')
    df_vx = df_vx[mask].reset_index(drop=True)
    if 'Expiration' in df_vx.columns:
        df_vx['Expiration'] = pd.to_datetime(df_vx['Expiration'], errors='coerce')
        df_vx = df_vx.sort_values('Expiration').reset_index(drop=True)
    for col in ['Last','Change','High','Low','Settlement','Volume']:
        if col in df_vx.columns:
            df_vx[col] = pd.to_numeric(df_vx[col].astype(str).str.replace(',','',regex=False), errors='coerce')
    today = pd.Timestamp('today').normalize()
    if 'Expiration' in df_vx.columns:
        df_vx['DTE'] = (df_vx['Expiration'] - today).dt.days
    df_vx['Price'] = df_vx.apply(
        lambda r: r['Last'] if pd.notna(r.get('Last')) and r.get('Last',0) > 0 else r.get('Settlement',0), axis=1)
    df_vx['Scraped_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"CBOE_SCRAPE: {len(df_vx)} contratos mensuales ✅")
    return df_vx

@st.cache_data(ttl=55)
def fetch_vix_spot():
    try:
        h = yf.Ticker("^VIX").history(period="5d")
        if not h.empty:
            c = round(float(h['Close'].iloc[-1]),2)
            p = round(float(h['Close'].iloc[-2]),2) if len(h)>1 else c
            return dict(price=c, prev=p, chg=round(c-p,2))
    except: pass
    return None

@st.cache_data(ttl=55)
def fetch_etps():
    out = {}
    for name, sym in [("VXX","VXX"),("SVXY","SVXY"),("SVIX","SVIX"),("SPY","SPY")]:
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                out[name] = dict(
                    close=round(float(h['Close'].iloc[-1]),2),
                    prev=round(float(h['Close'].iloc[-2]),2) if len(h)>1 else None)
        except: continue
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 DATA — CSV Drive + Strategy (TTL 300s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=300)
def load_and_process_strategy() -> tuple:
    """Carga CSV + aplica estrategia. Cacheado 5 min. Retorna (bt, trades_df, metrics)."""
    log = logging.getLogger("vix_controller")
    tmp = "/tmp/master_historico.csv"
    try:
        import gdown
        gdown.download(DRIVE_URL, tmp, quiet=True, fuzzy=True)
        df = pd.read_csv(tmp, index_col=0, parse_dates=True).sort_index()
        log.info(f"CSV cargado: {len(df):,} filas · {df.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        log.error(f"Error CSV: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Filtrar período con datos completos
    bt = df[df['VXX_Close'].notna() & df['SVXY_Close'].notna() & df['M1_Price'].notna()].copy()

    # Indicadores BB
    vxx = bt['VXX_Close']
    bt['BB_SMA20'] = vxx.rolling(20).mean()
    bt['BB_STD20'] = vxx.rolling(20).std()
    bt['BB_Upper'] = bt['BB_SMA20'] + 2.0 * bt['BB_STD20']
    bt['BB_Lower'] = bt['BB_SMA20'] - 2.0 * bt['BB_STD20']

    # Señal BB
    sig = pd.Series(0, index=bt.index)
    pos = 0
    for i in range(len(bt)):
        p_, s_, u_ = bt['VXX_Close'].iloc[i], bt['BB_SMA20'].iloc[i], bt['BB_Upper'].iloc[i]
        if pd.isna(s_) or pd.isna(u_) or pd.isna(p_):
            sig.iloc[i] = pos; continue
        if pos == 0 and p_ < s_: pos = 1
        elif pos == 1 and p_ > u_: pos = 0
        sig.iloc[i] = pos

    bt['sig_bb']     = sig.shift(1).fillna(0)
    bt['ct_filter']  = bt['In_Contango'].fillna(0).astype(int)
    bt['sig_final']  = (bt['sig_bb'] * bt['ct_filter']).astype(int)
    bt['strat_ret']  = bt['SVXY_ret'] * bt['sig_final']
    bt['equity']     = (1 + bt['strat_ret'].fillna(0)).cumprod()

    # Trades
    trades = []
    entry_date = None
    s = bt['sig_final']
    for i in range(1, len(s)):
        if s.iloc[i] == 1 and s.iloc[i-1] == 0:
            entry_date = s.index[i]
        elif s.iloc[i] == 0 and s.iloc[i-1] == 1 and entry_date is not None:
            exit_date = s.index[i]
            rets = bt['SVXY_ret'].loc[entry_date:exit_date].dropna()
            if len(rets) == 0: entry_date = None; continue
            prev = s.index[i-1]
            bb_exit = bt.loc[prev,'VXX_Close'] > bt.loc[prev,'BB_Upper'] if prev in bt.index else False
            ct_exit = bt.loc[prev,'In_Contango'] == 0 if prev in bt.index else False
            reason = 'Ambas' if (ct_exit and bb_exit) else ('Contango Rule' if ct_exit else 'BB Superior')
            trades.append({
                'Entrada': entry_date.strftime('%Y-%m-%d'),
                'Salida':  exit_date.strftime('%Y-%m-%d'),
                'Días':    len(rets),
                'Retorno': round(((1+rets).prod()-1)*100, 2),
                'Razón':   reason,
                'VIX':     round(bt.loc[entry_date,'VIX_Close'],1) if entry_date in bt.index else None,
                'Contango': round(bt.loc[entry_date,'Contango_pct'],2) if entry_date in bt.index else None,
            })
            entry_date = None
    if entry_date is not None:
        rets = bt['SVXY_ret'].loc[entry_date:].dropna()
        trades.append({
            'Entrada': entry_date.strftime('%Y-%m-%d'), 'Salida': '🔴 ABIERTO',
            'Días': len(rets), 'Retorno': round(((1+rets).prod()-1)*100,2) if len(rets)>0 else 0,
            'Razón': '—',
            'VIX': round(bt.loc[entry_date,'VIX_Close'],1) if entry_date in bt.index else None,
            'Contango': round(bt.loc[entry_date,'Contango_pct'],2) if entry_date in bt.index else None,
        })
    trades_df = pd.DataFrame(trades)

    # Métricas
    sr = bt['strat_ret'].dropna()
    eq = bt['equity']
    years = len(sr)/252
    cagr   = (eq.iloc[-1]**(1/years)-1)*100 if years>0 else 0
    peak   = eq.cummax()
    mdd    = ((eq-peak)/peak).min()*100
    sharpe = sr.mean()/sr.std()*np.sqrt(252) if sr.std()>0 else 0
    calmar = abs(cagr/mdd) if mdd!=0 else 0
    wr     = (sr[sr!=0]>0).mean()*100 if (sr!=0).sum()>0 else 0
    exp    = bt['sig_final'].mean()*100
    yearly = {}
    for yr in sorted(set(eq.index.year)):
        yr_r = sr[sr.index.year==yr]
        if len(yr_r)>20:
            yearly[yr] = round(((1+yr_r).cumprod().iloc[-1]-1)*100,1)
    metrics = dict(cagr=round(cagr,1), mdd=round(mdd,1), sharpe=round(sharpe,2),
                   calmar=round(calmar,2), wr=round(wr,1), exp=round(exp,1),
                   yearly=yearly, equity=eq)
    return bt, trades_df, metrics

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_term_chart(vix_spot, df_vx, show_prev=True):
    fig = go.Figure()
    if df_vx.empty: return fig
    labels = []
    for _, r in df_vx.iterrows():
        exp = r.get('Expiration')
        labels.append(MN.get(exp.month, str(exp.month)[:3]) if pd.notna(exp) else str(r.get('Symbol','')))
    xpos = list(range(len(df_vx)))
    prices = df_vx['Price'].tolist()
    prev_prices = []
    for _, r in df_vx.iterrows():
        p, c = r['Price'], r.get('Change', 0)
        prev_prices.append(round(p-c, 4) if pd.notna(p) and p>0 and pd.notna(c) else None)
    vx = [x for x,y in zip(xpos,prices) if pd.notna(y) and y>0]
    vy = [y for y in prices if pd.notna(y) and y>0]
    if vy:
        fig.add_trace(go.Scatter(x=vx, y=vy, mode='lines+markers+text', name='Last',
            line=dict(color='#4A90D9',width=3,shape='spline'),
            marker=dict(size=9,color='#4A90D9',line=dict(width=2,color='#0D1117')),
            text=[f"{v:.3f}" for v in vy], textposition='top center',
            textfont=dict(size=10,color='#C9D1D9',family='JetBrains Mono'),
            hovertemplate='%{text}<extra></extra>'))
    if show_prev:
        pvx = [x for x,y in zip(xpos,prev_prices) if y and y>0]
        pvy = [y for y in prev_prices if y and y>0]
        if len(pvy)>=2:
            fig.add_trace(go.Scatter(x=pvx, y=pvy, mode='lines+markers', name='Previous Close',
                line=dict(color='#8B949E',width=1.5,dash='dot',shape='spline'),
                marker=dict(size=5,color='#8B949E',symbol='diamond'),
                hovertemplate='Prev: %{y:.3f}<extra></extra>'))
    if vix_spot:
        fig.add_hline(y=vix_spot['price'], line_dash="dash", line_color="#3FB950", line_width=2,
                      annotation_text=f"  {vix_spot['price']:.2f}",
                      annotation_font=dict(size=12,color="#3FB950",family="Inter"))
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='lines',name='VIX Index',
            line=dict(color='#3FB950',width=2,dash='dash'),showlegend=True))
    all_y = vy + ([vix_spot['price']] if vix_spot else [])
    y_min = min(all_y)-1.5 if all_y else 15
    y_max = max(all_y)+1.5 if all_y else 30
    fig.update_layout(
        title=dict(text="<b>VIX Futures Term Structure</b>",
                   font=dict(size=15,color='#C9D1D9',family='Inter'),x=0.5),
        template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#161B22',
        height=420, margin=dict(l=50,r=30,t=65,b=50),
        xaxis=dict(tickvals=xpos,ticktext=labels,
                   tickfont=dict(size=11,color='#8B949E',family='JetBrains Mono'),
                   gridcolor='#21262D',
                   title=dict(text="Future Month",font=dict(size=11,color='#8B949E',family='Inter'))),
        yaxis=dict(range=[y_min,y_max],
                   title=dict(text="Volatility",font=dict(size=11,color='#8B949E',family='Inter')),
                   tickfont=dict(size=11,color='#8B949E',family='JetBrains Mono'),
                   gridcolor='#21262D'),
        legend=dict(orientation='v',yanchor='top',y=0.99,xanchor='right',x=0.99,
                    bgcolor='rgba(22,27,34,0.9)',bordercolor='#30363D',borderwidth=1,
                    font=dict(size=10,color='#C9D1D9',family='JetBrains Mono')),
        hovermode='x unified', dragmode=False)
    return fig

def build_bb_chart(bt, window=150):
    p = bt.tail(window).copy()
    sig = p['sig_final']
    fig = go.Figure()
    for i in range(1, len(p)):
        clr = 'rgba(63,185,80,0.07)' if sig.iloc[i]==1 else 'rgba(248,81,73,0.03)'
        fig.add_vrect(x0=p.index[i-1],x1=p.index[i],fillcolor=clr,layer="below",line_width=0)
    fig.add_trace(go.Scatter(x=p.index,y=p['BB_Upper'],mode='lines',name='BB 2σ',
        line=dict(color='#F85149',width=1.2)))
    fig.add_trace(go.Scatter(x=p.index,y=p['BB_Lower'],mode='lines',name='BB Lower',
        line=dict(color='#F85149',width=0.5),fill='tonexty',
        fillcolor='rgba(88,166,255,0.03)',showlegend=False))
    fig.add_trace(go.Scatter(x=p.index,y=p['BB_SMA20'],mode='lines',name='SMA(20)',
        line=dict(color='#58A6FF',width=1.5,dash='dash')))
    fig.add_trace(go.Scatter(x=p.index,y=p['VXX_Close'],mode='lines',name='VXX',
        line=dict(color='#F0F6FC',width=2)))
    for i in range(1,len(p)):
        if sig.iloc[i]==1 and sig.iloc[i-1]==0:
            fig.add_annotation(x=p.index[i],y=p['VXX_Close'].iloc[i],
                text="▲",showarrow=False,font=dict(size=14,color="#3FB950"),yshift=-18)
        elif sig.iloc[i]==0 and sig.iloc[i-1]==1:
            fig.add_annotation(x=p.index[i],y=p['VXX_Close'].iloc[i],
                text="▼",showarrow=False,font=dict(size=14,color="#F85149"),yshift=18)
    fig.add_trace(go.Scatter(x=[p.index[-1]],y=[p['VXX_Close'].iloc[-1]],mode='markers',
        name='Hoy',marker=dict(size=12,color='#D29922',line=dict(width=2,color='white')),showlegend=False))
    fig.update_layout(
        title=dict(text="<b>VXX + Bollinger Bands</b><sup>  ▲Entrada  ▼Salida  · Verde=LONG · Rojo=CASH</sup>",
                   font=dict(size=13,color='#C9D1D9',family='Inter'),x=0.5),
        template='plotly_dark',paper_bgcolor='#0D1117',plot_bgcolor='#161B22',
        height=420,margin=dict(l=50,r=30,t=55,b=40),
        xaxis=dict(gridcolor='#21262D',tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono'),
                   rangeselector=dict(
                       buttons=[dict(count=3,label="3M",step="month",stepmode="backward"),
                                dict(count=6,label="6M",step="month",stepmode="backward"),
                                dict(count=1,label="1A",step="year",stepmode="backward"),
                                dict(step="all",label="Todo")],
                       bgcolor='#161B22',activecolor='#F7931A',
                       font=dict(size=9,color='#C9D1D9',family='JetBrains Mono'))),
        yaxis=dict(title=dict(text="VXX",font=dict(size=11,color='#8B949E')),
                   gridcolor='#21262D',tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono')),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9,color='#8B949E',family='JetBrains Mono')),
        hovermode='x unified', dragmode=False)
    return fig

def build_price_chart(bt, col, label, color, trades_df):
    p = bt[bt[col].notna()].copy()
    sig = p['sig_final']
    fig = go.Figure()
    for i in range(1,len(p)):
        clr = 'rgba(63,185,80,0.07)' if sig.iloc[i]==1 else 'rgba(248,81,73,0.025)'
        fig.add_vrect(x0=p.index[i-1],x1=p.index[i],fillcolor=clr,layer="below",line_width=0)
    fig.add_trace(go.Scatter(x=p.index,y=p[col],mode='lines',name=label,
        line=dict(color=color,width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>'))
    closed = trades_df[trades_df['Salida'] != '🔴 ABIERTO']
    open_t = trades_df[trades_df['Salida'] == '🔴 ABIERTO']
    for _, t in closed.iterrows():
        for d_str, sym, yshift, clr in [
            (t['Entrada'],"▲",-18,"#3FB950"),
            (t['Salida'], "▼", 18,"#F85149")
        ]:
            d = pd.Timestamp(d_str)
            idx = p.index[p.index>=d]
            if len(idx)==0: continue
            fig.add_annotation(x=idx[0],y=p.loc[idx[0],col],text=sym,
                showarrow=False,font=dict(size=13,color=clr),yshift=yshift)
    for _, t in open_t.iterrows():
        d = pd.Timestamp(t['Entrada'])
        idx = p.index[p.index>=d]
        if len(idx)==0: continue
        fig.add_annotation(x=idx[0],y=p.loc[idx[0],col],text="▲ OPEN",
            showarrow=True,arrowhead=2,arrowcolor="#3FB950",
            font=dict(size=9,color="#3FB950",family="JetBrains Mono"),ax=0,ay=30)
    fig.update_layout(
        title=dict(text=f"<b>{label} — Histórico Operativo</b><sup>  ▲Entrada  ▼Salida</sup>",
                   font=dict(size=13,color='#C9D1D9',family='Inter'),x=0.5),
        template='plotly_dark',paper_bgcolor='#0D1117',plot_bgcolor='#161B22',
        height=380,margin=dict(l=55,r=30,t=55,b=40),
        xaxis=dict(gridcolor='#21262D',
                   tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono'),
                   rangeselector=dict(
                       buttons=[dict(count=3,label="3M",step="month",stepmode="backward"),
                                dict(count=6,label="6M",step="month",stepmode="backward"),
                                dict(count=1,label="1A",step="year",stepmode="backward"),
                                dict(step="all",label="Todo")],
                       bgcolor='#161B22',activecolor='#F7931A',
                       font=dict(size=9,color='#C9D1D9',family='JetBrains Mono'))),
        yaxis=dict(title=dict(text=f"{label} ($)",font=dict(size=11,color='#8B949E')),
                   gridcolor='#21262D',tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono')),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9,color='#8B949E',family='JetBrains Mono')),
        hovermode='x unified', dragmode=False)
    return fig

def build_equity_chart(equity):
    peak = equity.cummax()
    dd   = (equity-peak)/peak*100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index,y=dd,mode='lines',name='Drawdown %',
        line=dict(color='#F85149',width=1),fill='tozeroy',fillcolor='rgba(248,81,73,0.12)',yaxis='y2'))
    fig.add_trace(go.Scatter(x=equity.index,y=equity,mode='lines',name='Equity ($1)',
        line=dict(color='#3FB950',width=2.5)))
    fig.update_layout(
        title=dict(text="<b>Equity Curve — SVXY BB(2σ) + Contango</b>",
                   font=dict(size=13,color='#C9D1D9',family='Inter'),x=0.5),
        template='plotly_dark',paper_bgcolor='#0D1117',plot_bgcolor='#161B22',
        height=320,margin=dict(l=50,r=60,t=50,b=40),
        xaxis=dict(gridcolor='#21262D',tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono')),
        yaxis=dict(title='Equity ($)',gridcolor='#21262D',
                   tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono')),
        yaxis2=dict(title='DD %',overlaying='y',side='right',
                    tickfont=dict(size=9,color='#F85149',family='JetBrains Mono'),showgrid=False),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,bgcolor='rgba(0,0,0,0)',
                    font=dict(size=9,color='#8B949E',family='JetBrains Mono')),
        hovermode='x unified', dragmode=False)
    return fig

def build_yearly_chart(yearly):
    years = sorted(yearly.keys())
    vals  = [yearly[y] for y in years]
    fig   = go.Figure(go.Bar(
        x=[str(y) for y in years], y=vals,
        marker_color=['#3FB950' if v>=0 else '#F85149' for v in vals],
        text=[f"{v:+.1f}%" for v in vals], textposition='outside',
        textfont=dict(size=10,family='JetBrains Mono',color='#C9D1D9')))
    fig.update_layout(
        title=dict(text="<b>Retorno Anual</b>",
                   font=dict(size=13,color='#C9D1D9',family='Inter'),x=0.5),
        template='plotly_dark',paper_bgcolor='#0D1117',plot_bgcolor='#161B22',
        height=280,margin=dict(l=40,r=20,t=50,b=30),
        xaxis=dict(tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono')),
        yaxis=dict(tickfont=dict(size=10,color='#8B949E',family='JetBrains Mono'),
                   gridcolor='#21262D',zeroline=True,zerolinecolor='#30363D'),
        showlegend=False, dragmode=False)
    return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-REFRESH (solo datos live, NO recarga el CSV)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFRESH_INTERVAL = 60
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="autorefresh")
except ImportError:
    pass

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
if elapsed > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    scrape_cboe_futures.clear()
    fetch_vix_spot.clear()
    fetch_etps.clear()
    st.rerun()

# JS countdown visual
st.components.v1.html(f"""
<script>
(function(){{
    var r={REFRESH_INTERVAL};
    setInterval(function(){{
        r--;
        var el=window.parent.document.getElementById('refresh-countdown');
        if(el) el.textContent=r+'s';
        if(r<=0) r={REFRESH_INTERVAL};
    }},1000);
}})();
</script>
""", height=0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER + SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
<div class="hdr">
    <div class="logo">VIX CONTROLLER</div>
    <div class="sub">{now_str} · Refresh in <span id="refresh-countdown">{REFRESH_INTERVAL}s</span> · CBOE Delayed</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    N_MONTHS   = st.slider("Max futures months", 4, 12, 8)
    SHOW_PREV  = st.checkbox("Show previous day", True)
    SHOW_TABLE = st.checkbox("Show data table", True)
    if st.button("🔄 Refresh CBOE/yfinance"):
        scrape_cboe_futures.clear()
        fetch_vix_spot.clear()
        fetch_etps.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()
    if st.button("🗄️ Recargar CSV Drive"):
        load_and_process_strategy.clear()
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FETCH DATOS TAB 1 (bloqueante pero cacheado 55s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.spinner("🌐 Obteniendo datos CBOE…"):
    df_vx    = scrape_cboe_futures()
vix_spot     = fetch_vix_spot()
etps         = fetch_etps()

if not df_vx.empty and len(df_vx) > N_MONTHS:
    df_vx = df_vx.head(N_MONTHS).reset_index(drop=True)

m1p = df_vx['Price'].iloc[0] if not df_vx.empty and pd.notna(df_vx['Price'].iloc[0]) else None
m2p = df_vx['Price'].iloc[1] if len(df_vx)>1 and pd.notna(df_vx['Price'].iloc[1]) else None
front_ct = cpct(m1p, m2p)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab1, tab2, tab3 = st.tabs(["📈  Term Structure", "🎯  Monitor Operativo", "ℹ️  Help"])

# ━━━━━━━━━━━━━━━━━ TAB 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    vix_p = vix_spot['price'] if vix_spot else None
    last_price_col = df_vx['Price'].tolist() if not df_vx.empty else []
    total_ct = cpct(vix_p, last_price_col[-1]) if vix_p and last_price_col else None
    spot_m1  = cpct(vix_p, m1p)
    m1_lbl, m1_dte, m2_lbl = "", "?", ""
    if not df_vx.empty:
        exp1 = df_vx['Expiration'].iloc[0]
        if pd.notna(exp1):
            m1_lbl = MN.get(exp1.month,"")
            m1_dte = df_vx['DTE'].iloc[0] if 'DTE' in df_vx.columns else "?"
        if len(df_vx)>1:
            exp2 = df_vx['Expiration'].iloc[1]
            if pd.notna(exp2): m2_lbl = MN.get(exp2.month,"")
    st.markdown(f"""<div class="mrow">
        <div class="mpill"><div class="ml">VIX Index</div><div class="mv nt">{fv(vix_p)}</div></div>
        <div class="mpill"><div class="ml">M1 · {m1_lbl} · {m1_dte} DTE</div><div class="mv nt">{fv(m1p)}</div></div>
        <div class="mpill"><div class="ml">M2 · {m2_lbl}</div><div class="mv nt">{fv(m2p)}</div></div>
        <div class="mpill"><div class="ml">VIX → M1</div><div class="mv {vc(spot_m1)}">{fp(spot_m1)}</div></div>
        <div class="mpill"><div class="ml">M1 → M2 Contango</div><div class="mv {vc(front_ct)}">{fp(front_ct)}</div></div>
        <div class="mpill"><div class="ml">Total Curve</div><div class="mv {vc(total_ct)}">{fp(total_ct)}</div></div>
    </div>""", unsafe_allow_html=True)
    fig = build_term_chart(vix_spot, df_vx, show_prev=SHOW_PREV)
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=True,displaylogo=False,scrollZoom=False))
    # Contango table
    if len(df_vx)>=2:
        ct_cells, diff_cells = "", ""
        for i in range(len(df_vx)-1):
            p1, p2 = df_vx['Price'].iloc[i], df_vx['Price'].iloc[i+1]
            ct   = cpct(p1,p2)
            diff = round(p2-p1,2) if pd.notna(p1) and pd.notna(p2) and p1>0 and p2>0 else None
            ct_cells   += f'<td>{i+1}</td><td class="{"pos" if ct and ct>=0 else "neg"}">{fp(ct)}</td>'
            diff_cells += f'<td>{i+1}</td><td class="{"pos" if diff and diff>=0 else "neg"}">{fv(diff)}</td>'
        st.markdown(f"""<table class="ctx">
            <tr><td class="hdr-cell">% Contango</td>{ct_cells}</tr>
            <tr><td class="hdr-cell">Difference</td>{diff_cells}</tr>
        </table>""", unsafe_allow_html=True)
        if len(df_vx)>=7:
            p4,p7 = df_vx['Price'].iloc[3], df_vx['Price'].iloc[6]
            if pd.notna(p4) and pd.notna(p7) and p4>0 and p7>0:
                m74_ct  = cpct(p4,p7)
                m74_cls = "pos" if m74_ct and m74_ct>=0 else "neg"
                st.markdown(f"""<table class="ctx" style="width:auto;margin-top:4px">
                    <tr><td class="hdr-cell">Month 7 to 4</td>
                    <td class="{m74_cls}">{fp(m74_ct)}</td>
                    <td class="{m74_cls}">{fv(round(p7-p4,2))}</td></tr>
                </table>""", unsafe_allow_html=True)
    # Data table
    if SHOW_TABLE and not df_vx.empty:
        rows = ""
        prev_p = vix_p
        for _, r in df_vx.iterrows():
            sym   = r.get('Symbol','')
            exp   = r.get('Expiration')
            exp_s = exp.strftime('%m/%d/%Y') if pd.notna(exp) else "—"
            last  = r.get('Last',0); chg = r.get('Change',0)
            hi    = r.get('High',0); lo  = r.get('Low',0)
            settle= r.get('Settlement',0); vol = r.get('Volume',0)
            price = r.get('Price',0); dte = r.get('DTE','')
            ct    = cpct(prev_p,price) if prev_p and pd.notna(price) and price>0 else None
            rows += f"""<tr>
                <td style="color:var(--b);font-weight:600">{sym}</td><td>{exp_s}</td>
                <td style="font-weight:600">{"—" if not (pd.notna(last) and last>0) else f"{last:.2f}"}</td>
                <td style="color:{'var(--g)' if pd.notna(chg) and chg>0 else 'var(--r)' if pd.notna(chg) and chg<0 else ''}">{f"{chg:+.3f}" if pd.notna(chg) and chg!=0 else "—"}</td>
                <td>{"—" if not (pd.notna(hi) and hi>0) else f"{hi:.2f}"}</td>
                <td>{"—" if not (pd.notna(lo) and lo>0) else f"{lo:.2f}"}</td>
                <td>{"—" if not (pd.notna(settle) and settle>0) else f"{settle:.4f}"}</td>
                <td style="color:{'var(--g)' if ct and ct>=0 else 'var(--r)' if ct else ''}">{fp(ct) if ct else "—"}</td>
                <td>{dte}</td>
                <td>{"0" if not (pd.notna(vol) and vol>0) else f"{int(vol):,}"}</td>
            </tr>"""
            if pd.notna(price) and price>0: prev_p = price
        st.markdown(f"""<table class="dtbl">
            <thead><tr><th>Symbol</th><th>Expiration</th><th>Last</th><th>Change</th>
            <th>High</th><th>Low</th><th>Settlement</th><th>Contango</th><th>DTE</th><th>Volume</th></tr></thead>
            <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
    if df_vx.empty:
        st.warning("⚠️ No se pudieron obtener datos del CBOE.")
        if not pw_ready:
            st.error("❌ Playwright/Chromium no disponible.")
    else:
        scraped = df_vx['Scraped_At'].iloc[0] if 'Scraped_At' in df_vx.columns else "?"
        st.caption(f"Contratos: {len(df_vx)} mensuales · {scraped}")

# ━━━━━━━━━━━━━━━━━ TAB 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    with st.spinner("📂 Cargando histórico desde Google Drive…"):
        bt, trades_df, metrics = load_and_process_strategy()

    if bt is None or (isinstance(bt, pd.DataFrame) and bt.empty):
        st.error("❌ No se pudo cargar el CSV. Verifica que el archivo sea público en Google Drive.")
        st.stop()

    today_etps = fetch_etps()
    last_hist  = bt.iloc[-1]
    last_date  = bt.index[-1]

    # Señal de hoy
    vxx_today   = today_etps.get('VXX',{}).get('close', float(last_hist['VXX_Close']))
    sma20_today = float(last_hist['BB_SMA20'])
    bb_up_today = float(last_hist['BB_Upper'])
    bb_pos_hist = int(last_hist['sig_bb'])
    if bb_pos_hist == 0 and vxx_today < sma20_today:   bb_sig_today = 1
    elif bb_pos_hist == 1 and vxx_today > bb_up_today: bb_sig_today = 0
    else:                                               bb_sig_today = bb_pos_hist

    # Contango live del CBOE (Tab 1)
    if m1p and m2p and m1p > 0:
        ct_today  = cpct(m1p, m2p)
        ct_source = "CBOE live"
        m1_sym    = df_vx['Symbol'].iloc[0] if not df_vx.empty else "M1"
        m2_sym    = df_vx['Symbol'].iloc[1] if len(df_vx)>1 else "M2"
    else:
        ct_today  = float(last_hist.get('Contango_pct', 0))
        ct_source = "CSV"
        m1_sym    = str(last_hist.get('M1_Symbol','M1'))
        m2_sym    = str(last_hist.get('M2_Symbol','M2'))

    in_ct_today     = ct_today is not None and ct_today > 0
    final_sig_today = int(bb_sig_today == 1 and in_ct_today)

    exec_date = datetime.now().date() + timedelta(days=1)
    while exec_date.weekday() >= 5: exec_date += timedelta(days=1)

    pct_to_sma = (vxx_today/sma20_today-1)*100 if sma20_today else 0
    pct_to_bb  = (vxx_today/bb_up_today -1)*100 if bb_up_today else 0
    ct_str     = f"{ct_today:+.2f}%" if ct_today is not None else "N/A"
    vix_today  = today_etps.get('VIX',{}).get('close', float(last_hist.get('VIX_Close',0)))
    svxy_today = today_etps.get('SVXY',{}).get('close', 0)
    svix_today = today_etps.get('SVIX',{}).get('close', 0)

    if vix_today < 15:   regime, r_clr = "BAJO — óptimo",       "var(--g)"
    elif vix_today < 20: regime, r_clr = "NORMAL — bueno",      "var(--g)"
    elif vix_today < 28: regime, r_clr = "ELEVADO — precaución","var(--y)"
    else:                regime, r_clr = "CRISIS — peligro",    "var(--r)"

    # ── Sección 1: Señal de hoy ──
    sig_cls = "sig-long" if final_sig_today else "sig-cash"
    sig_txt = "LONG SVXY" if final_sig_today else "CASH"
    sig_clr = "var(--g)" if final_sig_today else "var(--r)"
    bb_ok   = "ok" if bb_sig_today else "no"
    ct_ok   = "ok" if in_ct_today else "no"

    c1, c2, c3, c4 = st.columns([1.3, 1.5, 1.5, 1.3])
    with c1:
        st.markdown(f"""<div class="sig-box {sig_cls}">
            <div class="sl" style="color:{sig_clr}">{sig_txt}</div>
            <div class="sd">Ejecutar {exec_date.strftime('%Y-%m-%d')} al OPEN</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        sma_clr = "var(--g)" if vxx_today < sma20_today else "var(--r)"
        bb_clr  = "var(--g)" if vxx_today <= bb_up_today else "var(--r)"
        st.markdown(f"""<div class="icard">
            <div class="ic-title">📊 BB Timing — VXX</div>
            <div class="ic-row"><span class="ic-label">Señal BB</span>
                <span class="ic-val"><span class="{bb_ok}">{"✓" if bb_sig_today else "✗"}</span> {"LONG" if bb_sig_today else "CASH"}</span></div>
            <div class="ic-row"><span class="ic-label">VXX</span><span class="ic-val" style="font-weight:700">${vxx_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">SMA(20)</span><span class="ic-val" style="color:{sma_clr}">${sma20_today:.2f} ({pct_to_sma:+.1f}%)</span></div>
            <div class="ic-row"><span class="ic-label">BB 2σ</span><span class="ic-val" style="color:{bb_clr}">${bb_up_today:.2f} ({pct_to_bb:+.1f}%)</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        ct_clr   = "var(--g)" if in_ct_today else "var(--r)"
        ct_estado = "CONTANGO" if in_ct_today else "BACKWARDATION"
        st.markdown(f"""<div class="icard">
            <div class="ic-title">📈 Contango ({ct_source})</div>
            <div class="ic-row"><span class="ic-label">Señal CT</span>
                <span class="ic-val"><span class="{ct_ok}">{"✓" if in_ct_today else "✗"}</span>
                <span style="color:{ct_clr};font-weight:700"> {ct_estado}</span></span></div>
            <div class="ic-row"><span class="ic-label">{m1_sym}</span><span class="ic-val">${m1p:.2f if m1p else "—"}</span></div>
            <div class="ic-row"><span class="ic-label">{m2_sym}</span><span class="ic-val">${m2p:.2f if m2p else "—"}</span></div>
            <div class="ic-row"><span class="ic-label">Contango %</span><span class="ic-val" style="color:{ct_clr};font-weight:700">{ct_str}</span></div>
            <div class="ic-row"><span class="ic-label">VIX</span><span class="ic-val" style="color:{r_clr}">{vix_today:.1f} · {regime}</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="icard">
            <div class="ic-title">💼 Vehículos</div>
            <div class="ic-row"><span class="ic-label">SVXY (-0.5x)</span><span class="ic-val" style="color:var(--c);font-weight:700">${svxy_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">SVIX (-1x)</span><span class="ic-val" style="color:var(--c)">${svix_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">VIX Spot</span><span class="ic-val">{vix_today:.2f}</span></div>
            <div class="ic-row"><span class="ic-label">CSV al</span><span class="ic-val" style="color:var(--dim)">{last_date.strftime('%Y-%m-%d')}</span></div>
        </div>""", unsafe_allow_html=True)

    # Alertas
    if final_sig_today and pct_to_bb > -3:
        st.warning(f"⚠️ VXX a {abs(pct_to_bb):.1f}% de la BB Superior — posible salida pronto")
    if ct_today is not None and 0 < ct_today < 1:
        st.warning(f"⚠️ Contango muy bajo ({ct_today:.2f}%) — monitorear")
    if not final_sig_today and abs(pct_to_sma) < 2 and in_ct_today:
        st.info(f"🔔 Posible entrada pronto — VXX a {abs(pct_to_sma):.1f}% de SMA(20)")

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Sección 2: Métricas ──
    if metrics:
        m = metrics
        st.markdown(f"""<div class="mrow">
            {mcard("CAGR", f"{m['cagr']:+.1f}%", "up" if m['cagr']>0 else "dn")}
            {mcard("Max DD", f"{m['mdd']:.1f}%", "dn")}
            {mcard("Sharpe", f"{m['sharpe']:.2f}", "up" if m['sharpe']>1 else "nt")}
            {mcard("Calmar", f"{m['calmar']:.2f}", "up" if m['calmar']>1 else "nt")}
            {mcard("Win Rate", f"{m['wr']:.1f}%", "nt")}
            {mcard("Exposición", f"{m['exp']:.1f}%", "nt")}
        </div>""", unsafe_allow_html=True)
        col_eq, col_yr = st.columns([2, 1])
        with col_eq:
            st.plotly_chart(build_equity_chart(m['equity']), use_container_width=True,
                config=dict(displayModeBar=False, displaylogo=False, scrollZoom=False))
        with col_yr:
            st.plotly_chart(build_yearly_chart(m['yearly']), use_container_width=True,
                config=dict(displayModeBar=False, displaylogo=False, scrollZoom=False))

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Sección 3: Gráficas operativas ──
    st.markdown("<span style='font-family:Inter;font-weight:700;font-size:0.9rem;color:#F0F6FC'>📊 Gráficas Operativas</span>", unsafe_allow_html=True)

    st.caption("SEÑAL DE TIMING · VXX vs BB(20, 2σ)")
    st.plotly_chart(build_bb_chart(bt, window=len(bt)), use_container_width=True,
        config=dict(displayModeBar=True, displaylogo=False, scrollZoom=False,
                    modeBarButtonsToRemove=['select2d','lasso2d']))

    if 'SVIX_Close' in bt.columns and bt['SVIX_Close'].notna().sum() > 10:
        st.caption("VEHÍCULO AGRESIVO · SVIX (-1x)")
        st.plotly_chart(build_price_chart(bt,'SVIX_Close','SVIX','#E91E63',trades_df),
            use_container_width=True,
            config=dict(displayModeBar=True, displaylogo=False, scrollZoom=False,
                        modeBarButtonsToRemove=['select2d','lasso2d']))

    st.caption("VEHÍCULO PRINCIPAL · SVXY (-0.5x)")
    st.plotly_chart(build_price_chart(bt,'SVXY_Close','SVXY','#39D2C0',trades_df),
        use_container_width=True,
        config=dict(displayModeBar=True, displaylogo=False, scrollZoom=False,
                    modeBarButtonsToRemove=['select2d','lasso2d']))

    st.markdown("<div style='border-top:1px solid #30363D;margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Sección 4: Historial de operaciones ──
    st.markdown("<span style='font-family:Inter;font-weight:700;font-size:0.9rem;color:#F0F6FC'>📋 Historial de Operaciones</span>", unsafe_allow_html=True)
    if not trades_df.empty:
        closed   = trades_df[trades_df['Salida'] != '🔴 ABIERTO']
        n_win    = (closed['Retorno'] > 0).sum()
        avg_ret  = closed['Retorno'].mean()
        avg_dur  = closed['Días'].mean()
        st.markdown(f"""<div class="mrow">
            {mcard("Trades", str(len(trades_df)), "nt")}
            {mcard("Ganadores", f"{n_win}/{len(closed)}", "up")}
            {mcard("Win Rate", f"{n_win/len(closed)*100:.1f}%" if len(closed)>0 else "—", "nt")}
            {mcard("Ret. promedio", f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",
                   "up" if not pd.isna(avg_ret) and avg_ret>0 else "dn")}
            {mcard("Duración media", f"{avg_dur:.0f}d" if not pd.isna(avg_dur) else "—", "nt")}
        </div>""", unsafe_allow_html=True)
        rows_html = ""
        for _, t in trades_df.iloc[::-1].iterrows():
            ret     = t['Retorno']
            is_open = t['Salida'] == '🔴 ABIERTO'
            ret_clr = "color:var(--g)" if ret>0 else "color:var(--r)"
            sal_clr = "color:var(--y);font-weight:700" if is_open else ""
            vix_e   = f"{t['VIX']:.1f}" if pd.notna(t.get('VIX')) else "—"
            ct_e    = f"{t['Contango']:+.2f}%" if pd.notna(t.get('Contango')) else "—"
            rows_html += f"""<tr>
                <td style="color:var(--b);font-weight:600">{t['Entrada']}</td>
                <td style="{sal_clr}">{t['Salida']}</td>
                <td>{t['Días']}</td>
                <td style="{ret_clr};font-weight:700">{ret:+.2f}%</td>
                <td>{t['Razón']}</td><td>{vix_e}</td><td>{ct_e}</td>
            </tr>"""
        st.markdown(f"""
        <div style="max-height:400px;overflow-y:auto;border:1px solid #30363D;border-radius:4px">
        <table class="dtbl">
            <thead style="position:sticky;top:0;background:#1C2128;z-index:1"><tr>
                <th>Entrada</th><th>Salida</th><th>Días</th><th>Retorno</th>
                <th>Razón salida</th><th>VIX</th><th>Contango</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>""", unsafe_allow_html=True)
        st.caption(f"Datos: {bt.index[0].strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} · Contango: {ct_source}")

# ━━━━━━━━━━━━━━━━━ TAB 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("""
    ### VIX Controller — Guía
    **Tab 1: Term Structure** — Datos CBOE Delayed Quotes (Playwright). Auto-refresh 60s.
    **Tab 2: Monitor Operativo** — CSV histórico de Google Drive (TTL 5 min) + contango live del CBOE.
    - Señal: BB(20, 2σ) sobre VXX × Contango Rule (M2 > M1)
    - Entrada al OPEN del día siguiente a la señal
    - Vehículo: SVXY (-0.5x)
    **Sidebar:** 🔄 refresca solo CBOE/yfinance · 🗄️ descarga nuevo CSV del Drive
    """)

st.markdown("""
<div style="text-align:center;padding:0.8rem 0 0.3rem;border-top:1px solid #30363D;margin-top:1rem">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#484F58">
        VIX CONTROLLER · Alberto Alarcón González · Not financial advice
    </span>
</div>""", unsafe_allow_html=True)
