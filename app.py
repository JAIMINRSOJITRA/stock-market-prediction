# ================================================================
# NSE Stock AI — Streamlit Prediction App
# Dashboard, Inference Engine, and Insights (No Training Code)
# ================================================================

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import ta
import warnings
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Required: lets joblib unpickle ENSEMBLE_*.pkl ─────────────────
class PreFittedVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = np.array([est.predict_proba(X) for _, est in self.estimators])
        return probas.mean(axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Config ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "category_models")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "plots")

FEATURES = [
    "SMA_Ratio_10", "SMA_Ratio_50", "RSI", "MACD_Pct", "MACD_Sig_Pct",
    "EMA_Ratio_20", "ATR_Pct", "Momentum", "Volume_Ratio", "BB_Width",
]

STOCKS = {
    # Large Cap
    "RELIANCE.NS": "Large Cap",  "HDFCBANK.NS": "Large Cap",
    "INFY.NS":     "Large Cap",  "TCS.NS":      "Large Cap",
    "ICICIBANK.NS":"Large Cap",  "KOTAKBANK.NS":"Large Cap",
    "LT.NS":       "Large Cap",  "AXISBANK.NS": "Large Cap",
    "HINDUNILVR.NS":"Large Cap", "ITC.NS":      "Large Cap",
    "SBIN.NS":     "Large Cap",  "BAJFINANCE.NS":"Large Cap",
    "WIPRO.NS":    "Large Cap",  "HCLTECH.NS":  "Large Cap",
    "MARUTI.NS":   "Large Cap",  "SUNPHARMA.NS":"Large Cap",
    "TITAN.NS":    "Large Cap",  "ASIANPAINT.NS":"Large Cap",
    "BHARTIARTL.NS":"Large Cap", "ULTRACEMCO.NS":"Large Cap",
    # Mid Cap
    "MUTHOOTFIN.NS":"Mid Cap",   "VOLTAS.NS":   "Mid Cap",
    "PERSISTENT.NS":"Mid Cap",   "COFORGE.NS":  "Mid Cap",
    "INDHOTEL.NS":  "Mid Cap",   "TATACHEM.NS": "Mid Cap",
    "CROMPTON.NS":  "Mid Cap",   "MAXHEALTH.NS":"Mid Cap",
    "PIIND.NS":     "Mid Cap",   "GODREJPROP.NS":"Mid Cap",
    "LUPIN.NS":     "Mid Cap",   "TATAELXSI.NS":"Mid Cap",
    "BALKRISIND.NS":"Mid Cap",   "OBEROIRLTY.NS":"Mid Cap",
    "JUBLFOOD.NS":  "Mid Cap",   "HONAUT.NS":   "Mid Cap",
    "IPCALAB.NS":   "Mid Cap",   "ALKEM.NS":    "Mid Cap",
    "SUNDARMFIN.NS":"Mid Cap",   "CARBORUNIV.NS":"Mid Cap",
    # Small Cap
    "NAZARA.NS":    "Small Cap", "KPITTECH.NS":  "Small Cap",
    "HAPPSTMNDS.NS":"Small Cap", "LATENTVIEW.NS":"Small Cap",
    "SAPPHIRE.NS":  "Small Cap", "MEDPLUS.NS":   "Small Cap",
    "SYRMA.NS":     "Small Cap", "FUSION.NS":    "Small Cap",
    "KAYNES.NS":    "Small Cap", "SENCO.NS":     "Small Cap",
    "BIKAJI.NS":    "Small Cap", "CRAFTSMAN.NS": "Small Cap",
    "RAILTEL.NS":   "Small Cap", "FINEORG.NS":   "Small Cap",
    "DODLA.NS":     "Small Cap", "LXCHEM.NS":    "Small Cap",
    "RATEGAIN.NS":  "Small Cap", "CAMPUS.NS":    "Small Cap",
    "HARSHA.NS":    "Small Cap", "SBCL.NS":      "Small Cap",
}

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]

MODEL_LABELS = {
    "ENSEMBLE": "🏆 Ensemble (Best)",
    "XGB":      "⚡ XGBoost",
    "LGBM":     "💡 LightGBM",
    "RF":       "🌲 Random Forest",
    "LR":       "📐 Logistic Regression",
    "SVM":      "🔬 SVM",
}

# ── Page setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock AI – Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, .stApp {
    background: #07101f !important;
    font-family: 'Inter', sans-serif;
    color: #e2eaff;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1628 0%, #0e1f3a 100%) !important;
    border-right: 1px solid rgba(79,142,247,0.15);
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }
[data-testid="stSidebar"] label { color: #6a84b5 !important; font-size: 12px !important; }

/* ── Selectbox & Radio ── */
.stSelectbox > div > div, .stRadio > div {
    background: #0f1e36 !important;
    border: 1px solid rgba(79,142,247,0.25) !important;
    border-radius: 10px !important;
    color: #e2eaff !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f8ef7 0%, #7c5bf2 100%);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 14px 0;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    box-shadow: 0 0 20px rgba(79,142,247,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 36px rgba(79,142,247,0.55);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f1e36;
    border: 1px solid rgba(79,142,247,0.18);
    border-radius: 14px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #6a84b5 !important; font-size: 11px !important; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #e2eaff !important; font-weight: 700 !important; }

/* ── Signal card ── */
.signal-card {
    border-radius: 20px;
    padding: 36px 24px;
    text-align: center;
    margin: 24px 0;
    animation: fadeIn 0.4s ease;
}
.signal-buy  { background: linear-gradient(135deg,#0d2b1e,#0f3329); border: 2px solid #22d3a5; }
.signal-sell { background: linear-gradient(135deg,#2b0d10,#330f14); border: 2px solid #f7566a; }
.signal-label-buy  { color:#22d3a5; font-size:52px; font-weight:800; letter-spacing:4px; }
.signal-label-sell { color:#f7566a; font-size:52px; font-weight:800; letter-spacing:4px; }
.signal-sub { color:#7a8db5; font-size:13px; margin-top: 6px; }

/* ── Confidence bar ── */
.conf-wrap  { background:#0f1e36; border-radius:8px; height:8px; margin:14px 0 4px; }
.conf-fill-buy  { height:8px; border-radius:8px; background:linear-gradient(90deg,#4f8ef7,#22d3a5); }
.conf-fill-sell { height:8px; border-radius:8px; background:linear-gradient(90deg,#f7566a,#f7c948); }

/* ── Info pill ── */
.pill {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:12px; font-weight:600; margin:2px;
}
.pill-blue   { background:#4f8ef720; color:#4f8ef7; border:1px solid #4f8ef740; }
.pill-green  { background:#22d3a520; color:#22d3a5; border:1px solid #22d3a540; }
.pill-red    { background:#f7566a20; color:#f7566a; border:1px solid #f7566a40; }
.pill-yellow { background:#f7c94820; color:#f7c948; border:1px solid #f7c94840; }

/* ── Divider & Spinner ── */
hr { border-color: rgba(79,142,247,0.12) !important; }
.stSpinner > div { border-top-color:#4f8ef7 !important; }
@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
</style>
""")


# ── Helper functions ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_data(symbol: str):
    """Download 200 days of data and compute all 10 technical features."""
    df = yf.download(symbol, period="200d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
        return None

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    macd_obj = ta.trend.MACD(close=close)
    bb_obj   = ta.volatility.BollingerBands(close=close, window=20)

    df["SMA_Ratio_10"] = close / ta.trend.sma_indicator(close, window=10)
    df["SMA_Ratio_50"] = close / ta.trend.sma_indicator(close, window=50)
    df["EMA_Ratio_20"] = close / ta.trend.ema_indicator(close, window=20)
    df["ATR_Pct"]      = ta.volatility.average_true_range(high, low, close) / close * 100
    df["Volume_Ratio"] = vol / vol.rolling(20).mean()
    df["MACD_Pct"]     = macd_obj.macd()   / close * 100
    df["MACD_Sig_Pct"] = macd_obj.macd_signal() / close * 100
    df["RSI"]          = ta.momentum.rsi(close, window=14)
    df["Momentum"]     = close.pct_change(5)
    df["BB_Width"]     = bb_obj.bollinger_wband()

    df.dropna(inplace=True)
    return df if not df.empty else None

@st.cache_resource(show_spinner=False)
def load_model(model_key: str, category: str):
    """Load a pre-trained model pkl from category_models/."""
    cat_key = category.lower().replace(" ", "_")
    path    = os.path.join(MODEL_DIR, f"{model_key}_{cat_key}.pkl")
    if not os.path.exists(path):
        return None, None
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        return bundle["model"], bundle["scaler"]
    return bundle, None

def predict(symbol: str, category: str, model_key: str):
    """Return prediction dict or None."""
    df = fetch_live_data(symbol)
    if df is None: return None
    model, scaler = load_model(model_key, category)
    if model is None: return None

    row = df[FEATURES].iloc[[-1]].copy()
    row_input = scaler.transform(row) if scaler is not None else row

    pred   = model.predict(row_input)[0]
    proba  = model.predict_proba(row_input)[0]
    signal = "BUY" if pred == 1 else "SELL"

    if len(proba) > 1:
        prob_buy  = float(proba[1]) * 100
        prob_sell = float(proba[0]) * 100
    else:
        is_buy_class = (len(model.classes_) > 0 and model.classes_[0] == 1)
        prob_buy  = float(proba[0]) * 100 if is_buy_class else 0.0
        prob_sell = 0.0 if is_buy_class else float(proba[0]) * 100

    return {
        "signal":     signal,
        "confidence": float(max(proba)) * 100,
        "prob_buy":   prob_buy,
        "prob_sell":  prob_sell,
        "price":      float(df["Close"].squeeze().iloc[-1]),
        "date":       df.index[-1].strftime("%d %b %Y"),
        "rsi":        float(df["RSI"].iloc[-1]),
        "momentum":   float(df["Momentum"].iloc[-1]) * 100,
        "volume_ratio": float(df["Volume_Ratio"].iloc[-1]),
        "df":         df,
    }

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div style="background:linear-gradient(135deg,#4f8ef7,#7c5bf2);
                border-radius:14px;padding:18px;text-align:center;margin-bottom:20px">
        <div style="font-size:28px">📈</div>
        <div style="color:white;font-size:17px;font-weight:800;margin:4px 0 2px">NSE Stock AI</div>
        <div style="color:rgba(255,255,255,0.7);font-size:11px">ML Pipeline · India</div>
    </div>
    """)

    page = st.radio("Navigation", ["🏠 Dashboard", "🔮 Live Predict", "📊 Model Insights"], label_visibility="collapsed")
    st.markdown("---")

    if page == "🔮 Live Predict":
        st.markdown("#### 1️⃣ Choose Category")
        category = st.selectbox("Market Cap", CATEGORIES, label_visibility="collapsed")
        
        st.markdown("#### 2️⃣ Choose Stock")
        stock_list = [s for s, c in STOCKS.items() if c == category]
        symbol     = st.selectbox("Stock", stock_list, label_visibility="collapsed")
        
        st.markdown("#### 3️⃣ Choose Model")
        model_key = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k], label_visibility="collapsed")
        st.markdown("---")

    st.html(f"""
    <div style="color:#4a6085;font-size:11px;line-height:1.8">
        📦 60 NSE stocks<br>
        🤖 6 ML models per category<br>
        🎯 5-day return &gt; 1% = BUY<br>
        📅 {datetime.now().strftime('%d %b %Y, %H:%M')}
    </div>
    """)

# ================================================================
# PAGE: DASHBOARD
# ================================================================
if page == "🏠 Dashboard":
    st.html("""
    <div style="background:linear-gradient(135deg,#0a1222,#142340);
                border:1px solid rgba(79,142,247,0.15);border-radius:18px;
                padding:40px;margin-bottom:30px">
        <h1 style="font-size:36px;font-weight:800;margin:0 0 10px;
                   background:linear-gradient(135deg,#ffffff,#4f8ef7,#22d3a5);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            NSE Stock Market AI
        </h1>
        <p style="color:#a8bce0;font-size:15px;margin:0">
            60 NSE Stocks · 6 ML Models · 3 Market Categories<br>
            Binary Classification: <b>Will this stock return > 1% in the next 5 days?</b>
        </p>
    </div>
    """)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📦 Total Stocks", "60", "NSE Listed")
    col2.metric("🏗️ Models Trained", "18", "6 per category")
    col3.metric("🔢 Features", "10", "Indicators")
    col4.metric("📅 Data Span", "7 Years", "2000+ days")
    col5.metric("🎯 Target", "5 Days", ">1% return")

    st.markdown("---")
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        st.markdown("#### 📊 Overall Model Accuracy")
        img_acc = os.path.join(PLOT_DIR, "eval_accuracy_grouped.png")
        if os.path.exists(img_acc): st.image(img_acc, use_container_width=True)
        else: st.info("Run `python stock_ml_v3.py` to generate the plots.")
    
    with c_right:
        st.markdown("#### 🎯 BUY vs SELL Distribution")
        img_dist = os.path.join(PLOT_DIR, "eda_target_distribution.png")
        if os.path.exists(img_dist): st.image(img_dist, use_container_width=True)

# ================================================================
# PAGE: LIVE PREDICT
# ================================================================
elif page == "🔮 Live Predict":
    st.html("""
    <h1 style="font-size:30px;font-weight:800;margin-bottom:2px;
               background:linear-gradient(135deg,#ffffff,#4f8ef7,#22d3a5);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Real-Time Signal Predictor
    </h1>
    <p style="color:#4a6085;font-size:13px;margin:0 0 24px">
        Select a stock from the sidebar → click <b style="color:#e2eaff">Predict</b>
    </p>
    """)

    st.html(f"""
    <div style="background:#0f1e36;border:1px solid rgba(79,142,247,0.2);
                border-radius:12px;padding:14px 18px;margin-bottom:20px;
                display:flex;align-items:center;gap:10px">
        <span style="font-size:20px">🔎</span>
        <span style="font-size:14px;color:#e2eaff;font-weight:600">{symbol}</span>
        <span class="pill pill-blue">{category}</span>
        <span class="pill pill-blue" style="margin-left:auto">{MODEL_LABELS[model_key]}</span>
    </div>
    """)

    predict_btn = st.button("🚀  Predict BUY / SELL")

    if predict_btn:
        with st.spinner(f"Fetching live data for {symbol}…"):
            result = predict(symbol, category, model_key)

        if result is None:
            st.error("❌  Could not load data or model.")
        else:
            signal = result["signal"]
            conf   = result["confidence"]
            price  = result["price"]
            is_buy = signal == "BUY"

            card_cls  = "signal-card signal-buy"  if is_buy else "signal-card signal-sell"
            label_cls = "signal-label-buy"        if is_buy else "signal-label-sell"
            icon      = "🟢" if is_buy else "🔴"
            fill_cls  = "conf-fill-buy"           if is_buy else "conf-fill-sell"

            st.html(f"""
            <div class="{card_cls}">
                <div style="color:#6a84b5;font-size:11px;letter-spacing:2px;margin-bottom:10px">AI SIGNAL — {symbol}</div>
                <div class="{label_cls}">{icon} &nbsp;{signal}</div>
                <div class="signal-sub">
                    As of <b style="color:#c8d8ff">{result['date']}</b> &nbsp;·&nbsp; Price: <b style="color:#f7c948">₹{price:,.2f}</b>
                </div>
                <div style="margin-top:18px">
                    <div style="color:#6a84b5;font-size:11px;letter-spacing:1px">MODEL CONFIDENCE</div>
                    <div class="conf-wrap">
                        <div class="{fill_cls}" style="width:{conf:.1f}%"></div>
                    </div>
                    <div style="color:#e2eaff;font-size:22px;font-weight:700">{conf:.1f}%</div>
                </div>
            </div>
            """)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📈 Price", f"₹{price:,.2f}")
            c2.metric("⚡ RSI", f"{result['rsi']:.1f}", "Overbought" if result["rsi"]>70 else ("Oversold" if result["rsi"]<30 else "Neutral"))
            c3.metric("🚀 Momentum", f"{result['momentum']:.2f}%")
            c4.metric("📦 Vol Ratio", f"{result['volume_ratio']:.2f}x")

            st.markdown("---")
            st.markdown("#### 📊 Signal Probability")
            prob_buy  = result["prob_buy"]
            prob_sell = result["prob_sell"]
            fig_prob = go.Figure()
            fig_prob.add_bar(x=[prob_buy], y=[""], orientation="h", name="BUY", marker_color="#22d3a5", text=[f"BUY {prob_buy:.1f}%"], textposition="inside")
            fig_prob.add_bar(x=[prob_sell], y=[""], orientation="h", name="SELL", marker_color="#f7566a", text=[f"SELL {prob_sell:.1f}%"], textposition="inside")
            fig_prob.update_layout(barmode="stack", height=80, template="plotly_dark", paper_bgcolor="#07101f", plot_bgcolor="#07101f", margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis=dict(range=[0,100], showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False))
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("#### 📈 Price Chart (Last 200 Days)")
            df_c  = result["df"]
            close = df_c["Close"].squeeze()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=df_c.index, y=close, name="Close", line=dict(color="#4f8ef7", width=2), fill="tozeroy", fillcolor="rgba(79,142,247,0.07)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=ta.trend.sma_indicator(close, window=10), name="SMA 10", line=dict(color="#f7c948", width=1.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=ta.trend.sma_indicator(close, window=50), name="SMA 50", line=dict(color="#22d3a5", width=1.5, dash="dash")), row=1, col=1)
            fig.add_trace(go.Bar(x=df_c.index, y=df_c["Volume"].squeeze(), name="Volume", marker_color="rgba(124,91,242,0.45)"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#07101f", plot_bgcolor="#07101f", margin=dict(l=0,r=0,t=10,b=0), legend=dict(bgcolor="rgba(11,22,40,0.8)", font=dict(size=11)), xaxis2=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis2=dict(gridcolor="rgba(255,255,255,0.05)", title="Volume"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📡 RSI Indicator")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_c.index, y=df_c["RSI"].squeeze(), line=dict(color="#a478f7", width=2), name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f7566a", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22d3a5", annotation_text="Oversold (30)")
            fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(79,142,247,0.04)", line_width=0)
            fig_rsi.update_layout(template="plotly_dark", height=200, paper_bgcolor="#07101f", plot_bgcolor="#07101f", margin=dict(l=0,r=0,t=10,b=0), showlegend=False, yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            
            st.html(f"""
            <div style="background:#0f1e36;border:1px solid rgba(79,142,247,0.15);border-radius:12px;padding:14px 18px;margin-top:8px;color:#4a6085;font-size:12px;line-height:1.8">
                ℹ️ &nbsp;Prediction target: <b style="color:#c8d8ff">5-day forward return &gt; 1%</b> &nbsp;·&nbsp; Model: <b style="color:#c8d8ff">{MODEL_LABELS[model_key]}</b><br>
                ⚠️ &nbsp;<span style="color:#f7c948">Not financial advice.</span>
            </div>
            """)
    else:
        st.html("""
        <div style="background:#0f1e36;border:1px dashed rgba(79,142,247,0.25);border-radius:18px;padding:56px 24px;text-align:center;margin-top:8px">
            <div style="font-size:56px;margin-bottom:12px">🤖</div>
            <div style="font-size:18px;font-weight:700;color:#e2eaff;margin-bottom:8px">Ready to Predict</div>
            <div style="color:#4a6085;font-size:13px">Pick a stock &amp; model from the sidebar, then click <b style="color:#4f8ef7">Predict BUY / SELL</b>.</div>
        </div>
        """)

# ================================================================
# PAGE: MODEL INSIGHTS
# ================================================================
elif page == "📊 Model Insights":
    st.html("""
    <h1 style="font-size:30px;font-weight:800;margin-bottom:24px;
               background:linear-gradient(135deg,#ffffff,#4f8ef7,#22d3a5);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Insights, EDA & Models
    </h1>
    """)

    tabs = st.tabs(["📉 Exploratory Data Analysis (EDA)", "🟦 Confusion Matrices", "🧠 Model Architecture"])
    
    with tabs[0]:
        st.markdown("#### Feature Relationships & Trends")
        eda_files = [
            ("eda_overview.png", "Overview: Volume, Return, RSI"),
            ("eda_ma_ratios.png", "Moving Average Ratios & Volatility"),
            ("eda_macd.png", "MACD & Signals Normalized"),
            ("eda_price_trends.png", "Price Trends across 5 Samples"),
            ("eda_correlation_heatmap.png", "Feature Correlation Heatmap")
        ]
        for fname, caption in eda_files:
            p = os.path.join(PLOT_DIR, fname)
            if os.path.exists(p):
                st.image(p, caption=caption, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
    with tabs[1]:
        st.markdown("#### Confusion Matrices (Ensemble Model)")
        st.markdown("Shows exactly how accurate the ensemble model was predicting BUY vs SELL out-of-sample.")
        cfm_p = os.path.join(PLOT_DIR, "eval_confusion_matrices.png")
        if os.path.exists(cfm_p): st.image(cfm_p, use_container_width=True)

    with tabs[2]:
        st.markdown("#### Model Architecture Parameters")
        arch_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'SVM', 'Ensemble'],
            'Type': ['Linear', 'Bagging', 'Boosting', 'Boosting', 'Kernel', 'Voting'],
            'Feature Scaling': ['StandardScaler', 'None', 'None', 'None', 'StandardScaler', 'None'],
            'Key Parameters': ['C=0.5, solver=saga', '200 trees, depth=8', 'LR=0.05, depth=5', 'LR=0.05, leaves=31', 'C=1.0, RBF kernel', 'Soft-vote: XGB+RF+LGBM']
        }
        st.dataframe(pd.DataFrame(arch_data), hide_index=True, use_container_width=True)
        
        st.markdown("#### Technical Features Used")
        feat_data = {
            'Feature': FEATURES,
            'Description': [
                'Close Price / SMA(10)', 'Close Price / SMA(50)', 'Relative Strength Index (14d)', 
                'MACD (% of Close)', 'MACD Signal (% of Close)', 'Close Price / EMA(20)', 
                'Average True Range (% of Close)', '5-Day Momentum (%)', 'Volume Ratio vs 20d Average', 
                'Bollinger Band Width'
            ]
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)
