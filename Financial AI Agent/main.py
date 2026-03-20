"""
app.py  -  AI Financial Stock Analyzer  (Streamlit Edition)
Run with:  streamlit run app.py

Replaces the old tkinter main.py + dashboard.py entirely.
All backend modules (data_loader, technical_features, prediction_model,
sentiment_analysis, macro_loader, genai_engine, portfolio_recommendation,
multi_stock_prediction, realtime_prediction, stock_database) are unchanged.
"""

import threading
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

#  Page config (must be first Streamlit call) 
st.set_page_config(
    page_title="AI Financial Analyzer",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS - dark terminal-finance aesthetic 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #c9d1e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1221;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #8fa3c0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #4fc3f7 !important; }

/* Main area */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0d1628;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="metric-container"] label { color: #5a7a9e !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e2eaf5 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 700; }
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; }

/* Headings */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #4fc3f7 !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #7eb8d4 !important; }

/* Signal badge */
.signal-buy  { background:#0d2e1a; border:1px solid #1db954; color:#1db954; border-radius:6px; padding:0.4rem 1.2rem; font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:1.1rem; display:inline-block; }
.signal-sell { background:#2e0d0d; border:1px solid #e05252; color:#e05252; border-radius:6px; padding:0.4rem 1.2rem; font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:1.1rem; display:inline-block; }

/* Explanation box */
.explain-box {
    background: #0d1628;
    border: 1px solid #1e2d4a;
    border-left: 3px solid #4fc3f7;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #a8bdd4;
    white-space: pre-wrap;
}

/* DataFrames */
div[data-testid="stDataFrame"] { border: 1px solid #1e2d4a; border-radius: 8px; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: #0d1e36;
    color: #4fc3f7;
    border: 1px solid #2a4a6e;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: #1a3a5c;
    border-color: #4fc3f7;
    color: #e2f4ff;
}

/* Select boxes */
div[data-testid="stSelectbox"] > div { background: #0d1628 !important; border: 1px solid #1e2d4a !important; border-radius: 6px !important; }

/* Spinner */
div[data-testid="stSpinner"] { color: #4fc3f7; }

/* Divider */
hr { border-color: #1e2d4a; }

/* Tab */
div[data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; color: #5a7a9e; border-bottom: 2px solid transparent; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #4fc3f7 !important; border-bottom-color: #4fc3f7 !important; }

.ticker-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e2eaf5;
    letter-spacing: -0.03em;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: #3d6080;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


#  Lazy imports with spinner (heavy libs only loaded once) 
@st.cache_resource(show_spinner=False)
def _load_backend():
    from stock_database import get_stock_list
    from data_loader import load_stock, prefetch_symbols
    from technical_features import add_technical_features
    from sentiment_analysis import get_sentiment
    from macro_loader import get_macro_data
    from prediction_model import predict_price, warm_cache
    from genai_engine import generate_explanation
    from portfolio_recommendation import recommend_portfolio
    from multi_stock_prediction import multi_stock_prediction
    from realtime_prediction import get_cached_df, run_realtime_prediction_streamlit
    return {
        "get_stock_list": get_stock_list,
        "load_stock": load_stock,
        "prefetch_symbols": prefetch_symbols,
        "add_technical_features": add_technical_features,
        "get_sentiment": get_sentiment,
        "get_macro_data": get_macro_data,
        "predict_price": predict_price,
        "warm_cache": warm_cache,
        "generate_explanation": generate_explanation,
        "recommend_portfolio": recommend_portfolio,
        "multi_stock_prediction": multi_stock_prediction,
        "get_cached_df": get_cached_df,
        "run_realtime_prediction_streamlit": run_realtime_prediction_streamlit,
    }


with st.spinner("Loading AI engine"):
    B = _load_backend()

STOCKS = B["get_stock_list"]()

#  Pre-warm cache once per session 
if "prefetch_done" not in st.session_state:
    st.session_state["prefetch_done"] = True
    threading.Thread(
        target=B["prefetch_symbols"], args=(STOCKS,), kwargs={"max_workers": 10}, daemon=True
    ).start()


# 
# SIDEBAR
# 
with st.sidebar:
    st.markdown("## AI Financial\nAnalyzer")
    st.markdown("---")

    selected = st.selectbox("Select Stock", STOCKS, key="stock_sel")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Stock Analysis", "Multi-Stock Ranking", "Portfolio Optimizer", "Real-Time Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#2a4060;font-family:IBM Plex Mono,monospace'>"
        "Powered by LSTM / Ridge / Monte Carlo<br>Data via yfinance</div>",
        unsafe_allow_html=True,
    )


# 
# HELPERS
# 
def _get_df(symbol):
    df = B["get_cached_df"](symbol)
    if df is None or df.empty:
        df = B["load_stock"](symbol)
        if df is not None and not df.empty:
            df = B["add_technical_features"](df)
    return df


def _make_price_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA_20"],
        mode="lines", name="SMA 20",
        line=dict(color="#f9a825", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1222",
        font=dict(family="IBM Plex Mono", color="#8fa3c0"),
        title=dict(text=f"{symbol} / Price & SMA20", font=dict(color="#4fc3f7", size=14)),
        xaxis=dict(showgrid=False, color="#2a4060"),
        yaxis=dict(showgrid=True, gridcolor="#121d30", color="#2a4060"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=340,
    )
    return fig


def _make_rsi_chart(df):
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(224,82,82,0.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(29,185,84,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color="#e05252", line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#1db954", line_width=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        mode="lines", name="RSI",
        line=dict(color="#ce93d8", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1222",
        font=dict(family="IBM Plex Mono", color="#8fa3c0"),
        title=dict(text="RSI (14)", font=dict(color="#7eb8d4", size=13)),
        xaxis=dict(showgrid=False, color="#2a4060"),
        yaxis=dict(showgrid=True, gridcolor="#121d30", color="#2a4060", range=[0, 100]),
        margin=dict(l=0, r=0, t=40, b=0),
        height=200,
        showlegend=False,
    )
    return fig


def _make_volatility_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volatility"],
        marker_color=["#e05252" if v > 0.03 else "#4fc3f7" for v in df["Volatility"]],
        name="Volatility",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1222",
        font=dict(family="IBM Plex Mono", color="#8fa3c0"),
        title=dict(text="Daily Volatility", font=dict(color="#7eb8d4", size=13)),
        xaxis=dict(showgrid=False, color="#2a4060"),
        yaxis=dict(showgrid=True, gridcolor="#121d30", color="#2a4060"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=200,
        showlegend=False,
    )
    return fig


# 
# PAGE 1 / STOCK ANALYSIS
# 
if page == "Stock Analysis":

    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown(f"<div class='ticker-header'>{selected}</div>", unsafe_allow_html=True)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze", use_container_width=True)

    st.markdown("---")

    if run or st.session_state.get(f"analysis_{selected}"):

        with st.spinner(f"Analyzing {selected}..."):
            try:
                df = _get_df(selected)
                if df is None or df.empty:
                    st.error(f"No data available for {selected}")
                    st.stop()

                prediction  = B["predict_price"](df, symbol=selected)
                sentiment   = B["get_sentiment"]()
                inflation, interest = B["get_macro_data"]()
                latest      = df.iloc[-1]
                close       = float(latest["Close"])
                sma20       = float(latest.get("SMA_20", 0))
                rsi         = float(latest.get("RSI", 0))
                vol         = float(latest.get("Volatility", 0))
                exp_return  = (prediction - close) / close * 100
                explanation = B["generate_explanation"](
                    selected, close, sma20, rsi, vol, sentiment, inflation, prediction
                )

                # cache result
                st.session_state[f"analysis_{selected}"] = {
                    "df": df, "prediction": prediction, "sentiment": sentiment,
                    "inflation": inflation, "interest": interest,
                    "close": close, "sma20": sma20, "rsi": rsi, "vol": vol,
                    "exp_return": exp_return, "explanation": explanation,
                }
                B["warm_cache"](selected, df)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    cache = st.session_state.get(f"analysis_{selected}")
    if cache:
        close      = cache["close"]
        prediction = cache["prediction"]
        sma20      = cache["sma20"]
        rsi        = cache["rsi"]
        vol        = cache["vol"]
        exp_return = cache["exp_return"]
        df         = cache["df"]
        explanation= cache["explanation"]
        signal     = "BUY" if prediction > close else "SELL"
        sig_class  = "signal-buy" if signal == "BUY" else "signal-sell"

        #  Metric row 
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Close Price",   f"Rs.{close:,.2f}" if ".NS" in selected else f"${close:,.2f}")
        c2.metric("Predicted",     f"Rs.{prediction:,.2f}" if ".NS" in selected else f"${prediction:,.2f}",
                  delta=f"{exp_return:+.2f}%")
        c3.metric("SMA 20",        f"{sma20:,.2f}")
        c4.metric("RSI",           f"{rsi:.1f}",
                  delta="Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"))
        c5.metric("Volatility",    f"{vol:.4f}")
        c6.metric("Sentiment",     cache["sentiment"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Signal badge
        st.markdown(
            f"<span class='{sig_class}'>{signal} SIGNAL &nbsp; | &nbsp; "
            f"Expected Return: {exp_return:+.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        #  Charts 
        st.plotly_chart(_make_price_chart(df, selected), use_container_width=True)

        col_rsi, col_vol = st.columns(2)
        with col_rsi:
            st.plotly_chart(_make_rsi_chart(df), use_container_width=True)
        with col_vol:
            st.plotly_chart(_make_volatility_chart(df), use_container_width=True)

        #  AI Explanation 
        st.markdown("### AI Analysis Report")
        st.markdown(f"<div class='explain-box'>{explanation}</div>", unsafe_allow_html=True)

        #  Raw data expander 
        with st.expander("Raw OHLCV Data"):
            st.dataframe(
                df[["Open","High","Low","Close","Volume","SMA_20","RSI","Volatility"]]
                .tail(30)
                .style.format("{:.2f}", subset=["Open","High","Low","Close","SMA_20","RSI","Volatility"]),
                use_container_width=True,
            )
    else:
        st.info("Click **Analyze** to run the AI analysis for the selected stock.")


# 
# PAGE 2 / MULTI-STOCK RANKING
# 
elif page == "Multi-Stock Ranking":

    st.markdown("<div class='ticker-header'>Multi-Stock AI Ranking</div>", unsafe_allow_html=True)
    st.markdown("---")

    run_multi = st.button("Run Prediction for All Stocks", use_container_width=False)

    if run_multi or "multi_results" in st.session_state:

        if run_multi:
            with st.spinner("Running predictions across all stocks in parallel"):
                try:
                    results = B["multi_stock_prediction"](STOCKS)
                    st.session_state["multi_results"] = results
                except Exception as e:
                    st.error(f"Multi-stock prediction failed: {e}")
                    st.stop()

        results = st.session_state.get("multi_results", [])
        if not results:
            st.warning("No results available.")
        else:
            df_results = pd.DataFrame(results)
            df_results["Expected Return %"] = (df_results["Expected Return"] * 100).round(2)
            df_results = df_results[["Rank","Symbol","Current Price","Predicted Price","Expected Return %"]]

            # Color-coded bar chart
            fig_bar = px.bar(
                df_results,
                x="Symbol",
                y="Expected Return %",
                color="Expected Return %",
                color_continuous_scale=["#e05252","#2a4060","#1db954"],
                color_continuous_midpoint=0,
                text="Expected Return %",
            )
            fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1222",
                font=dict(family="IBM Plex Mono", color="#8fa3c0"),
                title=dict(text="Expected Return by Stock", font=dict(color="#4fc3f7", size=14)),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#121d30"),
                margin=dict(l=0, r=0, t=50, b=0),
                height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### Full Ranking Table")
            st.dataframe(
                df_results.style
                .background_gradient(subset=["Expected Return %"], cmap="RdYlGn")
                .format({"Current Price": "{:.2f}", "Predicted Price": "{:.2f}", "Expected Return %": "{:+.2f}%"}),
                use_container_width=True,
                height=500,
            )
    else:
        st.info("Click **Run Prediction for All Stocks** to rank all stocks by predicted return.")


# 
# PAGE 3 / PORTFOLIO OPTIMIZER
# 
elif page == "Portfolio Optimizer":

    st.markdown("<div class='ticker-header'>Portfolio Optimizer</div>", unsafe_allow_html=True)
    st.markdown("*Monte Carlo Sharpe-ratio optimisation across all stocks*")
    st.markdown("---")

    run_port = st.button("Optimize Portfolio", use_container_width=False)

    if run_port or "portfolio_result" in st.session_state:

        if run_port:
            with st.spinner("Running Monte Carlo simulation"):
                try:
                    portfolio = B["recommend_portfolio"](STOCKS)
                    st.session_state["portfolio_result"] = portfolio
                except Exception as e:
                    st.error(f"Portfolio optimization failed: {e}")
                    st.stop()

        portfolio = st.session_state.get("portfolio_result", [])
        if not portfolio:
            st.warning("Not enough data to build a portfolio.")
        else:
            df_port = pd.DataFrame(portfolio, columns=["Symbol","Weight"])
            df_port["Allocation %"] = (df_port["Weight"] * 100).round(2)
            df_port = df_port.sort_values("Allocation %", ascending=False).reset_index(drop=True)
            df_port.index += 1

            col_pie, col_bar = st.columns([1, 1])

            with col_pie:
                fig_pie = px.pie(
                    df_port,
                    names="Symbol",
                    values="Allocation %",
                    hole=0.55,
                    color_discrete_sequence=px.colors.sequential.ice,
                )
                fig_pie.update_traces(
                    textposition="outside",
                    textinfo="label+percent",
                    textfont=dict(family="IBM Plex Mono", size=11),
                )
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    font=dict(family="IBM Plex Mono", color="#8fa3c0"),
                    title=dict(text="Optimal Allocation", font=dict(color="#4fc3f7")),
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=400,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                fig_hbar = px.bar(
                    df_port.head(15),
                    x="Allocation %",
                    y="Symbol",
                    orientation="h",
                    color="Allocation %",
                    color_continuous_scale="ice",
                    text="Allocation %",
                )
                fig_hbar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_hbar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0d1222",
                    font=dict(family="IBM Plex Mono", color="#8fa3c0"),
                    title=dict(text="Weight Distribution", font=dict(color="#4fc3f7")),
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=True, gridcolor="#121d30"),
                    yaxis=dict(showgrid=False, autorange="reversed"),
                    margin=dict(l=0, r=60, t=50, b=0),
                    height=400,
                )
                st.plotly_chart(fig_hbar, use_container_width=True)

            st.markdown("### Allocation Table")
            st.dataframe(
                df_port[["Symbol","Allocation %"]]
                .style.bar(subset=["Allocation %"], color="#1a3a5c")
                .format({"Allocation %": "{:.2f}%"}),
                use_container_width=True,
                height=420,
            )
    else:
        st.info("Click **Optimize Portfolio** to run the Monte Carlo simulation.")


# 
# PAGE 4 / REAL-TIME MONITOR
# 
elif page == "Real-Time Monitor":

    st.markdown("<div class='ticker-header'>Real-Time Monitor</div>", unsafe_allow_html=True)
    st.markdown(f"*Live prediction for **{selected}** - auto-refreshes every 60 s*")
    st.markdown("---")

    col_go, col_stop = st.columns([1, 5])
    with col_go:
        fetch_now = st.button("Fetch Now", use_container_width=True)

    # Fetch on button press or on first visit
    if fetch_now or f"rt_{selected}" not in st.session_state:
        with st.spinner(f"Fetching live data for {selected}..."):
            try:
                cp, pp, sig = B["run_realtime_prediction_streamlit"](selected)
                st.session_state[f"rt_{selected}"] = {
                    "cp": cp, "pp": pp, "sig": sig,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as e:
                st.error(f"Real-time fetch failed: {e}")
                st.stop()

    rt = st.session_state.get(f"rt_{selected}", {})
    if rt and rt.get("cp") is not None:
        cp  = rt["cp"]
        pp  = rt["pp"]
        sig = rt["sig"]
        ts  = rt["ts"]
        chg = (pp - cp) / cp * 100

        m1, m2, m3, m4 = st.columns(4)
        currency = "Rs." if ".NS" in selected else "$"
        m1.metric("Current Price",   f"{currency}{cp:,.2f}")
        m2.metric("Predicted Price", f"{currency}{pp:,.2f}", delta=f"{chg:+.2f}%")
        m3.metric("Signal",          sig)
        m4.metric("Last Updated",    ts)

        sig_class = "signal-buy" if sig == "BUY" else "signal-sell"
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='{sig_class}'>{sig} &nbsp;|&nbsp; {selected} &nbsp;|&nbsp; "
            f"Delta {chg:+.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Show price history chart
        df = _get_df(selected)
        if df is not None and not df.empty:
            st.plotly_chart(_make_price_chart(df, selected), use_container_width=True)

        # Auto-refresh toggle
        st.markdown("---")
        auto = st.toggle("Auto-refresh every 60 seconds", value=False)
        if auto:
            placeholder = st.empty()
            for i in range(60, 0, -1):
                placeholder.markdown(
                    f"<div style='font-family:IBM Plex Mono;color:#3d6080;font-size:0.8rem'>"
                    f"Next refresh in {i}s</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(1)
            st.rerun()
    else:
        st.info("Click **Fetch Now** to get the latest prediction.")