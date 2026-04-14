
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr
import os

st.set_page_config(
    page_title="Sentiment → Price Signals",
    page_icon="📰", layout="wide"
)
st.title("Financial News Sentiment → Stock Price Signals")
st.caption("FinBERT NLP · Nifty 50 Companies · NSE Data")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    signals  = pd.read_csv("data/signals.csv")
    ic       = pd.read_csv("data/ic_results.csv")
    bt       = pd.read_csv("data/backtest.csv")
    sent     = pd.read_csv("data/sentiment.csv")
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    bt["date"]      = pd.to_datetime(bt["date"])
    return signals, ic, bt, sent

try:
    df_signals, ic_df, bt_df, df_sent = load_data()
except FileNotFoundError:
    st.error("Run the Jupyter notebook first to generate data files.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    selected_ticker = st.selectbox(
        "Company",
        options=df_signals["ticker"].unique()
    )
    signal_col = st.selectbox(
        "Sentiment Signal",
        options=["sentiment_wavg", "polarity_ratio", "sentiment_mean"],
        format_func=lambda x: {
            "sentiment_wavg":  "Confidence-Weighted Sentiment",
            "polarity_ratio":  "Polarity Ratio",
            "sentiment_mean":  "Simple Mean Sentiment"
        }.get(x, x)
    )
    return_horizon = st.selectbox(
        "Return Horizon",
        options=["ret_0d", "ret_1d", "ret_2d", "ret_5d"],
        index=1,
        format_func=lambda x: {
            "ret_0d": "Same Day Return",
            "ret_1d": "Next Day Return",
            "ret_2d": "2-Day Forward",
            "ret_5d": "5-Day Forward"
        }.get(x, x)
    )
    bt_threshold = st.slider(
        "Signal Threshold for Strategy",
        0.05, 0.50, 0.20, 0.05
    )

# ── Top metrics ───────────────────────────────────────────────────────────────
st.subheader("Dashboard Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Companies Tracked",  df_signals["ticker"].nunique())
m2.metric("Total Articles",     len(df_sent))
m3.metric("Observation Days",   df_signals["date"].nunique())

clean = df_signals.dropna(subset=[signal_col, return_horizon])
overall_ic, overall_pval = spearmanr(
    clean[signal_col], clean[return_horizon]
) if len(clean) > 5 else (0, 1)

m4.metric("Overall IC",
          f"{overall_ic:+.3f}",
          delta="significant" if overall_pval < 0.05 else "not significant",
          delta_color="normal" if overall_pval < 0.05 else "off")

bt_ret = bt_df["cumulative_return"].iloc[-1] * 100 if not bt_df.empty else 0
m5.metric("Strategy Return", f"{bt_ret:+.1f}%")

st.divider()

# ── Company deep dive ─────────────────────────────────────────────────────────
st.subheader(f"Company Deep Dive — {selected_ticker}")

company_data = df_signals[
    df_signals["ticker"] == selected_ticker
].dropna(subset=[signal_col, "close"]).sort_values("date")

if not company_data.empty:
    fig = go.Figure()

    colors = ["#22c55e" if v > 0 else "#ef4444"
              for v in company_data[signal_col]]

    fig.add_bar(
        x=company_data["date"],
        y=company_data[signal_col],
        name="Sentiment",
        marker_color=colors, opacity=0.7
    )

    fig.add_trace(go.Scatter(
        x=company_data["date"],
        y=company_data["close"],
        mode="lines", name="Price",
        line=dict(color="#f59e0b", width=2),
        yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark", height=420,
        yaxis=dict(title="Sentiment"),
        yaxis2=dict(title="Price (₹)", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    # IC for this company
    c_clean = company_data.dropna(subset=[signal_col, return_horizon])
    if len(c_clean) >= 5:
        c_ic, c_pval = spearmanr(c_clean[signal_col], c_clean[return_horizon])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IC (this company)", f"{c_ic:+.3f}")
        c2.metric("P-value", f"{c_pval:.3f}")
        c3.metric("Observations", len(c_clean))
        c4.metric("Avg Articles/Day",
                  f"{company_data['n_articles'].mean():.1f}")

st.divider()

# ── IC comparison across companies ───────────────────────────────────────────
st.subheader("Information Coefficient by Company")

ic_plot = ic_df[ic_df["ticker"] != "ALL"].sort_values("ic", ascending=True)
fig_ic = go.Figure(go.Bar(
    x=ic_plot["ic"],
    y=ic_plot["ticker"],
    orientation="h",
    marker_color=["#22c55e" if v > 0 else "#ef4444" for v in ic_plot["ic"]],
    text=[f"{v:+.3f}" for v in ic_plot["ic"]],
    textposition="outside"
))
fig_ic.add_vline(x=0, line_color="white", line_width=0.5)
fig_ic.add_vline(x=0.05, line_dash="dash", line_color="#22c55e",
                 opacity=0.5, annotation_text="Useful (0.05)")
fig_ic.update_layout(
    template="plotly_dark", height=400,
    xaxis_title="Information Coefficient"
)
st.plotly_chart(fig_ic, use_container_width=True)

st.divider()

# ── Sentiment heatmap ─────────────────────────────────────────────────────────
st.subheader("Sentiment Heatmap — All Companies")

df_signals["date_str"] = pd.to_datetime(df_signals["date"]).dt.strftime("%b %d")
pivot = df_signals.pivot_table(
    index="ticker", columns="date_str",
    values=signal_col, aggfunc="mean"
)
if not pivot.empty:
    fig_hm = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(color="Sentiment")
    )
    fig_hm.update_layout(
        template="plotly_dark", height=420,
        xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ── Backtest results ──────────────────────────────────────────────────────────
st.subheader("Strategy Backtest — Long/Short on Sentiment")
st.caption(f"Signal threshold: ±{bt_threshold} | No transaction costs")

if not bt_df.empty:
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_df["date"],
        y=bt_df["cumulative_return"] * 100,
        mode="lines",
        name="Sentiment Strategy",
        line=dict(color="#22c55e", width=2),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.1)"
    ))
    fig_bt.add_hline(y=0, line_color="white", line_dash="dash")
    fig_bt.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Return",   f"{bt_df['cumulative_return'].iloc[-1]*100:+.1f}%")
    b2.metric("Win Rate",
              f"{(bt_df['port_return'] > 0).mean()*100:.0f}%")
    daily_std = bt_df["port_return"].std()
    sharpe = (bt_df["port_return"].mean() / daily_std * np.sqrt(252)
              if daily_std > 0 else 0)
    b3.metric("Sharpe Ratio",   f"{sharpe:.2f}")
    b4.metric("Trading Days",   len(bt_df))

st.divider()

# ── Recent headlines ──────────────────────────────────────────────────────────
st.subheader(f"Recent Headlines — {selected_ticker}")
recent = (
    df_sent[df_sent["ticker"] == selected_ticker]
    .sort_values("published_at", ascending=False)
    .head(15)
    [["published_at", "headline", "sentiment_label",
      "sentiment_score", "source"]]
)

def colour_sentiment(row):
    if row["sentiment_label"] == "positive":
        return ["background-color: #14532d"] * len(row)
    elif row["sentiment_label"] == "negative":
        return ["background-color: #450a0a"] * len(row)
    return [""] * len(row)

st.dataframe(
    recent.style.apply(colour_sentiment, axis=1)
               .format({"sentiment_score": "{:.3f}"}),
    hide_index=True,
    use_container_width=True
)

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
csv = df_signals.to_csv(index=False)
st.download_button("Download Full Signal Dataset",
                   csv, "sentiment_signals.csv", "text/csv")
