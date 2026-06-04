import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forecaster import load_and_prepare, run_forecast, get_forecast_summary

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting Dashboard")
st.caption("Historical sales analysis + AI-powered revenue forecasting using Facebook Prophet")

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    uploaded = st.file_uploader("Upload Superstore CSV", type=["csv"])
    category = st.selectbox("Product category", ["All", "Furniture", "Office Supplies", "Technology"])
    horizon = st.radio("Forecast horizon", [30, 60, 90], format_func=lambda x: f"{x} days", horizontal=True)
    run_btn = st.button("🚀 Run Forecast", type="primary", disabled=not uploaded)

if not uploaded:
    st.info("👈 Upload the Superstore CSV in the sidebar to get started.")
    st.stop()

df = load_and_prepare(uploaded, category)
df_prophet = df[["ds", "y"]].copy()

tab1, tab2, tab3 = st.tabs(["📊 Historical Data", "🔮 Forecast", "🧩 Components"])

with tab1:
    st.subheader(f"Historical Sales — {category}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total records", f"{len(df):,}")
    m2.metric("Avg daily sales", f"${df['y'].mean():,.0f}")
    m3.metric("Peak day", f"${df['y'].max():,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Daily Sales", line=dict(color="#4CAF50", width=1)))
    df["rolling_30"] = df["y"].rolling(30).mean()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["rolling_30"], mode="lines", name="30-day avg", line=dict(color="#FF9800", width=2)))
    fig.update_layout(title="Daily Sales with 30-Day Rolling Average", xaxis_title="Date", yaxis_title="Sales ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not run_btn:
        st.info("👈 Click Run Forecast in the sidebar.")
        st.stop()

    with st.spinner("Training Prophet model..."):
        forecast, model = run_forecast(df_prophet, horizon)
        summary = get_forecast_summary(forecast, horizon)

    st.success(f"✅ Forecast complete — {horizon}-day horizon")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Projected total", f"${summary['total']:,.0f}")
    s2.metric("Avg daily", f"${summary['avg_daily']:,.0f}")
    s3.metric("Peak day", summary["peak_date"])
    s4.metric("Peak value", f"${summary['peak_value']:,.0f}")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="Actual Sales", line=dict(color="#4CAF50", width=1)))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(33,150,243,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence interval"
    ))
    fig_fc.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color="#2196F3", width=2, dash="dot")))
    split_date = df_prophet["ds"].max()
    fig_fc.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="gray", dash="dash"))
    fig_fc.update_layout(title=f"{horizon}-Day Forecast with Confidence Intervals", xaxis_title="Date", yaxis_title="Sales ($)", hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

    future_rows = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_rows.columns = ["Date", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
    future_rows["Date"] = future_rows["Date"].dt.date
    st.dataframe(future_rows.round(2), use_container_width=True)
    st.download_button("⬇️ Download forecast CSV", future_rows.to_csv(index=False), file_name=f"forecast_{horizon}days.csv")

with tab3:
    if not run_btn:
        st.info("👈 Run the forecast first.")
        st.stop()

    fig_trend = go.Figure(go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", line=dict(color="#9C27B0", width=2)))
    fig_trend.update_layout(title="Overall Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    weekly = forecast[["ds", "weekly"]].copy()
    weekly["day"] = weekly["ds"].dt.day_name()
    weekly_avg = weekly.groupby("day")["weekly"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    fig_w = go.Figure(go.Bar(x=weekly_avg.index, y=weekly_avg.values, marker_color="#FF5722"))
    fig_w.update_layout(title="Weekly Seasonality")
    st.plotly_chart(fig_w, use_container_width=True)
