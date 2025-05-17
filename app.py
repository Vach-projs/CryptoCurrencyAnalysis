#Run this on the local machine for smoother and easier operation, you will need an internet connection for fetching the current price on a given day. I'm not sure how to run this on colab or even you can run streamlit on colab. Look into other options if you can't run this on a local machine.

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import requests

# ----------- PAGE SETUP -----------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# ----------- PAGE SELECTOR -----------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Forecasting", "Volatility Analysis", "Correlation Insights", "Anomaly Detection", "Financial Tools"])

# ----------- LOAD DATA -----------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_sentimentn.csv')
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

# ----------- LIVE PRICE FUNCTION -----------
def get_live_btc_price():
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"}
        )
        response.raise_for_status()
        return response.json()["bitcoin"]["usd"]
    except:
        return None

live_price = get_live_btc_price()

# ----------- HOME PAGE -----------
if page == "Home":
    st.title("Welcome to the Crypto Forecasting Dashboard")

    st.subheader("Bitcoin Historical Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Historical Close'))

    if live_price:
        fig.add_trace(go.Scatter(
            x=[df['date'].min(), df['date'].max()],
            y=[live_price, live_price],
            mode='lines',
            name='Live Price',
            line=dict(color='red', dash='dash')
        ))

    st.plotly_chart(fig, use_container_width=True)

    if live_price:
        st.markdown(f"### Current Bitcoin Price: **${live_price:,.2f}**")
    else:
        st.warning("Unable to fetch live Bitcoin price.")

# ----------- FORECASTING PAGE -----------
elif page == "Forecasting":
    st.title("Price Forecasting")

    try:
        forecast_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'prophet_forecast.csv'))
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

        forecast_days = st.slider("Select forecast horizon (days)", min_value=1, max_value=len(forecast_df), value=30)

        st.subheader("Forecasted Price")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'][:forecast_days],
            y=forecast_df['yhat'][:forecast_days],
            mode='lines',
            name='Forecast'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'][:forecast_days],
            y=forecast_df['yhat_upper'][:forecast_days],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'][:forecast_days],
            y=forecast_df['yhat_lower'][:forecast_days],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0),
            name='Confidence Interval'
        ))
        fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("Forecast data not found. Please make sure 'prophet_forecast.csv' exists in the data folder.")

# ----------- VOLATILITY PAGE -----------
elif page == "Volatility Analysis":
    st.title("Volatility Analysis (EGARCH)")

    st.markdown("This page visualizes the volatility of Bitcoin using the more accurate EGARCH model.")

    # Plot EGARCH Volatility over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['egarch_vol'], mode='lines', name='EGARCH Volatility', line=dict(color='purple')))
    fig.update_layout(title="EGARCH Volatility Over Time", xaxis_title="Date", yaxis_title="Volatility")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Real-Time Risk Monitoring")

    # Volatility Threshold Alert
    latest_vol = df['egarch_vol'].iloc[-1]
    threshold = df['egarch_vol'].mean() + df['egarch_vol'].std()

    if latest_vol > threshold:
        st.error(f"High Volatility Alert: {latest_vol:.4f} exceeds threshold ({threshold:.4f})")
    else:
        st.success(f"Volatility Normal: {latest_vol:.4f} is within safe bounds")

    # Risk Meter Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_vol,
        title={'text': "Current Volatility Risk"},
        gauge={
            'axis': {'range': [0, df['egarch_vol'].max()]},
            'bar': {'color': "darkred" if latest_vol > threshold else "green"},
            'steps': [
                {'range': [0, threshold], 'color': "lightgreen"},
                {'range': [threshold, df['egarch_vol'].max()], 'color': "pink"}
            ],
        }
    ))
    st.plotly_chart(fig_gauge)

    # Volatility Regime Classification with Real-World Insights
    def classify_volatility_risk(v):
        if v < df['egarch_vol'].quantile(0.33):
            return "Low Volatility — Market is stable. Favorable conditions for investing or holding."
        elif v < df['egarch_vol'].quantile(0.66):
            return "Moderate Volatility — Proceed with caution. Normal market fluctuations expected."
        else:
            return "High Volatility — Risk is elevated. Consider waiting or applying risk management strategies."

    volatility_insight = classify_volatility_risk(latest_vol)
    st.info(volatility_insight)

# ----------- CORRELATION INSIGHTS PAGE -----------
elif page == "Correlation Insights":
    st.title("Correlation Insights")

    st.markdown("""
    Select any two features, and the scatter plot will show how they relate.
    - If points form an upward trend → **positive relationship**
    - Downward trend → **negative relationship**
    - Scattered randomly → **no strong relationship**
    """)

    features = ['close', 'volume', 'log_return', 'sentiment_score', 'egarch_vol']
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature", features, index=0)
    with col2:
        y_feature = st.selectbox("Select Y-axis feature", features, index=1)

    filtered_df = df[[x_feature, y_feature]].dropna()

    import plotly.express as px
    scatter_fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        title=f"Scatter Plot: {x_feature} vs {y_feature}",
        opacity=0.7,
        trendline="ols",
        color_discrete_sequence=["#00cc96"]
    )
    scatter_fig.update_traces(marker=dict(size=7))
    scatter_fig.update_layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        title_font_size=20,
        legend=dict(font=dict(color="white")),
    )
    scatter_fig.update_traces(line=dict(color="#FFA15A"))
    scatter_fig.update_layout(height=500)

    st.plotly_chart(scatter_fig, use_container_width=True)

# ----------- ANOMALY DETECTION PAGE -----------
elif page == "Anomaly Detection":
    st.title("Anomaly Detection in Bitcoin Returns")

    st.markdown("""
    This page detects anomalies using a **rolling z-score method** on Bitcoin's log returns.
    
    - A z-score tells us how far a data point is from the rolling average.
    - Points that are significantly far from the mean (e.g. |z| > 3) are marked as **anomalies**.
    
    You can adjust the sensitivity using the settings below.
    """)

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        window = st.slider("Rolling window size (days)", min_value=5, max_value=60, value=20)
    with col2:
        threshold = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

    # Calculate rolling stats
    df['rolling_mean'] = df['log_return'].rolling(window=window).mean()
    df['rolling_std'] = df['log_return'].rolling(window=window).std()
    df['z_score'] = (df['log_return'] - df['rolling_mean']) / df['rolling_std']

    # Flag anomalies
    df['anomaly'] = df['z_score'].abs() > threshold

    # Drop rows with NaNs from rolling
    plot_df = df.dropna(subset=['z_score'])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['date'],
        y=plot_df['log_return'],
        mode='lines',
        name='Log Return',
        line=dict(color='lightblue')
    ))
    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['anomaly']]['date'],
        y=plot_df[plot_df['anomaly']]['log_return'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8, symbol='x')
    ))
    fig.update_layout(
        title="Bitcoin Log Return Anomalies",
        xaxis_title="Date",
        yaxis_title="Log Return",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show raw data with anomalies"):
        st.dataframe(plot_df[['date', 'log_return', 'z_score', 'anomaly']].tail(50))

# ----------- FINANCIAL DECISION-MAKING TOOLS PAGE -----------
elif page == "Financial Tools":
    st.title("📈 Financial Decision-Making Tools")

    try:
        forecast_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'prophet_forecast.csv'))
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        forecast_price = forecast_df['yhat'].iloc[-1]
    except:
        st.warning("Forecast data not available.")
        forecast_price = None

    latest_vol = df['egarch_vol'].iloc[-1]

    # Volatility Regime Classification
    def classify_volatility(v):
        if v < df['egarch_vol'].quantile(0.33):
            return "Low"
        elif v < df['egarch_vol'].quantile(0.66):
            return "Moderate"
        else:
            return "High"

    vol_level = classify_volatility(latest_vol)

    st.subheader(" Current Market Assessment")
    st.markdown(f"- **Forecasted Price (Next Day)**: ${forecast_price:,.2f}" if forecast_price else "- Forecast unavailable.")
    st.markdown(f"- **Volatility Level**: {vol_level} ({latest_vol:.4f})")

    st.subheader(" Suggested Action")

    # Decision Logic without sentiment
    if vol_level == "Low":
        st.success("Favorable market conditions. Consider entering or holding positions.")
    elif vol_level == "Moderate":
        st.warning("Moderate volatility detected. Caution is advised—monitor closely.")
    else:  # High Volatility
        st.error("High market volatility. Consider waiting or using strong risk management strategies.")

