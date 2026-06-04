# 📈 Sales Forecasting Dashboard

An interactive sales forecasting dashboard built with Facebook Prophet and Streamlit.

## What it does
- Loads historical retail sales data (Superstore dataset)
- Forecasts revenue over 30 / 60 / 90-day horizons using Facebook Prophet
- Visualizes confidence intervals interactively with Plotly
- Breaks down forecast into trend, weekly, and yearly seasonality components

## Key results
- Forecasts 90-day revenue with confidence intervals
- Processes 4 years of retail data (9,994 orders)
- Peak sales months: November & December (holiday seasonality)
- Peak sales days: Monday & Friday

## Tech stack
- Facebook Prophet — time series forecasting
- Plotly — interactive charts
- Streamlit — web app framework
- Pandas — data processing

## How to run
1. Upload the Superstore CSV in the sidebar
2. Select a product category
3. Choose forecast horizon (30/60/90 days)
4. Click Run Forecast
