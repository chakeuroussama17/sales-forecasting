import pandas as pd
from prophet import Prophet

def load_and_prepare(filepath, category="All"):
    df = pd.read_csv(filepath, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False)
    if category != "All":
        df = df[df["Category"] == category]
    daily = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "ds", "Sales": "y"})
    )
    return daily.sort_values("ds").reset_index(drop=True)

def run_forecast(df, horizon_days):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    return forecast, model

def get_forecast_summary(forecast, horizon_days):
    future_only = forecast.tail(horizon_days)
    return {
        "avg_daily":   round(future_only["yhat"].mean(), 2),
        "total":       round(future_only["yhat"].sum(), 2),
        "peak_date":   str(future_only.loc[future_only["yhat"].idxmax(), "ds"].date()),
        "peak_value":  round(future_only["yhat"].max(), 2),
        "lower_bound": round(future_only["yhat_lower"].mean(), 2),
        "upper_bound": round(future_only["yhat_upper"].mean(), 2),
    }
