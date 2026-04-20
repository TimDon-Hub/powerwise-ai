"""
PowerWise AI - ML Intelligence Layer
Uses Linear Regression trained on historical consumption data to:
  - Predict daily energy usage
  - Estimate how long prepaid units will last
  - Identify weekly usage patterns
  - Generate smart energy-saving recommendations
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta


# ── Model Training ──────────────────────────────────────────────
def train_prediction_model(historical_data: pd.DataFrame):
    """
    Trains a Linear Regression model on historical daily consumption.

    Features:
        - day_index      : sequential day number (captures trend)
        - is_weekend     : binary flag for weekend (captures weekly pattern)
        - rolling_3day   : 3-day rolling average (captures momentum)

    Returns:
        model    : trained LinearRegression instance
        scaler   : fitted StandardScaler
        metrics  : dict with MAE and R² scores
        df       : enriched DataFrame used for training
    """
    df = historical_data.copy()
    df = df.reset_index(drop=True)
    df["day_index"] = df.index
    df["rolling_3day"] = (
        df["consumption_kwh"].rolling(window=3, min_periods=1).mean()
    )

    features = ["day_index", "is_weekend", "rolling_3day"]
    X = df[features].values
    y = df["consumption_kwh"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    metrics = {
        "mae":      round(mean_absolute_error(y, y_pred), 4),
        "r2":       round(r2_score(y, y_pred), 4),
        "avg_kwh":  round(float(np.mean(y)), 3),
    }

    return model, scaler, metrics, df


# ── Unit Depletion Prediction ───────────────────────────────────
def predict_unit_depletion(
    model,
    scaler,
    historical_data: pd.DataFrame,
    remaining_units: float,
    forecast_days: int = 30,
) -> tuple:
    """
    Predicts how many days until prepaid units are exhausted,
    and returns a forecast DataFrame for charting.

    Returns:
        prediction_df : DataFrame with date, predicted_kwh, cumulative_kwh
        days_left     : estimated days until units run out (int)
    """
    last_idx = len(historical_data)
    last_rolling = historical_data["consumption_kwh"].iloc[-3:].mean()

    future_rows = []
    for i in range(forecast_days):
        future_date = datetime.now().date() + timedelta(days=i + 1)
        is_weekend = int(future_date.weekday() >= 5)
        future_rows.append({
            "day_index":    last_idx + i,
            "is_weekend":   is_weekend,
            "rolling_3day": last_rolling,
        })

    X_future = pd.DataFrame(future_rows)[["day_index", "is_weekend", "rolling_3day"]].values
    X_future_scaled = scaler.transform(X_future)
    predicted_daily = model.predict(X_future_scaled)
    # Clip to realistic bounds for a Nigerian household
    predicted_daily = np.clip(predicted_daily, 0.5, 12.0)

    cumulative = np.cumsum(predicted_daily)

    days_left = forecast_days  # default: more than forecast window
    for i, cum in enumerate(cumulative):
        if cum >= remaining_units:
            days_left = i + 1
            break

    future_dates = [
        datetime.now().date() + timedelta(days=i + 1) for i in range(forecast_days)
    ]
    prediction_df = pd.DataFrame({
        "date":            future_dates,
        "predicted_kwh":   np.round(predicted_daily, 3),
        "cumulative_kwh":  np.round(cumulative, 3),
    })

    return prediction_df, days_left


# ── Weekly Pattern Analysis ─────────────────────────────────────
def get_weekly_pattern(historical_data: pd.DataFrame) -> pd.Series:
    """Returns average consumption grouped by day of week."""
    df = historical_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_name"] = df["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = df.groupby("day_name")["consumption_kwh"].mean().reindex(order)
    return weekly


# ── Smart Recommendations ───────────────────────────────────────
def generate_recommendations(
    devices: list,
    avg_daily_kwh: float,
    days_left: int,
    cost_per_unit: float,
) -> list:
    """
    Generates personalised energy-saving tips based on
    device consumption data and usage patterns.
    """
    from tracker import calculate_cost

    # Sort devices by consumption (descending)
    sorted_devices = sorted(devices, key=lambda d: d.get_total_kwh(), reverse=True)
    top_device = sorted_devices[0] if sorted_devices else None

    tips = []

    if top_device and top_device.get_total_kwh() > 0:
        tips.append(
            f"🔌 **{top_device.icon} {top_device.name}** is your biggest consumer today. "
            f"Reducing its usage by 30 min/day could save ~"
            f"**{(top_device.watts/1000 * 0.5):.3f} kWh** daily."
        )

    daily_cost = calculate_cost(avg_daily_kwh, cost_per_unit)
    tips.append(
        f"💰 At current usage you spend approx **₦{daily_cost:.0f}/day** on electricity "
        f"(₦{daily_cost * 30:.0f}/month)."
    )

    if days_left <= 5:
        tips.append(
            "⚠️ **Units critically low!** Consider recharging now and turning off "
            "non-essential devices immediately."
        )
    elif days_left <= 10:
        tips.append(
            f"⏳ Units will last ~**{days_left} more days** at current consumption. "
            "Plan to recharge soon."
        )

    tips.append(
        "💡 Switching to **LED bulbs** (8W vs 60W) can cut lighting costs by up to **87%** "
        "with no change to brightness."
    )

    tips.append(
        "🌙 Enable **Auto Shutoff** to turn off idle devices automatically — "
        "this alone can save 10–20% of household electricity."
    )

    tips.append(
        f"📉 Reducing daily usage by just **0.5 kWh** could extend your units by "
        f"~**{int(0.5 / max(avg_daily_kwh, 0.01) * days_left)} extra days**."
    )

    return tips
