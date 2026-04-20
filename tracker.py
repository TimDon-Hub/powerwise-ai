"""
PowerWise AI - Data & Consumption Tracker
Tracks device activity, calculates energy in kWh,
estimates costs in Naira, and generates historical data for ML.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# ── Cost Calculation ────────────────────────────────────────────
def calculate_cost(kwh: float, cost_per_unit: float = 68.7) -> float:
    """
    Convert kWh to Naira cost.
    Default rate ~₦68.7/kWh (approximate DisCo Band A tariff).
    """
    return round(kwh * cost_per_unit, 2)


# ── Device Summary ──────────────────────────────────────────────
def get_consumption_summary(devices: list, cost_per_unit: float = 68.7) -> tuple:
    """
    Returns a DataFrame summary of all device consumption
    and the total kWh consumed.
    """
    rows = []
    total_kwh = 0.0

    for device in devices:
        kwh = device.get_total_kwh()
        total_kwh += kwh
        rows.append({
            "Device":       f"{device.icon} {device.name}",
            "Room":         device.room,
            "Status":       device.status,
            "Watts (W)":    device.watts,
            "Time On (min)": round(device.get_total_minutes_on(), 1),
            "kWh Used":     round(kwh, 4),
            "Cost (₦)":     calculate_cost(kwh, cost_per_unit),
        })

    df = pd.DataFrame(rows)
    return df, total_kwh


# ── Historical Data Generator ───────────────────────────────────
def generate_historical_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic daily energy consumption data
    for a Nigerian household over the given number of days.

    Patterns modelled:
    - Weekends use ~30% more electricity
    - Occasional high-usage days (ironing, guests)
    - Slight upward trend over time (new devices, season)
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    base_kwh = 3.2  # Average Nigerian household daily usage

    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).date()
        day_of_week = date.weekday()

        # Weekend factor
        weekend_factor = 1.3 if day_of_week >= 5 else 1.0
        # Slight growth trend
        trend = 1 + (i / days) * 0.05
        # Random noise
        noise = np.random.normal(0, 0.4)
        # Occasional spike day (e.g., 1 in 7 chance)
        spike = 1.5 if random.random() < 0.14 else 1.0

        consumption = max(0.5, base_kwh * weekend_factor * trend * spike + noise)

        records.append({
            "date":            date,
            "consumption_kwh": round(consumption, 3),
            "units_used":      round(consumption, 3),  # 1 unit = 1 kWh
            "day_of_week":     day_of_week,
            "is_weekend":      int(day_of_week >= 5),
        })

    return pd.DataFrame(records)


# ── Live Session Logger ─────────────────────────────────────────
def log_session_event(log: list, event_type: str, device_name: str, kwh: float = 0.0):
    """Appends a timestamped event to the session log list."""
    log.append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "event":      event_type,
        "device":     device_name,
        "kwh":        round(kwh, 5),
    })
    return log
