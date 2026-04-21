"""
PowerWise AI — Smart Home Energy Management System
===================================================
A Streamlit web application that simulates and manages household
electricity usage in the Nigerian context.

Developed for: SPE LASU Nexus 3.0 Hackathon 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from devices import Device, get_default_devices
from tracker import (
    generate_historical_data,
    calculate_cost,
    get_consumption_summary,
    log_session_event,
)
from ml_model import (
    train_prediction_model,
    predict_unit_depletion,
    get_weekly_pattern,
    generate_recommendations,
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PowerWise AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Header */
    .pw-header {
        font-size: 2.4rem;
        font-weight: 900;
        color: #FF6B35;
        letter-spacing: -1px;
    }
    .pw-sub {
        color: #6c757d;
        font-size: 0.95rem;
        margin-top: -10px;
    }

    /* Alert boxes */
    .alert-warning {
        background: #fff8e1;
        border-left: 5px solid #FFC107;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.9rem;
    }
    .alert-danger {
        background: #fdecea;
        border-left: 5px solid #f44336;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.9rem;
    }

    /* Device cards */
    .device-badge-on  { color: #28a745; font-weight: 700; }
    .device-badge-off { color: #dc3545; font-weight: 700; }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════
def init_state():
    if "devices" not in st.session_state:
        st.session_state.devices = get_default_devices()

    if "remaining_units" not in st.session_state:
        st.session_state.remaining_units = 50.0

    if "cost_per_unit" not in st.session_state:
        st.session_state.cost_per_unit = 68.7  # ₦/kWh

    if "historical_data" not in st.session_state:
        st.session_state.historical_data = generate_historical_data(30)

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    if "event_log" not in st.session_state:
        st.session_state.event_log = []

    if "auto_shutoff" not in st.session_state:
        st.session_state.auto_shutoff = True

    if "idle_threshold" not in st.session_state:
        st.session_state.idle_threshold = 10


init_state()
devices = st.session_state.devices


# ═══════════════════════════════════════════════════════════════
# AUTO SHUTOFF ENGINE
# ═══════════════════════════════════════════════════════════════
def run_auto_shutoff():
    for device in devices:
        device.idle_minutes_threshold = st.session_state.idle_threshold
        if st.session_state.auto_shutoff and device.check_auto_shutoff():
            kwh_saved = device.get_current_session_kwh()
            device.turn_off()
            msg = (
                f"🔴 **{device.icon} {device.name}** auto-switched OFF "
                f"(idle ≥ {st.session_state.idle_threshold} min) — "
                f"saved **{kwh_saved:.4f} kWh**"
            )
            entry = {"message": msg, "time": datetime.now().strftime("%H:%M:%S"), "type": "warning"}
            if entry not in st.session_state.alerts:
                st.session_state.alerts.insert(0, entry)
            log_session_event(
                st.session_state.event_log, "AUTO_OFF", device.name, kwh_saved
            )


run_auto_shutoff()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ PowerWise AI")
    st.caption("Smart Home Energy Manager")
    st.divider()

    st.markdown("### ⚙️ Settings")

    st.session_state.remaining_units = st.number_input(
        "Prepaid Units Remaining (kWh)",
        min_value=0.0, max_value=1000.0,
        value=st.session_state.remaining_units,
        step=0.5,
        help="Enter your current meter unit balance",
    )

    st.session_state.cost_per_unit = st.number_input(
        "Cost per Unit (₦ / kWh)",
        min_value=10.0, max_value=500.0,
        value=st.session_state.cost_per_unit,
        step=1.0,
        help="Your electricity tariff per kWh",
    )

    st.divider()
    st.markdown("### 🤖 AI Auto Shutoff")

    st.session_state.auto_shutoff = st.toggle(
        "Enable Auto Shutoff",
        value=st.session_state.auto_shutoff,
        help="Automatically turns off devices after idle time",
    )

    if st.session_state.auto_shutoff:
        st.session_state.idle_threshold = st.slider(
            "Idle Timeout (minutes)",
            min_value=1, max_value=60,
            value=st.session_state.idle_threshold,
        )

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    if st.button("🗑️ Clear Alerts", use_container_width=True):
        st.session_state.alerts = []
        st.rerun()
# ═══════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="pw-header">⚡ PowerWise AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="pw-sub">Smart Home Energy Management System</p>',
    unsafe_allow_html=True,
)
st.divider()


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔌 Device Control",
    "📊 Analytics",
    "🤖 AI Predictions",
])


# ───────────────────────────────────────────────────────────────
# TAB 1 · DASHBOARD
# ───────────────────────────────────────────────────────────────
with tab1:
    summary_df, total_kwh = get_consumption_summary(
        devices, st.session_state.cost_per_unit
    )
    total_cost     = calculate_cost(total_kwh, st.session_state.cost_per_unit)
    units_left     = max(0.0, st.session_state.remaining_units - total_kwh)
    active_count   = sum(1 for d in devices if d.is_on)
    pct_used       = (total_kwh / max(st.session_state.remaining_units, 0.001)) * 100
    efficiency_pct = max(0, 100 - pct_used)

    # ── KPI Cards ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Units Remaining",  f"{units_left:.2f} kWh",
              delta=f"-{total_kwh:.4f} kWh used")
    c2.metric("💰 Cost So Far",      f"₦{total_cost:,.2f}")
    c3.metric("🔌 Active Devices",   f"{active_count} / {len(devices)}")
    c4.metric("🌱 Efficiency Score", f"{efficiency_pct:.0f} %")

    # ── Low Unit Warning ────────────────────────────────────────
    if units_left < 5:
        st.markdown(
            f'<div class="alert-danger">⛔ <b>Critical:</b> Only {units_left:.2f} kWh left! '
            f"Please recharge immediately.</div>",
            unsafe_allow_html=True,
        )
    elif units_left < 15:
        st.markdown(
            f'<div class="alert-warning">⚠️ <b>Low Units:</b> {units_left:.2f} kWh remaining. '
            f"Consider recharging soon.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Alerts & Summary ───────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🚨 Smart Alerts")
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:6]:
                box_class = "alert-danger" if alert.get("type") == "danger" else "alert-warning"
                st.markdown(
                    f'<div class="{box_class}">{alert["message"]} '
                    f'<small style="color:#999">({alert["time"]})</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No alerts — your home is running efficiently!")

    with right:
        st.markdown("### 📋 Device Overview")
        st.dataframe(
            summary_df[["Device", "Room", "Status", "kWh Used", "Cost (₦)"]],
            hide_index=True,
            use_container_width=True,
        )


# ───────────────────────────────────────────────────────────────
# TAB 2 · DEVICE CONTROL
# ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔌 Device Control Panel")
    st.caption(
        "Toggle devices ON/OFF. Hit **'I'm Here'** to reset the idle timer "
        "and prevent auto-shutoff."
    )

    # Group by room
    rooms: dict[str, list] = {}
    for device in devices:
        rooms.setdefault(device.room, []).append(device)

    for room_name, room_devices in rooms.items():
        st.markdown(f"#### 🏠 {room_name}")
        cols = st.columns(len(room_devices))

        for idx, device in enumerate(room_devices):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{device.icon} {device.name}**")
                    st.caption(f"{device.watts} W")

                    new_state = st.toggle(
                        "Power",
                        value=device.is_on,
                        key=f"tog_{device.name}",
                    )

                    # Apply state change
                    if new_state and not device.is_on:
                        device.turn_on()
                        log_session_event(
                            st.session_state.event_log, "TURNED_ON", device.name
                        )
                    elif not new_state and device.is_on:
                        kwh = device.get_current_session_kwh()
                        device.turn_off()
                        log_session_event(
                            st.session_state.event_log, "TURNED_OFF", device.name, kwh
                        )

                    badge = "🟢 ON" if device.is_on else "🔴 OFF"
                    st.markdown(f"**{badge}**")
                    st.caption(f"Used: {device.get_total_kwh():.4f} kWh")

                    if device.is_on:
                        idle = device.get_idle_minutes()
                        if idle > 0:
                            color = "red" if idle >= device.idle_minutes_threshold * 0.8 else "orange"
                            st.markdown(
                                f"<small style='color:{color}'>⏱ Idle: {idle:.1f} min</small>",
                                unsafe_allow_html=True,
                            )
                        if st.button("👋 I'm Here", key=f"here_{device.name}"):
                            device.update_interaction()
                            st.rerun()

        st.divider()

    # Bulk controls
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("🔴 Turn OFF All Devices", use_container_width=True, type="primary"):
            for d in devices:
                d.turn_off()
            st.success("All devices turned OFF.")
            st.rerun()
    with bc2:
        if st.button("🟢 Turn ON All Devices", use_container_width=True):
            for d in devices:
                d.turn_on()
            st.rerun()

    # Event log
    if st.session_state.event_log:
        st.markdown("### 📝 Session Event Log")
        log_df = pd.DataFrame(st.session_state.event_log[:20])
        st.dataframe(log_df, hide_index=True, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 3 · ANALYTICS
# ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Energy Analytics")

    summary_df, total_kwh = get_consumption_summary(
        devices, st.session_state.cost_per_unit
    )
    hist = st.session_state.historical_data

    # ── Device Charts ───────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = px.bar(
            summary_df,
            x="Device", y="kWh Used",
            color="Status",
            color_discrete_map={"ON": "#28a745", "OFF": "#6c757d"},
            title="⚡ Energy Consumption per Device",
            labels={"kWh Used": "kWh"},
        )
        fig_bar.update_layout(xaxis_tickangle=-30, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        cost_data = summary_df[summary_df["Cost (₦)"] > 0]
        if not cost_data.empty:
            fig_pie = px.pie(
                cost_data,
                names="Device", values="Cost (₦)",
                title="💰 Cost Share per Device",
                color_discrete_sequence=px.colors.sequential.Oranges_r,
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Turn on some devices to see cost distribution.")

    # ── Historical Trend ────────────────────────────────────────
    st.markdown("#### 📈 30-Day Historical Consumption")
    avg_hist = hist["consumption_kwh"].mean()
    fig_hist = px.area(
        hist, x="date", y="consumption_kwh",
        title="Daily Energy Consumption — Last 30 Days",
        labels={"consumption_kwh": "kWh", "date": "Date"},
        color_discrete_sequence=["#FF6B35"],
    )
    fig_hist.add_hline(
        y=avg_hist,
        line_dash="dash",
        line_color="navy",
        annotation_text=f"Avg {avg_hist:.2f} kWh",
        annotation_position="top left",
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Weekly Pattern ──────────────────────────────────────────
    col_w, col_stats = st.columns([2, 1])

    with col_w:
        weekly = get_weekly_pattern(hist)
        fig_week = px.bar(
            x=weekly.index, y=weekly.values,
            color=weekly.values,
            color_continuous_scale="RdYlGn_r",
            title="📅 Average Consumption by Day of Week",
            labels={"x": "Day", "y": "Avg kWh", "color": "kWh"},
        )
        fig_week.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_week, use_container_width=True)

    with col_stats:
        st.markdown("#### 📌 Stats")
        st.metric("Avg Daily",  f"{hist['consumption_kwh'].mean():.2f} kWh")
        st.metric("Max Day",    f"{hist['consumption_kwh'].max():.2f} kWh")
        st.metric("Min Day",    f"{hist['consumption_kwh'].min():.2f} kWh")
        monthly_est = hist["consumption_kwh"].mean() * 30
        st.metric(
            "Est. Monthly Cost",
            f"₦{calculate_cost(monthly_est, st.session_state.cost_per_unit):,.0f}",
        )

    # ── Wattage Table ───────────────────────────────────────────
    st.markdown("#### 🔩 Full Device Report")
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 4 · AI PREDICTIONS
# ───────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🤖 AI-Powered Energy Predictions")
    st.caption(
        "Linear Regression model trained on 30 days of historical consumption data — "
        "features: trend, weekday/weekend pattern, 3-day rolling average."
    )

    hist = st.session_state.historical_data
    _, total_kwh = get_consumption_summary(devices, st.session_state.cost_per_unit)
    units_left = max(0.0, st.session_state.remaining_units - total_kwh)

    # Train model
    model, scaler, metrics, enriched_df = train_prediction_model(hist)
    prediction_df, days_left = predict_unit_depletion(
        model, scaler, enriched_df, units_left
    )

    # ── Model Stats ─────────────────────────────────────────────
    with st.expander("📐 Model Performance Metrics"):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mean Absolute Error (MAE)", f"{metrics['mae']} kWh")
        mc2.metric("R² Score",                 f"{metrics['r2']}")
        mc3.metric("Avg Daily Consumption",    f"{metrics['avg_kwh']} kWh")

    # ── Prediction KPIs ─────────────────────────────────────────
    recharge_date = datetime.now() + timedelta(days=days_left)
    pk1, pk2, pk3 = st.columns(3)
    pk1.metric("📅 Est. Days Until Empty",   f"~{days_left} days")
    pk2.metric("📊 Avg Daily Usage",         f"{metrics['avg_kwh']} kWh")
    pk3.metric("🗓️ Est. Recharge Date",      recharge_date.strftime("%b %d, %Y"))

    if days_left <= 3:
        st.error("⛔ Units will run out in 3 days or less. Recharge immediately!")
    elif days_left <= 7:
        st.warning(f"⚠️ Units estimated to last ~{days_left} days. Plan your recharge.")
    else:
        st.success(f"✅ You have enough units for approximately {days_left} more days.")

    st.divider()

    # ── Historical vs Predicted Chart ───────────────────────────
    st.markdown("#### 📈 Historical vs Predicted Daily Consumption")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=hist["date"], y=hist["consumption_kwh"],
        name="Historical Usage",
        line=dict(color="#007bff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,123,255,0.08)",
    ))
    fig_pred.add_trace(go.Scatter(
        x=prediction_df["date"], y=prediction_df["predicted_kwh"],
        name="AI Prediction",
        line=dict(color="#FF6B35", width=2, dash="dash"),
    ))
    fig_pred.update_layout(
        xaxis_title="Date", yaxis_title="kWh",
        legend=dict(orientation="h"),
        height=380,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Cumulative Depletion Chart ──────────────────────────────
    st.markdown("#### 📉 Predicted Unit Depletion (Next 30 Days)")

    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=prediction_df["date"],
        y=prediction_df["cumulative_kwh"],
        fill="tozeroy",
        fillcolor="rgba(220,53,69,0.15)",
        line=dict(color="#dc3545", width=2),
        name="Cumulative kWh Predicted",
    ))
    fig_dep.add_hline(
        y=units_left,
        line_dash="dash",
        line_color="#28a745",
        line_width=2,
        annotation_text=f"  Your Units: {units_left:.1f} kWh",
        annotation_font_color="#28a745",
    )
    # Mark the depletion point
    if days_left <= 30:
        dep_date = datetime.now().date() + timedelta(days=days_left)
        # Add the line without annotation to avoid Plotly's internal sum() bug
        fig_dep.add_vline(
            x=dep_date,
            line_dash="dot",
            line_color="orange",
        )
        # Add the annotation separately
        fig_dep.add_annotation(
            x=dep_date,
            y=units_left,
            text=f"Empty ~{dep_date.strftime('%b %d')}",
            showarrow=False,
            xanchor="left",
            font=dict(color="orange"),
            yshift=10
        )
    fig_dep.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative kWh Used",
        height=350,
    )
    st.plotly_chart(fig_dep, use_container_width=True)

    # ── AI Recommendations ──────────────────────────────────────
    st.markdown("### 💡 AI-Generated Recommendations")
    tips = generate_recommendations(
        devices,
        metrics["avg_kwh"],
        days_left,
        st.session_state.cost_per_unit,
    )
    for tip in tips:
        st.info(tip)

