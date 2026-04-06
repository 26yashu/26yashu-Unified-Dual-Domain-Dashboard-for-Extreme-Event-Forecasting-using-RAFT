# SAVE AS: RAFT_General/app.py
import streamlit as st
import sys
import os
import time
import pandas as pd

# Add current directory to path so we can import your backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import YOUR working Crypto Backend
try:
    from crypto_experiment.crypto_backend_api import get_live_crypto_forecast
except ImportError:
    st.error("Could not find crypto_backend_api.py. Check your folder structure!")

# Import YOUR FRIEND'S working River Backend
try:
    from river_experiment.backend_api import get_live_river_forecast
except ImportError:
    st.error("Could not find river_experiment/backend_api.py. Did you copy your friend's folder correctly?")

# --- UI DESIGN START ---
st.set_page_config(page_title="RAFT Global Monitor", layout="wide")

st.title("🌍 RAFT: Dual-Domain Extreme Event Monitor")
st.markdown("Retrieval-Augmented Forecasting of Time Series (Live Inference Dashboard)")
st.markdown("---")

# 1. NEW: The Auto-Update Toggle Switch
col_toggle, col_spacer = st.columns([1, 3])
with col_toggle:
    auto_refresh = st.toggle("🔴 Enable Live Auto-Update")
    if auto_refresh:
        st.caption("Refreshing every 60 seconds...")

# 2. Create a container so the UI doesn't jump around when refreshing
dashboard_container = st.empty()

with dashboard_container.container():
    # Fetch data from BOTH real backends
    btc_curr, btc_pred, btc_status = get_live_crypto_forecast()
    riv_curr, riv_pred, riv_status = get_live_river_forecast()
    
    # Create the Split Screen
    col1, col2 = st.columns(2)
    
    # --- LEFT COLUMN: FINANCE (YOURS) ---
    with col1:
        st.header("📈 Financial Domain (Bitcoin)")
        st.markdown("**Source:** Live Binance REST API")
        
        if btc_curr is None:
            st.error(f"API Error: {btc_status}")
        else:
            # Metrics
            st.metric(label="Current BTC Price", value=f"${btc_curr:,.2f}")
            diff = btc_pred - btc_curr
            st.metric(label="RAFT Forecast (Next 5 Mins)", value=f"${btc_pred:,.2f}", delta=f"${diff:,.2f}")
            
            # Status Box
            if "CRASH" in btc_status:
                st.error(f"**SYSTEM ALERT:** {btc_status}")
            elif "SPIKE" in btc_status:
                st.success(f"**SYSTEM ALERT:** {btc_status}")
            else:
                st.info(f"**SYSTEM ALERT:** {btc_status}")
                
            # Live Trajectory Graph (Bitcoin Orange)
            st.markdown("##### 🚀 Forecast Trajectory")
            chart_data = pd.DataFrame(
                {"BTC Price (USD)": [btc_curr, btc_pred]},
                index=["Now", "Forecast (+5m)"]
            )
            st.line_chart(chart_data, color="#F7931A")

    # --- RIGHT COLUMN: HYDROLOGY (FRIEND'S) ---
    with col2:
        st.header("🌊 Hydrology Domain (River)")
        st.markdown("**Source:** USGS Colorado River API")
        
        if riv_curr is None:
            st.error(f"API Error: {riv_status}")
        else:
            # Metrics
            st.metric(label="Current Water Level", value=f"{riv_curr:.2f} ft")
            diff_riv = riv_pred - riv_curr
            st.metric(label="RAFT Forecast (Next 6 Hours)", value=f"{riv_pred:.2f} ft", delta=f"{diff_riv:.2f} ft")
            
            # Status Box
            if "FLOOD" in riv_status:
                st.error(f"**SYSTEM ALERT:** {riv_status}")
            else:
                st.info(f"**SYSTEM ALERT:** {riv_status}")
                
            # Live Trajectory Graph (Water Blue)
            st.markdown("##### 🚀 Forecast Trajectory")
            chart_data_riv = pd.DataFrame(
                {"Water Level (ft)": [riv_curr, riv_pred]},
                index=["Now", "Forecast (+6h)"]
            )
            st.line_chart(chart_data_riv, color="#00A4E4")

# 3. The Loop Logic
if auto_refresh:
    time.sleep(60) # Pauses for 60 seconds so it doesn't spam the APIs
    st.rerun()     # Tells the app to reload and fetch fresh data