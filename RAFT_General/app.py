# SAVE AS: RAFT_General/app.py
import streamlit as st
import sys
import os
import time

# Add current directory to path so we can import your backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import YOUR working Crypto Backend
try:
    from crypto_experiment.crypto_backend_api import get_live_crypto_forecast
except ImportError:
    st.error("Could not find crypto_backend_api.py. Check your folder structure!")

# --- DUMMY RIVER BACKEND (Since your friend has the real code) ---
def dummy_river_forecast():
    """Fakes the river data so your UI doesn't crash."""
    return 5.20, 5.20, "⚖️ AWAITING FRIEND'S DATA"

# --- UI DESIGN START ---
st.set_page_config(page_title="RAFT Global Monitor", layout="wide")

st.title("🌍 RAFT: Dual-Domain Extreme Event Monitor")
st.markdown("Retrieval-Augmented Forecasting of Time Series (Live Inference Dashboard)")
st.markdown("---")

# Create a button to trigger the live APIs
if st.button("🔄 Fetch Live Global Data"):
    with st.spinner("Querying Global APIs and RAFT Memory Banks..."):
        
        # 1. Fetch data from your real backend, and the dummy backend
        btc_curr, btc_pred, btc_status = get_live_crypto_forecast()
        riv_curr, riv_pred, riv_status = dummy_river_forecast()
        
        # 2. Create the Split Screen
        col1, col2 = st.columns(2)
        
        # --- LEFT COLUMN: FINANCE (YOUR REAL AI) ---
        with col1:
            st.header("📈 Financial Domain (Bitcoin)")
            st.markdown("**Source:** Live Binance REST API")
            
            if btc_curr is None:
                st.error(f"API Error: {btc_status}")
            else:
                # Display metrics nicely
                st.metric(label="Current BTC Price", value=f"${btc_curr:,.2f}")
                
                # Calculate the predicted change
                diff = btc_pred - btc_curr
                st.metric(label="RAFT Forecast (Next 5 Mins)", value=f"${btc_pred:,.2f}", delta=f"${diff:,.2f}")
                
                # Status Warning Box
                if "CRASH" in btc_status:
                    st.error(f"**SYSTEM ALERT:** {btc_status}")
                elif "SPIKE" in btc_status:
                    st.success(f"**SYSTEM ALERT:** {btc_status}")
                else:
                    st.info(f"**SYSTEM ALERT:** {btc_status}")

        # --- RIGHT COLUMN: HYDROLOGY (PLACEHOLDER) ---
        with col2:
            st.header("🌊 Hydrology Domain (River)")
            st.markdown("**Source:** USGS Colorado River API (Pending)")
            
            st.metric(label="Current Water Level", value=f"{riv_curr:.2f} ft")
            st.metric(label="RAFT Forecast (Next 6 Hours)", value=f"{riv_pred:.2f} ft", delta="0.00 ft")
            st.warning(f"**SYSTEM ALERT:** {riv_status}")