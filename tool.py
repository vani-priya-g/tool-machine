import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import base64

# Set page config
st.set_page_config(page_title="Tool Wear Prediction", layout="wide")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("Final_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Set background image
# Set background image with dark overlay for better text readability
def set_bg(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            color: white !important;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_bg("lathe-iron-cnc-machine-steel.jpg")  # Make sure this image is in the same folder

# Sidebar navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.selectbox("Choose Page", ["Home", "Prediction"])

# Page content
if page == "Home":
    st.title("🔧 Tool Wear Prediction using LSTM")
    st.write("""
Welcome to the CNC Tool Wear Prediction App.  
This model predicts:
- 🛠 Tool Condition (Worn / Unworn)  
- 🔧 Machining Finalized  
- 👁️ Visual Inspection Passed  
Enter your machining parameters in the sidebar and click **Predict**.
""")

elif page == "Prediction":
    st.title("🎯 Tool Wear Prediction")
    st.sidebar.header("📥 Input Parameters")

    # Input parameters from sidebar
    material = st.sidebar.number_input("Material", min_value=0, max_value=5, value=0)
    feedrate = st.sidebar.number_input("Feedrate", step=0.01)
    clamp_pressure = st.sidebar.number_input("Clamp Pressure", step=0.01)
    X1_ActualPosition = st.sidebar.number_input("X1 Actual Position", step=0.01)
    Y1_ActualPosition = st.sidebar.number_input("Y1 Actual Position", step=0.01)
    Z1_ActualPosition = st.sidebar.number_input("Z1 Actual Position", step=0.01)
    X1_CurrentFeedback = st.sidebar.number_input("X1 Current Feedback", step=0.01)
    Y1_CurrentFeedback = st.sidebar.number_input("Y1 Current Feedback", step=0.01)
    M1_CURRENT_FEEDRATE = st.sidebar.number_input("M1 Current Feedrate", step=0.01)
    X1_DCBusVoltage = st.sidebar.number_input("X1 DCBus Voltage", step=0.01)
    X1_OutputPower = st.sidebar.number_input("X1 Output Power", step=0.01)
    Y1_OutputPower = st.sidebar.number_input("Y1 Output Power", step=0.01)
    S1_OutputPower = st.sidebar.number_input("S1 Output Power", step=0.01)

    if st.sidebar.button("Predict"):
        # Collect inputs
        input_data = np.array([[
            material, feedrate, clamp_pressure,
            X1_ActualPosition, Y1_ActualPosition, Z1_ActualPosition,
            X1_CurrentFeedback, Y1_CurrentFeedback, M1_CURRENT_FEEDRATE,
            X1_DCBusVoltage, X1_OutputPower, Y1_OutputPower, S1_OutputPower
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # LSTM expects shape: (batch, time_steps, features)
        input_series = np.repeat(input_scaled, 10, axis=0).reshape(1, 10, 13)

        # Predict
        tc_out, mf_out, vi_out = model.predict(input_series)
        labels = {0: "Unworn", 1: "Worn", 2: "Damaged"}  # Adjust based on training

        # Output
        st.subheader("🔍 Prediction Results")
        st.write(f"**🛠 Tool Condition:** `{labels[np.argmax(tc_out[0])]}`")
        st.write(f"**🔧 Machining Finalized:** {'✅ Yes' if mf_out[0][0] > 0.5 else '❌ No'}")
        st.write(f"**👁️ Visual Inspection Passed:** {'✅ Yes' if vi_out[0][0] > 0.5 else '❌ No'}")
        st.success("✅ Prediction complete!")