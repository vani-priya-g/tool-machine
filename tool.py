import streamlit as st
import base64
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 📄 Page setup
st.set_page_config(page_title="Tool Wear Prediction", page_icon="🔧", layout="wide")

# 🎨 Background image
def set_bg(image_path: str):
    img = open(image_path, "rb").read()
    data = base64.b64encode(img).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpeg;base64,{data}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background-color: transparent !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg(r"D:\cnc\archive (22)\lathe-iron-cnc-machine-steel.jpg")

# ✨ Global styling + custom input box styles
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { color: #ffffff !important; }
  h1, h2, h3 { color: #ffffff !important; }
  [data-testid="stWidgetLabel"] p { color: #ffd54f !important; }
  .block-container { background-color: rgba(0,0,0,0.5) !important; border-radius:8px; padding:1rem !important; }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #1e1e1e !important;
  }
  [data-testid="stSidebar"] .block-container {
    background-color: #1e1e1e !important;
  }
  [data-testid="stSidebar"] * {
    color: #ffffff !important;
  }

  /* Style number inputs and text inputs with specific aria-labels */
  .stNumberInput input[aria-label="feedrate"],
  .stNumberInput input[aria-label="clamp_pressure"] {
    background-color: #333333 !important;
    color: #ffffff !important;
  }
  .stNumberInput input[aria-label="material"] {
    background-color: #444444 !important;
    color: #ffffff !important;
  }
  /* Add more if you want specific colors per input */

  /* Style generic text_input by label */
  .stTextInput input[aria-label="Enter any text"] {
    background-color: #333333 !important;
    color: #ffffff !important;
  }

  /* Highlight selectbox div */
  div[data-baseweb="select"] > div {
    background-color: #333333 !important;
    color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

# 🧠 Load model & scaler
@st.cache_resource
def load_resources(model_path: str, scaler_path: str):
    m = load_model(model_path)
    with open(scaler_path, "rb") as f:
        sc = pickle.load(f)
    return m, sc

model, scaler = load_resources(
    r"D:\cnc\archive (22)\Final_model.h5",
    r"D:\cnc\archive (22)\scaler.pkl"
)

# 🔢 Features list
selected_columns = [
    "material", "feedrate", "clamp_pressure",
    "X1_ActualPosition", "Y1_ActualPosition", "Z1_ActualPosition",
    "X1_CurrentFeedback", "Y1_CurrentFeedback",
    "M1_CURRENT_FEEDRATE", "X1_DCBusVoltage",
    "X1_OutputPower", "Y1_OutputPower", "S1_OutputPower"
]

# 🧭 Sidebar navigation
menu = ["Home", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🔬 Tool Wear Prediction using LSTM")
    st.write("""
Welcome! This app uses an LSTM model to predict:
- Tool Condition (Good, Worn, Damaged)  
- Machining Finalization  
- Visual Inspection Outcome  
Enter your parameters in the sidebar and click **Predict**.
""")
else:
    st.title("🎨 Tool Wear Prediction")
    st.sidebar.header("📥 Input Parameters")

    # Use number_inputs with exact labels matching CSS selectors
    user_input = {
        "material": st.sidebar.number_input("material", value=0),
        "feedrate": st.sidebar.number_input("feedrate", value=0.0),
        "clamp_pressure": st.sidebar.number_input("clamp_pressure", value=0.0),
        **{
            feat: st.sidebar.number_input(feat, value=0.0)
            for feat in selected_columns[3:]
        }
    }

    df_in = pd.DataFrame([user_input])
    scaled = scaler.transform(df_in)
    seq = np.tile(scaled, (10,1)).reshape(1, 10, len(selected_columns))

    if st.sidebar.button("Predict"):
        with st.spinner("Predicting..."):
            tc_out, mf_out, vi_out = model.predict(seq)
        labels = {0: "Good", 1: "Worn", 2: "Damaged"}

        st.subheader("🔍 Results")
        st.write(f"**🛠 Tool Condition:** {labels[np.argmax(tc_out[0])]}")
        st.write(f"**🔄 Machining Finalized:** {'✅ Yes' if mf_out[0][0]>0.5 else '❌ No'}")
        st.write(f"**👀 Visual Inspection Passed:** {'✅ Yes' if vi_out[0][0]>0.5 else '❌ No'}")
        st.success("✅ Prediction Complete!")
