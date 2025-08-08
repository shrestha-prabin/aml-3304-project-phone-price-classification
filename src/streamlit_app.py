import json
import os

import h2o
import openai
import pandas as pd
import streamlit as st

from agent import run_agent
from web_search import web_search

price_descriptions = {
    0: "Low Budget",
    1: "Mid-Range",
    2: "High-End",
    3: "Very High-End",
}


# Initialize H2O and load the trained classification model
h2o.init()
model = h2o.load_model("StackedEnsemble_BestOfFamily_1_AutoML_3_20250807_141418.model")

st.header("📱 Phone Price Range Prediction")
st.caption(
    "Predict the price range of a phone based on its specifications. "
    "This app uses a machine learning model trained on phone specifications data."
)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        screen_size = st.slider(
            "Screen Size (inches)",
            min_value=4.0,
            max_value=7.0,
            value=5.5,
            step=0.1,
            format="%.1f in",
        )
        int_memory = st.slider(
            "Internal Memory (GB)",
            min_value=1,
            max_value=256,
            value=32,
            step=1,
            format="%d GB",
        )
        pc = st.number_input(
            "Primary Camera (MP)",
            min_value=2,
            max_value=20,
            value=12,
            step=1,
        )

        talk_time = st.number_input("Talk Time (hours)", value=10, min_value=1)

    with col2:
        battery_power = st.slider(
            "Battery Power (mAh)",
            min_value=500,
            max_value=3000,
            value=1000,
            step=100,
            format="%.0f mAh",
        )
        ram = st.slider("RAM (MB)", value=2000, min_value=100, max_value=4000, step=4)

        fc = st.number_input(
            "Front Camera (MP)",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
        )
        n_cores = st.number_input(
            "Number of Cores",
            value=4,
            min_value=1,
            max_value=8,
        )

    submitted = st.form_submit_button("Predict Price Range")

battery_power = battery_power / 2

# Estimate screen pixel density based on screen size
px_height = int(screen_size * 320)
px_width = int(screen_size / 1.6 * 320)

# Estimate height width (in cm) based on overall screen size
screen_diagonal_cm = screen_size * 2.54
screen_height_cm = int(px_height / 160 * 2.54)
screen_width_cm = int(px_width / 160 * 2.54)


if submitted:
    # assemble input dict using form values and defaults
    input_vals = {
        "screen_size": screen_size,
        "battery_power": battery_power,
        "ram": ram,
        "int_memory": int_memory,
        "mobile_wt": 150,
        "px_width": px_width,
        "pc": pc,
        "four_g": 1,
        "wifi": 1,
        "blue": 1,
        "clock_speed": 1.5,
        "dual_sim": 0,
        "fc": fc,
        "m_dep": 0.5,
        "n_cores": 4,
        "px_height": px_height,
        "sc_h": screen_width_cm,
        "sc_w": screen_height_cm,
        "talk_time": talk_time,
        "three_g": 1,
        "touch_screen": 1,
    }
    input_df = pd.DataFrame([input_vals])
    hf = h2o.H2OFrame(input_df)
    pred = model.predict(hf)
    result = pred.as_data_frame()
    # st.subheader("Prediction")
    # st.write(result)
    # show only the predicted class label
    predicted = result["predict"].iloc[0]
    st.info(f"#### Prediction: {price_descriptions[predicted].upper()} price range")

    with st.spinner("Searching Web..."):
        # recommendations = run_agent(input_vals)
        recommendations = web_search(input_vals)

        output_cols = st.columns(2)
        for i, phone in enumerate(recommendations):
            with output_cols[i % 2]:
                c = st.container(border=True)
                c.write(f"#### {phone['brand']} {phone['name']}")
                c.write(f"RAM: {phone['specs']['ram']}")
                c.write(f"Storage: {phone['specs']['int_memory']}")
                c.write(f"Battery: {phone['specs']['battery_power']}")
                c.write(f"Size: {phone['specs']['screen_size']}")
