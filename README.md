# ⚡ Energy Consumption Dashboard

An interactive Streamlit dashboard for visualizing, filtering, and predicting household electric power consumption data.

🔗 **Try it live**: [https://electricpowerconsumption.streamlit.app/](https://electricpowerconsumption.streamlit.app/)

## Features

- 📅 Filter data by date and hour
- 📈 View line plots and average power usage
- 🌡️ Analyze voltage and sub-metering stats
- 🧊 Explore a heatmap of power usage
- ⚠️ Detect high-usage anomalies
- 🤖 Predict power consumption with a machine learning model
- 🧠 Live input predictions

## Run Locally

```bash
pip install streamlit pandas plotly scikit-learn joblib gdown
streamlit run energy_dashboard.py
