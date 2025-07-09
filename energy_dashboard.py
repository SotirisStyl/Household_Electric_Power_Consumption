import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# -------- Load and preprocess data with caching --------
@st.cache_data
@st.cache_data
def load_data(nrows=100000):
    url = "https://drive.google.com/uc?id=1islxIxYjfOt8TXuc5kO5LDGjhA9yt7Ex"
    local_csv = "energy_data.csv"
    # Download from Google Drive if not present
    if not os.path.exists(local_csv):
        try:
            import gdown
        except ImportError:
            st.error("gdown is required to download from Google Drive. Please install it with 'pip install gdown'.")
            st.stop()
        gdown.download(url, local_csv, quiet=False)
    df = pd.read_csv(local_csv, sep=';', na_values='?', low_memory=False, nrows=nrows)
    
    # Print columns to debug KeyError
    print("Columns loaded:", df.columns.tolist())
    # Adjust column names if needed
    if 'Date' not in df.columns or 'Time' not in df.columns:
        # Try lower case or strip spaces
        df.columns = df.columns.str.strip()
    if 'Date' not in df.columns or 'Time' not in df.columns:
        st.error(f"CSV columns are: {df.columns.tolist()}. 'Date' and 'Time' columns not found.")
        st.stop()
    df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df['dt'] = df['Date_Time']
    df.set_index('dt', inplace=True)
    df.dropna(inplace=True)

    # Ensure index is a DatetimeIndex for time features
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Time features using index since 'dt' is now the index
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    df['date'] = df.index.date
    df['week'] = df.index.isocalendar().week
    df['month'] = df.index.to_period('M').astype(str)

    return df
df = load_data()
st.info("Loaded a subset of the data for performance. Adjust 'nrows' in load_data() if needed.")

# -------- User inputs --------
start_date = st.date_input("Start Date", df.index.min().date())
end_date = st.date_input("End Date", df.index.max().date())
hour_to_view = st.slider("Select Hour of Day", 0, 23, 12)

filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
hourly_filtered = filtered[filtered['hour'] == hour_to_view]

st.write("Sample of Filtered Data:")
st.dataframe(hourly_filtered.reset_index()[['dt', 'Global_active_power']].head(10))

# -------- Line chart for hourly filtered data --------
fig = px.line(hourly_filtered.reset_index(), x='dt', y='Global_active_power',
              title=f"Power Consumption at Hour {hour_to_view}")
st.plotly_chart(fig)

# -------- Sidebar Info --------
st.sidebar.header("Data Overview")
st.sidebar.write(f"Total Records: {len(df)}")
st.sidebar.write(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")

# -------- Interval-based average plot --------
interval = st.selectbox("Select Interval", ["Day", "Week", "Month"])

@st.cache_data
def get_avg_power(df, interval):
    if interval == "Day":
        avg_df = df.groupby('date')['Global_active_power'].mean().reset_index()
        avg_df.columns = ['Date', 'Average_Global_active_power']
        x_col = 'Date'
    elif interval == "Week":
        avg_df = df.groupby('week')['Global_active_power'].mean().reset_index()
        avg_df.columns = ['Week', 'Average_Global_active_power']
        x_col = 'Week'
    else:  # Month
        avg_df = df.groupby('month')['Global_active_power'].mean().reset_index()
        avg_df.columns = ['Month', 'Average_Global_active_power']
        x_col = 'Month'
    return avg_df, x_col

avg_df, x_col = get_avg_power(filtered, interval)
st.subheader(f"Average Global Active Power by {interval}")
st.dataframe(avg_df.head())

fig_avg = px.line(avg_df, x=x_col, y='Average_Global_active_power',
                  title=f"Average Global Active Power per {interval}")
st.plotly_chart(fig_avg)

# -------- Sub-metering stats --------
st.header("Sub-Metering Averages")
st.write(f"Sub-Metering 1: {df['Sub_metering_1'].mean():.2f} kW")
st.write(f"Sub-Metering 2: {df['Sub_metering_2'].mean():.2f} kW")
st.write(f"Sub-Metering 3: {df['Sub_metering_3'].mean():.2f} kW")

# -------- Heatmap --------
heatmap_df = df.groupby(['date', 'hour'])['Global_active_power'].mean().reset_index()
heatmap_data = heatmap_df.pivot(index='date', columns='hour', values='Global_active_power')

fig_heatmap = px.imshow(heatmap_data,
                        labels=dict(x="Hour", y="Date", color="Power (kW)"),
                        title="Daily Hourly Power Heatmap",
                        color_continuous_scale='Viridis')
st.plotly_chart(fig_heatmap)

# -------- Voltage distribution --------
fig_voltage = px.histogram(df, x='Voltage', nbins=50, title="Voltage Distribution")
st.plotly_chart(fig_voltage)

# -------- Realtime Simulation --------
st.header("Realtime Simulation")
simulation_time = st.slider("Select Time of Day", 0, 1439, 720)
simulation_df = df[df['total_minutes'] == simulation_time]

fig_simulation = px.line(simulation_df.reset_index(), x='dt', y='Global_active_power',
                         title=f"Power Usage at {simulation_time} minutes")
st.plotly_chart(fig_simulation)

# -------- Anomaly Detection --------
st.header("Anomaly Detection")
threshold = df['Global_active_power'].quantile(0.99)
anomalies = df[df['Global_active_power'] > threshold]
st.write(f"Anomalies Detected: {len(anomalies)}")

fig_anomaly = px.line(df.reset_index(), x='dt', y='Global_active_power',
                      title="Global Active Power with Anomalies")
fig_anomaly.add_scatter(x=anomalies.index, y=anomalies['Global_active_power'],
                        mode='markers', marker=dict(color='red', size=6),
                        name='Anomaly')
st.plotly_chart(fig_anomaly)

# -------- Interactive Filters --------
st.header("Interactive Filters")
st.sidebar.header("Interactive Filters")

df_filtered = df.reset_index()  # get 'dt' back as column

# Sidebar filter values
hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))
day_of_week = st.sidebar.multiselect("Day of Week", options=df_filtered['dt'].dt.day_name().unique(),
                                     default=df_filtered['dt'].dt.day_name().unique())
month = st.sidebar.multiselect("Month", options=df_filtered['dt'].dt.month_name().unique(),
                               default=df_filtered['dt'].dt.month_name().unique())
year = st.sidebar.selectbox("Year", options=sorted(df_filtered['dt'].dt.year.unique()))
voltage_clean = df_filtered['Voltage'].dropna()
v_min, v_max = float(voltage_clean.min()), float(voltage_clean.max())
voltage_range = st.sidebar.slider("Voltage Range", v_min, v_max, (v_min, v_max))

sub_metering_zone = st.sidebar.multiselect(
    "Sub-Metering Zone",
    options=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
    default=['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
)

# Filtering
filtered = df_filtered[
    (df_filtered['hour'].between(hour_of_day[0], hour_of_day[1])) &
    (df_filtered['dt'].dt.day_name().isin(day_of_week)) &
    (df_filtered['dt'].dt.month_name().isin(month)) &
    (df_filtered['dt'].dt.year == year) &
    (df_filtered['Voltage'].between(voltage_range[0], voltage_range[1]))
]

if sub_metering_zone:
    sub_mask = False
    for zone in sub_metering_zone:
        sub_mask |= (filtered[zone] > 0)
    filtered = filtered[sub_mask]

# Output
if filtered.empty:
    st.warning("⚠️ No data matched your filter selection. Try relaxing the filters.")
else:
    st.subheader("Filtered Data Preview")
    st.write(f"Rows matched: {len(filtered)}")
    st.dataframe(filtered)


# -------- Machine Learning: Predict Global Active Power --------
st.header("Machine Learning: Predict Global Active Power")

# 1. Define the prediction goal
st.markdown("**Goal:** Predict 'Global_active_power' using available features.")

# 2. Load + clean data (already loaded as df, drop NaNs for ML)
ml_df = df.dropna(subset=[
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]).copy()

# 3. Feature engineering
ml_df.index = pd.to_datetime(ml_df.index)  # Ensure DatetimeIndex
ml_df['day_of_week'] = pd.to_datetime(ml_df.index).dayofweek
ml_df['is_weekend'] = ml_df['day_of_week'].isin([5, 6]).astype(int)

features = [
    'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'hour', 'minute', 'day_of_week', 'is_weekend'
]

# 4. Split features (X) and target (y)
X = ml_df[features]
y = ml_df['Global_active_power']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse:.3f}")
st.write(f"Mean Absolute Error: {mae:.3f}")
st.write(f"R² Score: {r2:.3f}")

# 8. Visualize results
fig_ml = px.scatter(
    x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
    title="Actual vs Predicted Global Active Power"
)
fig_ml.add_shape(
    type="line", x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash")
)
st.plotly_chart(fig_ml)

# 9. Add live predictions
st.subheader("Live Prediction")
with st.form("live_pred_form"):
    st.markdown("Enter feature values to predict Global Active Power:")
    col1, col2, col3 = st.columns(3)
    with col1:
        inp_reactive = st.number_input("Global Reactive Power", value=float(X['Global_reactive_power'].mean()))
        inp_voltage = st.number_input("Voltage", value=float(X['Voltage'].mean()))
        inp_intensity = st.number_input("Global Intensity", value=float(X['Global_intensity'].mean()))
    with col2:
        inp_sub1 = st.number_input("Sub_metering_1", value=float(X['Sub_metering_1'].mean()))
        inp_sub2 = st.number_input("Sub_metering_2", value=float(X['Sub_metering_2'].mean()))
        inp_sub3 = st.number_input("Sub_metering_3", value=float(X['Sub_metering_3'].mean()))
    with col3:
        inp_hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
        inp_minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
        inp_dayofweek = st.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=0)
        inp_isweekend = st.selectbox("Is Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    submitted = st.form_submit_button("Predict")
    if submitted:
        input_arr = np.array([[inp_reactive, inp_voltage, inp_intensity,
                               inp_sub1, inp_sub2, inp_sub3,
                               inp_hour, inp_minute, inp_dayofweek, inp_isweekend]])
        pred = model.predict(input_arr)[0]
        st.success(f"Predicted Global Active Power: {pred:.3f} kW")

# 10. Save model
joblib.dump(model, "global_active_power_gbr_model.joblib")
st.info("Trained model saved as 'global_active_power_gbr_model.joblib'.")
