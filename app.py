import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Title
st.title("📈 Stock Price Prediction using ARIMA")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('nsestocksindia.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# Show available stocks
stocks = df['stock'].unique()
st.write("Available Stocks:", stocks)

# Dropdown for stock selection
selected_stock = st.selectbox("Select Stock", stocks)

# Forecast steps input
steps = st.slider("Select number of days to forecast", 1, 30, 10)

# Filter selected stock
st_data = df[df['stock'] == selected_stock][['Close']].copy()

# Compute returns (optional)
st_data['Returns'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# Show raw data
if st.checkbox("Show Data"):
    st.write(st_data.tail())

# Train ARIMA model
if st.button("Predict"):
    try:
        model = ARIMA(st_data['Close'], order=(5, 0, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps)

        # Future dates
        future_dates = pd.date_range(
            start=st_data.index[-1],
            periods=steps + 1,
            freq='B'
        )[1:]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st_data['Close'], label="Actual Prices")
        ax.plot(future_dates, forecast, linestyle='dashed', color='red', label="Predicted Prices")

        ax.set_title(f"{selected_stock} Price Prediction")
        ax.legend()
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # Show forecast values
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": forecast
        }).set_index("Date")

        st.write("📊 Forecast Data")
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Error: {e}")
