import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("📈 Advanced Stock Dashboard")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv('nsestocksindia.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")

stocks = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

steps = st.sidebar.slider("Forecast Days", 1, 30, 10)

# ---------- FILTER DATA ----------
st_data = df[df['stock'] == selected_stock].copy()

# Ensure OHLC exists
required_cols = ['Open', 'High', 'Low', 'Close']
for col in required_cols:
    if col not in st_data.columns:
        st.error(f"Missing column: {col}")
        st.stop()

# ---------- RSI FUNCTION ----------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------- MACD FUNCTION ----------
def compute_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Calculate indicators
st_data['RSI'] = compute_rsi(st_data)
st_data['MACD'], st_data['Signal'] = compute_macd(st_data)

# ---------- METRICS ----------
col1, col2 = st.columns(2)
col1.metric("📌 Latest Price", round(st_data['Close'].iloc[-1], 2))
col2.metric("📊 RSI", round(st_data['RSI'].iloc[-1], 2))

# ---------- CANDLESTICK CHART ----------
st.subheader("📉 Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=st_data.index,
    open=st_data['Open'],
    high=st_data['High'],
    low=st_data['Low'],
    close=st_data['Close'],
    name='Candlestick'
)])

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# ---------- RSI CHART ----------
st.subheader("📊 RSI Indicator")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=st_data.index, y=st_data['RSI'], name='RSI'))

fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

fig_rsi.update_layout(template="plotly_dark")

st.plotly_chart(fig_rsi, use_container_width=True)

# ---------- MACD CHART ----------
st.subheader("📊 MACD Indicator")

fig_macd = go.Figure()

fig_macd.add_trace(go.Scatter(x=st_data.index, y=st_data['MACD'], name='MACD'))
fig_macd.add_trace(go.Scatter(x=st_data.index, y=st_data['Signal'], name='Signal'))

fig_macd.update_layout(template="plotly_dark")

st.plotly_chart(fig_macd, use_container_width=True)

# ---------- ARIMA PREDICTION ----------
st.subheader("🔮 Price Prediction")

if st.button("Predict Future Prices"):
    with st.spinner("Training ARIMA Model..."):
        try:
            model = ARIMA(st_data['Close'], order=(5, 0, 0))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=steps)

            future_dates = pd.date_range(
                start=st_data.index[-1],
                periods=steps + 1,
                freq='B'
            )[1:]

            # Plot forecast
            fig_pred = go.Figure()

            fig_pred.add_trace(go.Scatter(
                x=st_data.index,
                y=st_data['Close'],
                name="Actual"
            ))

            fig_pred.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                name="Forecast",
                line=dict(dash='dash')
            ))

            fig_pred.update_layout(template="plotly_dark")

            st.plotly_chart(fig_pred, use_container_width=True)

            # Forecast table
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Price": forecast
            }).set_index("Date")

            st.success("✅ Prediction Completed")
            st.dataframe(forecast_df)

        except Exception as e:
            st.error(f"Error: {e}")
