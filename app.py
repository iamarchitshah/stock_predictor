import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import *

st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")

st.title("🧠 AI-based Stock Price Predictor")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL or TCS.NS):", "AAPL")

model_type = st.selectbox("Choose Model:", ["LSTM", "GRU", "RNN", "Linear Regression", "Random Forest"])
predict_button = st.button("Predict Next 20 Days")

if predict_button:
    st.info("Fetching and processing data...")
    data = fetch_data(ticker)
    window_size = 60

    if model_type in ["LSTM", "GRU", "RNN"]:
        X, y, scaler = preprocess_data(data)
        input_shape = (X.shape[1], X.shape[2])
        
        if model_type == "LSTM":
            model = create_lstm_model(input_shape)
        elif model_type == "GRU":
            model = create_gru_model(input_shape)
        else:
            model = create_rnn_model(input_shape)
        
        preds = train_and_predict_dl(model, X, y, scaler)
    else:
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100)
        preds = train_and_predict_ml(model, data.values)

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=20)
    pred_df = pd.DataFrame(preds, columns=["Open", "Close"])
    pred_df["Date"] = future_dates
    pred_df.set_index("Date", inplace=True)

    st.success("Prediction complete!")
    st.subheader("📊 Predicted Prices")
    st.dataframe(pred_df)

    st.subheader("📈 Visual Comparison")
    fig, ax = plt.subplots()
    ax.plot(data.index[-100:], data["Close"].values[-100:], label="Past Close")
    ax.plot(pred_df.index, pred_df["Close"], label="Predicted Close", color="red")
    ax.legend()
    st.pyplot(fig)

    st.download_button("📥 Download Predictions as CSV", pred_df.to_csv().encode(), "predictions.csv", "text/csv")
