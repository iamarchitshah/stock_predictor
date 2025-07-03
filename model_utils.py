import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN
import yfinance as yf

def fetch_data(ticker):
    data = yf.download(ticker, start="2015-01-01", progress=False)
    return data[['Open', 'Close']]

def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(64))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, input_shape=input_shape))
    model.add(SimpleRNN(64))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict_dl(model, X, y, scaler, days=20, epochs=50):
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    last_input = X[-1].reshape(1, X.shape[1], X.shape[2])
    predicted = []

    for _ in range(days):
        pred = model.predict(last_input)[0]
        predicted.append(pred)
        last_input = np.append(last_input[:, 1:, :], [[pred]], axis=1)

    predicted = scaler.inverse_transform(predicted)
    return predicted

def train_and_predict_ml(model, data, window_size=60, days=20):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []

    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i].flatten())
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    model.fit(X, y)

    last_input = scaled[-window_size:].flatten().reshape(1, -1)
    predicted = []

    for _ in range(days):
        pred = model.predict(last_input)[0]
        predicted.append(pred)
        last_input = np.append(last_input[:, 2:], pred).reshape(1, -1)

    predicted = scaler.inverse_transform(predicted)
    return predicted
