import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, epochs=50):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
        
    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)

def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length, :-1])
        y.append(dataset[i+seq_length, -1])
    return np.array(X), np.array(y)

def main():
    from src.data.gather_market_data import fetch_market_data
    api_key = "RfuJFUgdUgBXbRRAlQsTZG2ykkaFcv6q"
    symbols = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    df = fetch_market_data(symbols, start_date, end_date, api_key)
    features = df[symbols[0]][['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Calculate indicators
    features['RSI'] = features['Close'].rolling(14).apply(lambda x: calculate_rsi(x))
    features['MACD'], features['Signal'] = calculate_macd(features['Close'])
    features['SMA'] = features['Close'].rolling(20).mean()
    
    # Create sequences for LSTM
    seq_length = 30
    X_seq, y_seq = create_sequences(features.values, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features.values, features['Close'].values, test_size=0.2, random_state=42)
    
    # Train models
    lstm_model = LSTMModel((seq_length, features.shape[1]))
    lstm_model.train(X_seq, y_seq)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    
    # Make predictions
    lstm_predictions = lstm_model.predict(X_seq[-10:])
    rf_predictions = rf_model.predict(X_test[-10:])
    
    # Evaluate models
    print("LSTM MSE:", mean_squared_error(y_seq[-10:], lstm_predictions.flatten()))
    print("Random Forest MSE:", mean_squared_error(y_test[-10:], rf_predictions))

def calculate_rsi(data, period=14):
    delta = data.diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean().abs()
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

if __name__ == "__main__":
    main()
