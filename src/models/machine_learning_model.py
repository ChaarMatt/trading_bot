import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    # Example usage
    from src.features.technical_indicators import calculate_rsi, calculate_macd, calculate_sma
    import pandas as pd
    
    # Load data
    df = pd.read_csv('data/processed/btcusd_technical_indicators.csv')
    
    # Prepare features and target
    features = df[['RSI', 'MACD', 'Signal', 'SMA']]
    target = df['Close']
    
    # Create sequences for LSTM
    seq_length = 30
    X_seq, y_seq = create_sequences(features.values, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.2, random_state=42)
    
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

if __name__ == "__main__":
    main()
