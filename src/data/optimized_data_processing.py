import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def process_data_chunk(data_chunk):
    # Vectorized operations for data processing
    data_chunk['RSI'] = calculate_rsi(data_chunk)
    macd, signal = calculate_macd(data_chunk)
    data_chunk['MACD'] = macd
    data_chunk['Signal'] = signal
    data_chunk['SMA'] = calculate_sma(data_chunk)
    return data_chunk

def calculate_rsi(data, period=14):
    delta = data['Close'].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean().abs()
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_sma(data, period=20):
    return data['Close'].rolling(window=period).mean()

def parallel_process_data(data, num_jobs=-1):
    chunks = np.array_split(data, num_jobs)
    processed_chunks = Parallel(n_jobs=num_jobs)(delayed(process_data_chunk)(chunk) for chunk in chunks)
    return pd.concat(processed_chunks, axis=0)

def main():
    from src.data.gather_market_data import fetch_market_data
    df = fetch_market_data("BTCUSD", "2020-01-01", "2024-12-31")
    processed_df = parallel_process_data(df)
    processed_df.to_csv('data/processed/optimized_btcusd_data.csv', index=True)

if __name__ == "__main__":
    import pandas as pd
