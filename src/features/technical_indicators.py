import pandas as pd

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI) for the given data."""
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
    """Calculate Moving Average Convergence Divergence (MACD) for the given data."""
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return macd, signal

def calculate_sma(data, period=20):
    """Calculate Simple Moving Average (SMA) for the given data."""
    return data['Close'].rolling(window=period).mean()

def main():
    # Example usage
    from src.data.gather_market_data import fetch_market_data
    import pandas as pd
    
    ticker = "BTCUSD"
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    df = fetch_market_data(ticker, start_date, end_date)
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df)
    macd, signal = calculate_macd(df)
    df['MACD'] = macd
    df['Signal'] = signal
    df['SMA'] = calculate_sma(df)
    
    # Save processed data
    df.to_csv('data/processed/btcusd_technical_indicators.csv', index=True)

if __name__ == "__main__":
    main()
