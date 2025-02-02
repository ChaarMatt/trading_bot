"""
Gathers historical market data from Polygon.io。
"""

import pandas as pd
import os
import requests
import json

def fetch_minute_data(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetches minute data for a given symbol。
    
    Args:
        symbol (str): The ticker symbol of the instrument。
        start_date (str): Start date in YYYY-MM-DD format。
        end_date (str): End date in YYYY-MM-DD format。
        api_key (str): Polygon API key。
        
    Returns:
        pd.DataFrame: DataFrame containing minute OHLCV data。
    """
    try:
        base_url = f"https://api.polygon.io/v2/ticks/stocks/{symbol}?from={start_date}T00:00:00&to={end_date}T23:59:59&limit=100000&apiKey={api_key}"
        response = requests.get(base_url)
        response.raise_for_status()
        data = json.loads(response.content)
        if 'ticks' in data and len(data['ticks']) > 0:
            df = pd.DataFrame(data['ticks'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        else:
            print(f"Error fetching data for {symbol}: No results found")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Request error for {symbol}: {str(e)}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"JSON error for {symbol}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_market_data(symbols: list, start_date: str, end_date: str, api_key: str) -> dict:
    """
    Fetches market data for multiple symbols。
    
    Args:
        symbols (list): List of ticker symbols。
        start_date (str): Start date in YYYY-MM-DD format。
        end_date (str): End date in YYYY-MM-DD format。
        api_key (str): Polygon API key。
        
    Returns:
        dict: Dictionary mapping symbols to their respective DataFrames。
    """
    data_dict = {}
    for symbol in symbols:
        df = fetch_minute_data(symbol, start_date, end_date, api_key)
        if not df.empty:
            data_dict[symbol] = df
        else:
            print(f"No data available for symbol {symbol}")
    return data_dict

if __name__ == "__main__":
    # Example usage
    api_key = "RfuJFUgdUgBXbRRAlQsTZG2ykkaFcv6q"
    symbols = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    data = fetch_market_data(symbols, start_date, end_date, api_key)
    for symbol, df in data.items():
        print(f"Data for {symbol}:")
        print(df.head())
