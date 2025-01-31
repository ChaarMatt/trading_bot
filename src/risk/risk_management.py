import pandas as pd

class RiskManager:
    def __init__(self, risk_profile='medium', max_drawdown=0.20):
        self.risk_profile = risk_profile
        self.max_drawdown = max_drawdown
        self.position_size = 0
        
    def calculate_position_size(self, portfolio_value, stop_loss_pips, risk_percentage=2):
        """Calculate position size based on risk tolerance"""
        risk_amount = portfolio_value * (risk_percentage / 100)
        self.position_size = risk_amount / stop_loss_pips
        return self.position_size
    
    def set_stop_loss(self, price, atr, risk_profile='medium'):
        """Set dynamic stop loss based on ATR"""
        if risk_profile == 'low':
            stop_loss_pips = price - (atr * 0.5)
        elif risk_profile == 'medium':
            stop_loss_pips = price - atr
        else:  # high risk
            stop_loss_pips = price - (atr * 2)
        return stop_loss_pips
    
    def set_take_profit(self, price, atr, risk_profile='medium'):
        """Set take profit based on ATR"""
        if risk_profile == 'low':
            take_profit_pips = price + (atr * 0.5)
        elif risk_profile == 'medium':
            take_profit_pips = price + atr
        else:  # high risk
            take_profit_pips = price + (atr * 2)
        return take_profit_pips
    
    def check_max_drawdown(self, current_drawdown):
        """Monitor maximum allowable drawdown"""
        if current_drawdown > self.max_drawdown:
            return False
        return True
    
    def generate_trading_report(self, trades):
        """Generate performance report"""
        total_profit = sum(trades['profit'])
        num_trades = len(trades)
        win_rate = len([t for t in trades if t['profit'] > 0]) / num_trades
        avg_profit = total_profit / num_trades
        
        report = {
            'total_profit': total_profit,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'average_profit': avg_profit
        }
        return report

def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)"""
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def main():
    # Example usage
    from src.data.gather_market_data import fetch_market_data
    import pandas as pd
    
    # Load data
    ticker = "BTCUSD"
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    df = fetch_market_data(ticker, start_date, end_date)
    
    # Calculate ATR
    atr = calculate_atr(df)
    
    # Initialize risk manager
    risk_manager = RiskManager(risk_profile='medium')
    
    # Example trade parameters
    price = df['Close'][-1]
    stop_loss = risk_manager.set_stop_loss(price, atr[-1])
    take_profit = risk_manager.set_take_profit(price, atr[-1])
    
    print(f"Stop Loss: {stop_loss}")
    print(f"Take Profit: {take_profit}")

if __name__ == "__main__":
    main()
