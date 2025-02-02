import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, risk_profile='medium', max_drawdown=0.20):
        self.risk_profile = risk_profile
        self.max_drawdown = max_drawdown
        self.position_size = 0
        self.volatility = None
        
    def calculate_position_size(self, portfolio_value, stop_loss_pips, risk_percentage=2):
        """Calculate position size based on risk tolerance and volatility"""
        risk_amount = portfolio_value * (risk_percentage / 100)
        self.position_size = risk_amount / stop_loss_pips
        return self.position_size
    
    def set_stop_loss(self, price, atr, risk_profile='medium'):
        """Dynamically adjust stop loss based on volatility"""
        if risk_profile == 'low':
            stop_loss_pips = price - (atr * 0.5)
        elif risk_profile == 'medium':
            stop_loss_pips = price - atr
        else:
            stop_loss_pips = price - (atr * 2)
        return stop_loss_pips
    
    def set_take_profit(self, price, atr, risk_profile='medium'):
        """Dynamically adjust take profit based on volatility"""
        if risk_profile == 'low':
            take_profit_pips = price + (atr * 0.5)
        elif risk_profile == 'medium':
            take_profit_pips = price + atr
        else:
            take_profit_pips = price + (atr * 2)
        return take_profit_pips
    
    def calculate_volatility(self, data, window=20):
        """Calculate volatility using ATR"""
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift(1))
        low_close = abs(data['Low'] - data['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        self.volatility = atr[-1]
        return atr
    
    def check_max_drawdown(self, current_drawdown):
        """Monitor and limit maximum allowable drawdown"""
        if current_drawdown > self.max_drawdown:
            return False
        return True
    
    def generate_trading_report(self, trades):
        """Generate comprehensive trading performance report"""
        total_profit = sum(trades['profit'])
        num_trades = len(trades)
        win_rate = sum(1 for t in trades if t['profit'] > 0) / num_trades
        avg_profit = total_profit / num_trades
        return {
            'total_profit': total_profit,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'average_profit': avg_profit,
            'risk_adjusted_return': total_profit / self.position_size
        }

def calculate_sharpe_ratio(returns, risk_free_rate=0, window=252):
    """Calculate Sharpe Ratio"""
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(window)

def main():
    from src.data.gather_market_data import fetch_market_data
    df = fetch_market_data("BTCUSD", "2020-01-01", "2024-12-31")
    
    # Calculate volatility
    atr = RiskManager().calculate_volatility(df)
    
    # Initialize risk manager
    risk_manager = RiskManager(risk_profile='medium')
    price = df['Close'][-1]
    stop_loss = risk_manager.set_stop_loss(price, atr[-1])
    take_profit = risk_manager.set_take_profit(price, atr[-1])
    
    print(f"Volatility (ATR): {atr[-1]:.2f}")
    print(f"Stop Loss: {stop_loss}")
    print(f"Take Profit: {take_profit}")

if __name__ == "__main__":
    main()
