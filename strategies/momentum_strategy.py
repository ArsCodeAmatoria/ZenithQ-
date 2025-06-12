"""
Momentum Strategy

A sample strategy that demonstrates the ZenithQ framework.
Uses MACD and RSI indicators to generate buy/sell signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import sys
import os

# Add parent directory to path to import backtesting modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtester import StrategyBase, Order, OrderType, OrderSide, Portfolio
from data_pipeline.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumStrategy(StrategyBase):
    """
    Momentum strategy using MACD and RSI.
    
    Entry Rules:
    - MACD line crosses above signal line (bullish momentum)
    - RSI is below 70 (not overbought)
    - Price is above 20-period EMA (uptrend)
    
    Exit Rules:
    - MACD line crosses below signal line
    - RSI is above 80 (overbought)
    - Stop loss at 2% below entry price
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 ema_period: int = 20,
                 position_size: float = 0.1,  # 10% of portfolio per position
                 stop_loss_pct: float = 0.02):  # 2% stop loss
        """
        Initialize momentum strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI level considered oversold
            rsi_overbought: RSI level considered overbought
            ema_period: Period for EMA calculation
            position_size: Fraction of portfolio to allocate per position
            stop_loss_pct: Stop loss percentage
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_period = ema_period
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        
        # Track entry prices for stop loss
        self.entry_prices = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], portfolio: Portfolio) -> List[Order]:
        """Generate trading signals based on momentum indicators."""
        signals = []
        
        for symbol, df in data.items():
            if len(df) < max(self.rsi_period, self.ema_period, 26):  # Need enough data for MACD
                continue
                
            try:
                # Add technical indicators
                df_with_indicators = TechnicalIndicators.add_all_indicators(df.copy())
                
                # Get latest values
                latest = df_with_indicators.iloc[-1]
                previous = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
                
                current_price = latest['close']
                current_position = portfolio.positions.get(symbol)
                
                # Check for entry signals
                if not current_position or current_position.quantity == 0:
                    entry_signal = self._check_entry_signal(latest, previous)
                    if entry_signal:
                        # Calculate position size
                        portfolio_value = portfolio.get_total_value({symbol: current_price})
                        position_value = portfolio_value * self.position_size
                        quantity = position_value / current_price
                        
                        # Create buy order
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=quantity
                        )
                        signals.append(order)
                        
                        # Record entry price for stop loss
                        self.entry_prices[symbol] = current_price
                        
                        logger.info(f"Generated BUY signal for {symbol} at {current_price:.2f}")
                
                # Check for exit signals
                elif current_position and current_position.quantity > 0:
                    exit_signal = self._check_exit_signal(latest, previous, symbol, current_price)
                    if exit_signal:
                        # Create sell order for entire position
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=current_position.quantity
                        )
                        signals.append(order)
                        
                        # Clean up entry price tracking
                        if symbol in self.entry_prices:
                            del self.entry_prices[symbol]
                        
                        logger.info(f"Generated SELL signal for {symbol} at {current_price:.2f}")
                        
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        return signals
    
    def _check_entry_signal(self, latest: pd.Series, previous: pd.Series) -> bool:
        """
        Check if entry conditions are met.
        
        Args:
            latest: Latest row of data with indicators
            previous: Previous row of data with indicators
            
        Returns:
            True if entry signal is generated
        """
        try:
            # Check if required indicators exist
            required_indicators = ['MACD_12_26_9', 'MACDs_12_26_9', f'rsi_{self.rsi_period}', f'ema_{self.ema_period}']
            for indicator in required_indicators:
                if indicator not in latest.index or pd.isna(latest[indicator]):
                    return False
            
            # MACD bullish crossover
            macd_line = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            prev_macd_line = previous['MACD_12_26_9']
            prev_macd_signal = previous['MACDs_12_26_9']
            
            macd_crossover = (prev_macd_line <= prev_macd_signal) and (macd_line > macd_signal)
            
            # RSI not overbought
            rsi = latest[f'rsi_{self.rsi_period}']
            rsi_condition = rsi < self.rsi_overbought
            
            # Price above EMA (uptrend)
            price = latest['close']
            ema = latest[f'ema_{self.ema_period}']
            trend_condition = price > ema
            
            return macd_crossover and rsi_condition and trend_condition
            
        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
            return False
    
    def _check_exit_signal(self, latest: pd.Series, previous: pd.Series, 
                          symbol: str, current_price: float) -> bool:
        """
        Check if exit conditions are met.
        
        Args:
            latest: Latest row of data with indicators
            previous: Previous row of data with indicators
            symbol: Trading symbol
            current_price: Current price of the symbol
            
        Returns:
            True if exit signal is generated
        """
        try:
            # Check if required indicators exist
            required_indicators = ['MACD_12_26_9', 'MACDs_12_26_9', f'rsi_{self.rsi_period}']
            for indicator in required_indicators:
                if indicator not in latest.index or pd.isna(latest[indicator]):
                    return False
            
            # MACD bearish crossover
            macd_line = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            prev_macd_line = previous['MACD_12_26_9']
            prev_macd_signal = previous['MACDs_12_26_9']
            
            macd_crossover = (prev_macd_line >= prev_macd_signal) and (macd_line < macd_signal)
            
            # RSI overbought
            rsi = latest[f'rsi_{self.rsi_period}']
            rsi_condition = rsi > 80  # More restrictive for exits
            
            # Stop loss
            stop_loss_triggered = False
            if symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                stop_price = entry_price * (1 - self.stop_loss_pct)
                stop_loss_triggered = current_price <= stop_price
                
                if stop_loss_triggered:
                    logger.info(f"Stop loss triggered for {symbol}: {current_price:.2f} <= {stop_price:.2f}")
            
            return macd_crossover or rsi_condition or stop_loss_triggered
            
        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return False
    
    def get_strategy_info(self) -> Dict:
        """Return strategy configuration and parameters."""
        return {
            'name': 'MomentumStrategy',
            'description': 'MACD and RSI momentum strategy',
            'parameters': {
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'ema_period': self.ema_period,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    from backtesting.backtester import Backtester
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Download sample data
    symbol = "AAPL"
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    print(f"Downloading data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if not df.empty:
        # Prepare data
        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        data = {symbol: df}
        
        # Create and run backtest
        strategy = MomentumStrategy()
        backtester = Backtester(initial_cash=100000)
        
        print("Running backtest...")
        results = backtester.run_backtest(strategy, data, start_date, end_date)
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"Volatility: {results.volatility:.2f}%")
        print(f"Win Rate: {results.win_rate:.2f}%")
        print(f"Total Trades: {results.total_trades}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        
    else:
        print("Failed to download data") 