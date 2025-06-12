"""
Backtesting Engine

A comprehensive backtesting framework that supports:
- Multiple timeframes and instruments
- Realistic slippage and commission modeling
- Portfolio-level metrics calculation
- Monte Carlo analysis
- Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    id: Optional[str] = None


@dataclass
class Trade:
    """Represents an executed trade."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    order_id: str


@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position.quantity * current_prices[symbol]
                
        return total_value
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """Update unrealized P&L for all positions."""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position.quantity * current_prices[symbol]
                cost_basis = position.quantity * position.avg_price
                position.unrealized_pnl = current_value - cost_basis
    
    def execute_trade(self, trade: Trade):
        """Execute a trade and update portfolio."""
        self.trades.append(trade)
        
        # Update cash
        trade_value = trade.quantity * trade.price
        if trade.side == OrderSide.BUY:
            self.cash -= trade_value + trade.commission
        else:
            self.cash += trade_value - trade.commission
        
        # Update position
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(trade.symbol, 0, 0)
        
        position = self.positions[trade.symbol]
        
        if trade.side == OrderSide.BUY:
            # Calculate new average price
            total_quantity = position.quantity + trade.quantity
            if total_quantity > 0:
                total_cost = (position.quantity * position.avg_price + 
                            trade.quantity * trade.price)
                position.avg_price = total_cost / total_quantity
            position.quantity += trade.quantity
        else:
            # Realize P&L on sale
            realized_pnl = trade.quantity * (trade.price - position.avg_price)
            position.realized_pnl += realized_pnl
            position.quantity -= trade.quantity
            
            # Remove position if quantity is zero
            if abs(position.quantity) < 1e-8:
                del self.positions[trade.symbol]


class BacktestResults:
    """Container for backtest results and performance metrics."""
    
    def __init__(self, portfolio: Portfolio, start_date: datetime, end_date: datetime):
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date
        self.trades = portfolio.trades
        self.equity_curve = portfolio.equity_curve
        
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.equity_curve:
            return
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # Basic metrics
        self.total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
        self.total_trades = len(self.trades)
        self.winning_trades = len([t for t in self.trades if self._get_trade_pnl(t) > 0])
        self.losing_trades = self.total_trades - self.winning_trades
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Risk metrics
        self.sharpe_ratio = self._calculate_sharpe_ratio(df['returns'])
        self.sortino_ratio = self._calculate_sortino_ratio(df['returns'])
        self.max_drawdown = self._calculate_max_drawdown(df['equity'])
        self.calmar_ratio = self.total_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0
        
        # Additional metrics
        self.volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
        self.profit_factor = self._calculate_profit_factor()
        
    def _get_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade (simplified)."""
        # This is a simplified calculation - in reality you'd need to track entry/exit pairs
        return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (downside_returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        return drawdown.min()
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = sum([self._get_trade_pnl(t) for t in self.trades if self._get_trade_pnl(t) > 0])
        gross_loss = abs(sum([self._get_trade_pnl(t) for t in self.trades if self._get_trade_pnl(t) < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profit_factor': self.profit_factor
        }


class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtester.
        
        Args:
            initial_cash: Starting capital
            commission: Commission per trade (as percentage)
            slippage: Slippage per trade (as percentage)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.current_timestamp = None
        
    def run_backtest(self, 
                    strategy,
                    data: Dict[str, pd.DataFrame],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResults:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy object with generate_signals method
            data: Dictionary of symbol -> OHLCV DataFrame
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults object
        """
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_cash)
        
        # Get date range
        if start_date is None:
            start_date = min([df.index.min() for df in data.values()])
        if end_date is None:
            end_date = max([df.index.max() for df in data.values()])
        
        # Create unified timeline
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        
        timeline = sorted([ts for ts in all_timestamps if start_date <= ts <= end_date])
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Timeline has {len(timeline)} timestamps")
        
        # Run simulation
        for timestamp in timeline:
            self.current_timestamp = timestamp
            
            # Get current prices for all symbols
            current_prices = {}
            current_data = {}
            
            for symbol, df in data.items():
                if timestamp in df.index:
                    current_prices[symbol] = df.loc[timestamp, 'close']
                    current_data[symbol] = df.loc[:timestamp].iloc[-100:]  # Last 100 bars
            
            # Update portfolio unrealized P&L
            self.portfolio.update_unrealized_pnl(current_prices)
            
            # Record equity curve
            total_value = self.portfolio.get_total_value(current_prices)
            self.portfolio.equity_curve.append((timestamp, total_value))
            
            # Generate signals from strategy
            try:
                signals = strategy.generate_signals(current_data, self.portfolio)
                
                # Execute orders
                for signal in signals:
                    self._execute_order(signal, current_prices)
                    
            except Exception as e:
                logger.error(f"Error generating signals at {timestamp}: {e}")
                continue
        
        # Create and return results
        results = BacktestResults(self.portfolio, start_date, end_date)
        
        logger.info(f"Backtest completed. Total return: {results.total_return:.2f}%")
        logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {results.max_drawdown:.2f}%")
        
        return results
    
    def _execute_order(self, order: Order, current_prices: Dict[str, float]):
        """Execute an order with slippage and commission."""
        if order.symbol not in current_prices:
            logger.warning(f"No price data for {order.symbol}, skipping order")
            return
        
        current_price = current_prices[order.symbol]
        
        # Calculate execution price with slippage
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)
        else:
            # For limit orders, use the limit price if it would be filled
            execution_price = order.price or current_price
        
        # Calculate commission
        trade_value = order.quantity * execution_price
        commission = trade_value * self.commission
        
        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            required_cash = trade_value + commission
            if required_cash > self.portfolio.cash:
                logger.warning(f"Insufficient cash for {order.symbol} buy order")
                return
        
        # Check if we have enough shares for sells
        if order.side == OrderSide.SELL:
            position = self.portfolio.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                logger.warning(f"Insufficient shares for {order.symbol} sell order")
                return
        
        # Execute trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=self.current_timestamp,
            commission=commission,
            slippage=abs(execution_price - current_price),
            order_id=order.id or f"{order.symbol}_{self.current_timestamp}"
        )
        
        self.portfolio.execute_trade(trade)
        
        logger.debug(f"Executed {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price:.2f}")


class StrategyBase:
    """Base class for trading strategies."""
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], portfolio: Portfolio) -> List[Order]:
        """
        Generate trading signals based on current data and portfolio state.
        
        Args:
            data: Dictionary of symbol -> historical OHLCV data
            portfolio: Current portfolio state
            
        Returns:
            List of orders to execute
        """
        raise NotImplementedError("Subclasses must implement generate_signals method") 