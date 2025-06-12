#!/usr/bin/env python3
"""
ZenithQ Demo Script

Demonstrates the core functionality of the ZenithQ trading system:
1. Data acquisition
2. Technical indicator calculation
3. Strategy backtesting
4. Performance analysis
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_pipeline.data_sources import DataSourceManager
from data_pipeline.indicators import TechnicalIndicators
from strategies.momentum_strategy import MomentumStrategy
from backtesting.backtester import Backtester


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('zenithq_demo.log')
        ]
    )


async def demo_data_pipeline():
    """Demonstrate data acquisition and preprocessing."""
    print("🔄 Demonstrating Data Pipeline...")
    
    # Initialize data source manager
    data_manager = DataSourceManager()
    
    # Fetch historical data
    symbol = "AAPL"
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    print(f"📊 Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
    
    # Try Yahoo Finance (most reliable for demo)
    df = await data_manager.get_historical_data(
        symbol=symbol,
        timeframe="1Day",
        start_date=start_date,
        end_date=end_date,
        source="yfinance"
    )
    
    if df.empty:
        print("❌ Failed to fetch data")
        return None
    
    print(f"✅ Downloaded {len(df)} trading days")
    print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Add technical indicators
    print("🔧 Adding technical indicators...")
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    
    indicators_added = len(df_with_indicators.columns) - len(df.columns)
    print(f"✅ Added {indicators_added} technical indicators")
    
    # Show sample of important indicators
    important_indicators = TechnicalIndicators.get_feature_importance_indicators(df_with_indicators)
    print(f"🎯 Key indicators: {', '.join(important_indicators[:5])}...")
    
    await data_manager.close()
    return {symbol: df_with_indicators}


def demo_strategy_backtest(data):
    """Demonstrate strategy backtesting."""
    print("\n🚀 Demonstrating Strategy Backtesting...")
    
    # Initialize strategy
    strategy = MomentumStrategy(
        rsi_period=14,
        rsi_overbought=70,
        ema_period=20,
        position_size=0.2,  # 20% per position for demo
        stop_loss_pct=0.03  # 3% stop loss
    )
    
    print(f"📋 Strategy: {strategy.get_strategy_info()['name']}")
    print(f"📝 Description: {strategy.get_strategy_info()['description']}")
    
    # Initialize backtester
    backtester = Backtester(
        initial_cash=100000,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Run backtest
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    print(f"⏱️  Running backtest from {start_date.date()} to {end_date.date()}")
    
    results = backtester.run_backtest(
        strategy=strategy,
        data=data,
        start_date=start_date,
        end_date=end_date
    )
    
    return results


def display_results(results):
    """Display backtest results in a nice format."""
    print("\n" + "="*60)
    print("📊 ZENITHQ BACKTEST RESULTS")
    print("="*60)
    
    metrics = results.to_dict()
    
    # Performance metrics
    print("\n💰 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"Total Return:      {metrics['total_return']:>8.2f}%")
    print(f"Volatility:        {metrics['volatility']:>8.2f}%")
    print(f"Max Drawdown:      {metrics['max_drawdown']:>8.2f}%")
    
    # Risk-adjusted metrics
    print("\n📈 RISK-ADJUSTED METRICS")
    print("-" * 30)
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}")
    print(f"Calmar Ratio:      {metrics['calmar_ratio']:>8.2f}")
    
    # Trading statistics
    print("\n📊 TRADING STATISTICS")
    print("-" * 30)
    print(f"Total Trades:      {metrics['total_trades']:>8.0f}")
    print(f"Win Rate:          {metrics['win_rate']:>8.2f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:>8.2f}")
    
    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("-" * 30)
    
    if metrics['total_return'] > 10:
        print("🟢 Strong positive returns")
    elif metrics['total_return'] > 0:
        print("🟡 Positive returns")
    else:
        print("🔴 Negative returns")
    
    if metrics['sharpe_ratio'] > 1.0:
        print("🟢 Excellent risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 0.5:
        print("🟡 Good risk-adjusted returns")
    else:
        print("🔴 Poor risk-adjusted returns")
    
    if abs(metrics['max_drawdown']) < 10:
        print("🟢 Low drawdown - well controlled risk")
    elif abs(metrics['max_drawdown']) < 20:
        print("🟡 Moderate drawdown")
    else:
        print("🔴 High drawdown - significant risk")
    
    print("\n" + "="*60)


def demo_strategy_optimization():
    """Demonstrate strategy parameter optimization."""
    print("\n🔧 STRATEGY OPTIMIZATION SUGGESTIONS")
    print("-" * 40)
    print("Based on your brainstorm, here are the next steps:")
    print()
    print("1. 🧬 Genetic Algorithm Optimization:")
    print("   - Evolve RSI period (10-21)")
    print("   - Optimize position sizing (5%-25%)")
    print("   - Find optimal stop loss (1%-5%)")
    print()
    print("2. 🤖 Machine Learning Enhancement:")
    print("   - Train XGBoost on indicator features")
    print("   - Use LSTM for price prediction")
    print("   - Apply reinforcement learning")
    print()
    print("3. 📊 Multi-timeframe Analysis:")
    print("   - Combine 1h, 4h, and 1d signals")
    print("   - Use higher timeframe for trend")
    print("   - Lower timeframe for entry timing")
    print()
    print("4. 🛡️ Risk Management:")
    print("   - Portfolio-level position sizing")
    print("   - Correlation-based allocation")
    print("   - Dynamic volatility targeting")


async def main():
    """Main demo function."""
    print("🌟 Welcome to ZenithQ - AI Trading System Demo")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Step 1: Data Pipeline Demo
        data = await demo_data_pipeline()
        if not data:
            print("❌ Demo failed - could not fetch data")
            return
        
        # Step 2: Strategy Backtest Demo
        results = demo_strategy_backtest(data)
        
        # Step 3: Display Results
        display_results(results)
        
        # Step 4: Optimization Suggestions
        demo_strategy_optimization()
        
        print("\n🎉 Demo completed successfully!")
        print("👉 Next: Implement ML models and genetic algorithms")
        print("📚 See README.md for full setup instructions")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        print("💡 Tip: Make sure you have internet connection for data download")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 