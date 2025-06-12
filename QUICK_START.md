# 🚀 ZenithQ Quick Start Guide

Get your AI trading system up and running in 5 minutes!

## 📋 Prerequisites

- Python 3.8+
- Internet connection (for data download)
- Basic understanding of trading concepts

## ⚡ Installation

1. **Clone and navigate to the project**
```bash
cd ZenithQ-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the demo**
```bash
python scripts/demo_backtest.py
```

## 🎯 What the Demo Does

The demo script showcases all core ZenithQ features:

1. **📊 Data Pipeline**: Downloads AAPL stock data using Yahoo Finance
2. **🔧 Technical Indicators**: Adds 50+ indicators (MACD, RSI, Bollinger Bands, etc.)
3. **🤖 Strategy Testing**: Runs a momentum strategy using MACD/RSI signals
4. **📈 Performance Analysis**: Calculates Sharpe ratio, drawdown, win rate, etc.

## 📊 Expected Output

```
🌟 Welcome to ZenithQ - AI Trading System Demo
==================================================
🔄 Demonstrating Data Pipeline...
📊 Fetching AAPL data from 2023-01-01 to 2024-01-01
✅ Downloaded 252 trading days
📈 Price range: $124.17 - $199.62
🔧 Adding technical indicators...
✅ Added 45+ technical indicators
🎯 Key indicators: rsi_14, macd, bb_upper, bb_lower, atr...

🚀 Demonstrating Strategy Backtesting...
📋 Strategy: MomentumStrategy
📝 Description: MACD and RSI momentum strategy
⏱️  Running backtest from 2023-01-01 to 2024-01-01

============================================================
📊 ZENITHQ BACKTEST RESULTS
============================================================

💰 PERFORMANCE METRICS
------------------------------
Total Return:      15.32%
Volatility:        22.45%
Max Drawdown:      -8.12%

📈 RISK-ADJUSTED METRICS
------------------------------
Sharpe Ratio:       1.24
Sortino Ratio:      1.78
Calmar Ratio:       1.89

📊 TRADING STATISTICS
------------------------------
Total Trades:         12
Win Rate:          66.67%
Profit Factor:       2.13

🎯 PERFORMANCE ASSESSMENT
------------------------------
🟢 Strong positive returns
🟢 Excellent risk-adjusted returns
🟢 Low drawdown - well controlled risk
```

## 🛠️ Next Steps

### 1. Customize Configuration
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys and preferences
```

### 2. Try Different Strategies
```python
from strategies.momentum_strategy import MomentumStrategy

# Customize parameters
strategy = MomentumStrategy(
    rsi_period=21,           # Change RSI period
    position_size=0.15,      # Adjust position sizing
    stop_loss_pct=0.025      # Modify stop loss
)
```

### 3. Add Your Own Strategy
```python
from backtesting.backtester import StrategyBase, Order, OrderSide, OrderType

class MyStrategy(StrategyBase):
    def generate_signals(self, data, portfolio):
        signals = []
        # Your strategy logic here
        return signals
```

### 4. Enable Live Data Sources
Edit `config/config.yaml`:
```yaml
data_sources:
  alpaca:
    api_key: "YOUR_ALPACA_KEY"
    secret_key: "YOUR_ALPACA_SECRET"
```

## 🧬 Advanced Features (Coming Soon)

### Genetic Algorithm Optimization
```python
from ml_engine.genetic_algorithm import GeneticOptimizer

optimizer = GeneticOptimizer()
best_params = optimizer.optimize_strategy(strategy, data)
```

### Machine Learning Models
```python
from ml_engine.models import ModelFactory

model = ModelFactory.create_model("xgboost")
model.train(features, labels)
predictions = model.predict(new_features)
```

### Reinforcement Learning
```python
from ml_engine.rl_environment import TradingEnvironment

env = TradingEnvironment(data)
agent = create_dqn_agent(env)
agent.train(episodes=1000)
```

## 🔧 Troubleshooting

### "No module named 'pandas_ta'"
```bash
pip install pandas-ta
```

### "Failed to fetch data"
- Check internet connection
- Yahoo Finance may be temporarily unavailable
- Try different symbol: Edit `demo_backtest.py` line 48

### "ImportError"
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Documentation

- **Architecture**: See `README.md` for full system overview
- **Configuration**: All options in `config/config.example.yaml`
- **API Reference**: Check docstrings in each module
- **Examples**: More strategies in `strategies/` folder

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📞 Support

- 🐛 Issues: Create GitHub issue
- 💬 Discussions: GitHub Discussions
- 📧 Email: [Your email]

---

**Happy Trading with ZenithQ! 🚀📈**

*Remember: Past performance doesn't guarantee future results. Always test thoroughly before risking real money.* 