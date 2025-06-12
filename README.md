================================================================================
                                                                              
    ███████ ███████ ███    ██ ██ ████████ ██   ██  ██████                    
         ██ ██      ████   ██ ██    ██    ██   ██ ██    ██                   
      ████  █████   ██ ██  ██ ██    ██    ███████ ██    ██                   
     ██     ██      ██  ██ ██ ██    ██    ██   ██ ██ ▄▄ ██                   
    ███████ ███████ ██   ████ ██    ██    ██   ██  ██████                    
                                                     ▀▀                       
                                                                              
           AI-Powered Trading Strategy Discovery System                       
                                                                              
================================================================================

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ArsCodeAmatoria/ZenithQ-?style=for-the-badge&logo=github)](https://github.com/ArsCodeAmatoria/ZenithQ-)
[![GitHub Issues](https://img.shields.io/github/issues/ArsCodeAmatoria/ZenithQ-?style=for-the-badge&logo=github)](https://github.com/ArsCodeAmatoria/ZenithQ-/issues)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

**A self-improving AI that learns, tests, and deploys trading strategies using advanced machine learning techniques**

![Image](https://github.com/user-attachments/assets/3c0cb7e0-2a61-40d9-bf46-df95f60ffea5)

## Overview

ZenithQ is a comprehensive machine learning system that automatically discovers, tests, and improves trading strategies across different timeframes. It combines supervised learning, reinforcement learning, and genetic algorithms to create a self-evolving trading ecosystem.

## Architecture

### Core Components
- **Data Pipeline**: Multi-source market data ingestion and preprocessing
- **ML Engine**: Traditional ML, Deep Learning, and RL models
- **Strategy Evolution**: Genetic algorithms for strategy optimization
- **Testing Lab**: Backtesting and paper trading validation
- **Dashboard**: Real-time monitoring and strategy management
- **Risk Management**: Dynamic position sizing and risk controls

### Supported Strategy Types
- Price Action (pin bars, engulfing, breakouts)
- Order Flow (VWAP, delta volume)
- Mean Reversion (Bollinger Bands, Z-score)
- Momentum (MACD cross, RSI overbought/oversold)
- News-based/Sentiment-based
- Arbitrage/Pairs trading
- ML Discovered Patterns

## Tech Stack

| Component | Technology |
|-----------|------------|
| Core Language | Python, Rust (execution) |
| ML Framework | PyTorch, JAX, Scikit-learn |
| RL Framework | Stable Baselines3, Ray RLlib |
| Backend | FastAPI |
| Frontend | Next.js + TailwindCSS + Shadcn/UI |
| Database | PostgreSQL, Redis |
| Cloud Storage | S3 |
| Hosting | Vercel (frontend), AWS (backend) |

## Project Structure

```
zenithq/
├── backend/           # FastAPI backend services
├── frontend/          # Next.js dashboard
├── ml_engine/         # Core ML and RL algorithms
├── data_pipeline/     # Data ingestion and preprocessing
├── backtesting/       # Testing and validation framework
├── strategies/        # Strategy implementations
├── risk_management/   # Risk controls and position sizing
├── config/           # Configuration files
├── tests/            # Test suites
└── docs/             # Documentation
```

## Quick Start

1. **Clone and setup environment**
```bash
git clone <repo-url>
cd zenithq
pip install -r requirements.txt
```

2. **Configure data sources**
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your API keys
```

3. **Start the backend**
```bash
cd backend
uvicorn main:app --reload
```

4. **Start the dashboard**
```bash
cd frontend
npm install
npm run dev
```

5. **Run your first backtest**
```bash
python scripts/run_backtest.py --strategy momentum_macd --symbol AAPL
```

## Self-Learning Loop

The system continuously:
1. Evaluates current strategies
2. Drops underperformers
3. Generates new strategies using RL/GA
4. Backtests and scores new strategies
5. Auto-deploys top performers

## Safety Features

- Dynamic position sizing (Kelly criterion)
- Auto-stop on drawdown limits
- Real-time VaR monitoring
- Anomaly detection alerts
- Paper trading validation

## Key Metrics

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.