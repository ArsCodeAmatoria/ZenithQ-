"""
Data Sources Manager

Handles data ingestion from multiple sources:
- Alpaca API
- Yahoo Finance
- CCXT for crypto
- News feeds
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import ccxt
from alpaca_trade_api import REST
import aiohttp
import yaml

logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manages multiple data sources for market data ingestion."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data sources from configuration."""
        self.config = self._load_config(config_path)
        self.alpaca_client = None
        self.ccxt_exchanges = {}
        self.session = None
        
        self._initialize_clients()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def _initialize_clients(self):
        """Initialize API clients for data sources."""
        data_sources = self.config.get('data_sources', {})
        
        # Initialize Alpaca
        alpaca_config = data_sources.get('alpaca', {})
        if alpaca_config.get('api_key'):
            self.alpaca_client = REST(
                key_id=alpaca_config['api_key'],
                secret_key=alpaca_config['secret_key'],
                base_url=alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
            )
            logger.info("Alpaca client initialized")
        
        # Initialize CCXT exchanges
        ccxt_config = data_sources.get('ccxt', {})
        for exchange_config in ccxt_config.get('exchanges', []):
            try:
                exchange_class = getattr(ccxt, exchange_config['name'])
                exchange = exchange_class({
                    'apiKey': exchange_config.get('api_key'),
                    'secret': exchange_config.get('secret'),
                    'sandbox': exchange_config.get('sandbox', True),
                    'enableRateLimit': True,
                })
                self.ccxt_exchanges[exchange_config['name']] = exchange
                logger.info(f"CCXT {exchange_config['name']} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_config['name']}: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "alpaca"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from specified source.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
            timeframe: Data timeframe ('1Min', '5Min', '1Hour', '1Day')
            start_date: Start date for historical data
            end_date: End date for historical data
            source: Data source ('alpaca', 'yfinance', 'ccxt')
        
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        try:
            if source == "alpaca" and self.alpaca_client:
                return await self._get_alpaca_data(symbol, timeframe, start_date, end_date)
            elif source == "yfinance":
                return await self._get_yfinance_data(symbol, timeframe, start_date, end_date)
            elif source == "ccxt":
                return await self._get_ccxt_data(symbol, timeframe, start_date, end_date)
            else:
                raise ValueError(f"Unsupported source: {source}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from {source}: {e}")
            return pd.DataFrame()
    
    async def _get_alpaca_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get data from Alpaca API."""
        try:
            bars = self.alpaca_client.get_bars(
                symbol,
                timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                asof=None,
                feed=None,
                page_token=None,
                limit=None,
                adjustment='raw'
            )
            
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Alpaca data: {e}")
            return pd.DataFrame()
    
    async def _get_yfinance_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get data from Yahoo Finance."""
        try:
            # Map timeframes to yfinance intervals
            interval_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d"
            }
            interval = interval_map.get(timeframe, "1d")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if not df.empty:
                df.index.name = 'timestamp'
                df.columns = [col.lower() for col in df.columns]
                # Remove 'dividends' and 'stock splits' columns if they exist
                df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    async def _get_ccxt_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get data from CCXT exchange."""
        try:
            # Use first available exchange
            if not self.ccxt_exchanges:
                raise ValueError("No CCXT exchanges configured")
            
            exchange_name = list(self.ccxt_exchanges.keys())[0]
            exchange = self.ccxt_exchanges[exchange_name]
            
            # Map timeframes to CCXT format
            timeframe_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d"
            }
            tf = timeframe_map.get(timeframe, "1d")
            
            since = int(start_date.timestamp() * 1000)
            limit = 1000
            
            ohlcv = exchange.fetch_ohlcv(symbol, tf, since, limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching CCXT data: {e}")
            return pd.DataFrame()
    
    async def get_live_data(self, symbol: str, source: str = "alpaca") -> Dict:
        """
        Get real-time market data.
        
        Args:
            symbol: Trading symbol
            source: Data source
        
        Returns:
            Dictionary with current market data
        """
        try:
            if source == "alpaca" and self.alpaca_client:
                quote = self.alpaca_client.get_latest_quote(symbol)
                return {
                    'symbol': symbol,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'last': (quote.bid_price + quote.ask_price) / 2,
                    'timestamp': quote.timestamp
                }
            elif source == "yfinance":
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    'symbol': symbol,
                    'last': info.get('regularMarketPrice', 0),
                    'timestamp': datetime.now()
                }
            else:
                raise ValueError(f"Live data not supported for source: {source}")
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return {}
    
    async def get_news_sentiment(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """
        Get news sentiment data for a symbol.
        
        Args:
            symbol: Trading symbol
            hours_back: Hours of historical news to fetch
        
        Returns:
            List of news articles with sentiment scores
        """
        news_data = []
        
        try:
            # This is a placeholder - you would integrate with actual news APIs
            # such as NewsAPI, Alpha Vantage News, or Twitter API
            
            news_config = self.config.get('data_sources', {}).get('news', {})
            
            if news_config.get('news_api_key'):
                # Implement NewsAPI integration
                pass
            
            if news_config.get('twitter_bearer_token'):
                # Implement Twitter API integration
                pass
            
            logger.info(f"Fetched {len(news_data)} news articles for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
        
        return news_data
    
    async def close(self):
        """Close all connections and cleanup resources."""
        if self.session:
            await self.session.close()
        
        for exchange in self.ccxt_exchanges.values():
            if hasattr(exchange, 'close'):
                await exchange.close()
        
        logger.info("Data source connections closed") 