"""
Technical Indicators Module

Implements a comprehensive set of technical indicators for the trading system:
- Trend indicators (EMA, SMA, MACD)
- Momentum indicators (RSI, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (VWAP, Volume Profile)
- Price action patterns
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Comprehensive technical indicators calculator."""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all available technical indicators to a dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        try:
            # Trend Indicators
            result_df = TechnicalIndicators.add_trend_indicators(result_df)
            
            # Momentum Indicators  
            result_df = TechnicalIndicators.add_momentum_indicators(result_df)
            
            # Volatility Indicators
            result_df = TechnicalIndicators.add_volatility_indicators(result_df)
            
            # Volume Indicators
            result_df = TechnicalIndicators.add_volume_indicators(result_df)
            
            # Support/Resistance
            result_df = TechnicalIndicators.add_support_resistance(result_df)
            
            # Price Patterns
            result_df = TechnicalIndicators.add_price_patterns(result_df)
            
            logger.info(f"Added {len(result_df.columns) - len(df.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            
        return result_df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        result_df = df.copy()
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            result_df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            result_df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
        # MACD
        macd_data = ta.macd(df['close'])
        if macd_data is not None:
            result_df = pd.concat([result_df, macd_data], axis=1)
        
        # Parabolic SAR
        psar = ta.psar(df['high'], df['low'], df['close'])
        if psar is not None:
            result_df = pd.concat([result_df, psar], axis=1)
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None and len(ichimoku) > 0:
            result_df = pd.concat([result_df, ichimoku[0]], axis=1)
        
        # Average Directional Index (ADX)
        adx_data = ta.adx(df['high'], df['low'], df['close'])
        if adx_data is not None:
            result_df = pd.concat([result_df, adx_data], axis=1)
        
        return result_df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators."""
        result_df = df.copy()
        
        # RSI
        for period in [14, 21]:
            result_df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None:
            result_df = pd.concat([result_df, stoch], axis=1)
        
        # Williams %R
        willr = ta.willr(df['high'], df['low'], df['close'])
        if willr is not None:
            result_df['willr'] = willr
        
        # Commodity Channel Index (CCI)
        cci = ta.cci(df['high'], df['low'], df['close'])
        if cci is not None:
            result_df['cci'] = cci
        
        # Rate of Change (ROC)
        for period in [10, 20]:
            roc = ta.roc(df['close'], length=period)
            if roc is not None:
                result_df[f'roc_{period}'] = roc
        
        # Money Flow Index (MFI)
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        if mfi is not None:
            result_df['mfi'] = mfi
        
        return result_df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        result_df = df.copy()
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        if bb is not None:
            result_df = pd.concat([result_df, bb], axis=1)
        
        # Average True Range (ATR)
        atr = ta.atr(df['high'], df['low'], df['close'])
        if atr is not None:
            result_df['atr'] = atr
        
        # Donchian Channels
        donchian = ta.donchian(df['high'], df['low'])
        if donchian is not None:
            result_df = pd.concat([result_df, donchian], axis=1)
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'])
        if kc is not None:
            result_df = pd.concat([result_df, kc], axis=1)
        
        # Historical Volatility
        result_df['volatility_10'] = df['close'].pct_change().rolling(10).std() * np.sqrt(252)
        result_df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        return result_df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        result_df = df.copy()
        
        # Volume Weighted Average Price (VWAP)
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap is not None:
            result_df['vwap'] = vwap
        
        # On Balance Volume (OBV)
        obv = ta.obv(df['close'], df['volume'])
        if obv is not None:
            result_df['obv'] = obv
        
        # Accumulation/Distribution Line
        ad = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        if ad is not None:
            result_df['ad'] = ad
        
        # Chaikin Money Flow (CMF)
        cmf = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        if cmf is not None:
            result_df['cmf'] = cmf
        
        # Volume SMA
        for period in [10, 20, 50]:
            result_df[f'volume_sma_{period}'] = ta.sma(df['volume'], length=period)
        
        # Volume Rate of Change
        volume_roc = ta.roc(df['volume'], length=10)
        if volume_roc is not None:
            result_df['volume_roc'] = volume_roc
        
        return result_df
    
    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels."""
        result_df = df.copy()
        
        # Pivot Points
        high_roll = df['high'].rolling(window=window, center=True).max()
        low_roll = df['low'].rolling(window=window, center=True).min()
        
        result_df['resistance'] = np.where(df['high'] == high_roll, df['high'], np.nan)
        result_df['support'] = np.where(df['low'] == low_roll, df['low'], np.nan)
        
        # Forward fill support/resistance levels
        result_df['resistance'] = result_df['resistance'].fillna(method='ffill')
        result_df['support'] = result_df['support'].fillna(method='ffill')
        
        # Distance from support/resistance
        result_df['distance_to_resistance'] = (result_df['resistance'] - df['close']) / df['close']
        result_df['distance_to_support'] = (df['close'] - result_df['support']) / df['close']
        
        return result_df
    
    @staticmethod
    def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition."""
        result_df = df.copy()
        
        # Candlestick patterns (simplified)
        result_df['doji'] = TechnicalIndicators.detect_doji(df)
        result_df['hammer'] = TechnicalIndicators.detect_hammer(df)
        result_df['engulfing_bullish'] = TechnicalIndicators.detect_bullish_engulfing(df)
        result_df['engulfing_bearish'] = TechnicalIndicators.detect_bearish_engulfing(df)
        
        # Gap detection
        result_df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        result_df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        # Inside/Outside bars
        result_df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                                  (df['low'] > df['low'].shift(1))).astype(int)
        result_df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & 
                                   (df['low'] < df['low'].shift(1))).astype(int)
        
        return result_df
    
    @staticmethod
    def detect_doji(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Detect Doji candlestick pattern."""
        body = abs(df['close'] - df['open'])
        range_val = df['high'] - df['low']
        return (body / range_val < threshold).astype(int)
    
    @staticmethod
    def detect_hammer(df: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern."""
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        
        return ((lower_shadow > 2 * body) & 
                (upper_shadow < body) & 
                (body > 0)).astype(int)
    
    @staticmethod
    def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Detect Bullish Engulfing pattern."""
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)
        
        return ((prev_close < prev_open) &  # Previous candle bearish
                (df['close'] > df['open']) &  # Current candle bullish
                (df['open'] < prev_close) &   # Current open below prev close
                (df['close'] > prev_open)).astype(int)  # Current close above prev open
    
    @staticmethod
    def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Detect Bearish Engulfing pattern."""
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)
        
        return ((prev_close > prev_open) &  # Previous candle bullish
                (df['close'] < df['open']) &  # Current candle bearish
                (df['open'] > prev_close) &   # Current open above prev close
                (df['close'] < prev_open)).astype(int)  # Current close below prev open
    
    @staticmethod
    def calculate_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom and advanced indicators."""
        result_df = df.copy()
        
        # Z-Score for mean reversion
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            result_df[f'zscore_{period}'] = (df['close'] - mean) / std
        
        # Price position within the day's range
        result_df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Intraday strength
        result_df['intraday_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Volume-Price Trend (VPT)
        result_df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # True Strength Index (TSI)
        momentum = df['close'].diff()
        smoothed_momentum = momentum.ewm(span=25).mean().ewm(span=13).mean()
        smoothed_abs_momentum = abs(momentum).ewm(span=25).mean().ewm(span=13).mean()
        result_df['tsi'] = 100 * (smoothed_momentum / smoothed_abs_momentum)
        
        return result_df
    
    @staticmethod
    def get_feature_importance_indicators(df: pd.DataFrame) -> List[str]:
        """
        Return list of most important indicators for ML models.
        
        Returns:
            List of column names for key indicators
        """
        key_indicators = [
            'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr',
            'vwap', 'obv', 'ema_20', 'sma_50', 'volume_sma_20',
            'volatility_20', 'distance_to_resistance', 'distance_to_support',
            'zscore_20', 'price_position', 'tsi'
        ]
        
        # Filter to only include indicators that exist in the dataframe
        return [col for col in key_indicators if col in df.columns] 