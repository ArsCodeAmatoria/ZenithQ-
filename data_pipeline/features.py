"""
Feature Engineering Module

Advanced feature engineering for ML models including:
- Lagged features
- Rolling statistics
- Interaction features
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for trading ML models."""
    
    @staticmethod
    def create_lagged_features(df: pd.DataFrame, 
                              columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features added
        """
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                result_df[lag_col] = df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, 
                               columns: List[str],
                               windows: List[int]) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling statistics
                result_df[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
                result_df[f"{col}_roll_std_{window}"] = df[col].rolling(window).std()
                result_df[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()
                result_df[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()
                result_df[f"{col}_roll_skew_{window}"] = df[col].rolling(window).skew()
                result_df[f"{col}_roll_kurt_{window}"] = df[col].rolling(window).kurt()
                
                # Position within rolling window
                roll_min = df[col].rolling(window).min()
                roll_max = df[col].rolling(window).max()
                result_df[f"{col}_roll_position_{window}"] = (
                    (df[col] - roll_min) / (roll_max - roll_min)
                )
        
        return result_df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, 
                                   feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples (feature1, feature2)
            
        Returns:
            DataFrame with interaction features added
        """
        result_df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 not in df.columns or feat2 not in df.columns:
                continue
                
            # Multiplicative interaction
            result_df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]
            
            # Ratio (with safe division)
            result_df[f"{feat1}_div_{feat2}"] = np.where(
                df[feat2] != 0, df[feat1] / df[feat2], 0
            )
            
            # Difference
            result_df[f"{feat1}_minus_{feat2}"] = df[feat1] - df[feat2]
        
        return result_df
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from DataFrame index.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        result_df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not datetime, skipping time features")
            return result_df
        
        # Basic time features
        result_df['hour'] = df.index.hour
        result_df['day_of_week'] = df.index.dayofweek
        result_df['day_of_month'] = df.index.day
        result_df['month'] = df.index.month
        result_df['quarter'] = df.index.quarter
        result_df['year'] = df.index.year
        
        # Cyclical encoding
        result_df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        result_df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        result_df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Market session indicators (assuming NYSE hours)
        result_df['is_market_hours'] = (
            (df.index.hour >= 9) & (df.index.hour < 16) & 
            (df.index.dayofweek < 5)
        ).astype(int)
        
        result_df['is_opening_hour'] = (df.index.hour == 9).astype(int)
        result_df['is_closing_hour'] = (df.index.hour == 15).astype(int)
        result_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return result_df
    
    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features added
        """
        result_df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required OHLCV columns")
            return result_df
        
        # Basic price features
        result_df['price_range'] = df['high'] - df['low']
        result_df['body_size'] = abs(df['close'] - df['open'])
        result_df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        result_df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Relative features
        result_df['body_to_range'] = np.where(
            result_df['price_range'] > 0,
            result_df['body_size'] / result_df['price_range'],
            0
        )
        
        result_df['upper_shadow_to_range'] = np.where(
            result_df['price_range'] > 0,
            result_df['upper_shadow'] / result_df['price_range'],
            0
        )
        
        result_df['lower_shadow_to_range'] = np.where(
            result_df['price_range'] > 0,
            result_df['lower_shadow'] / result_df['price_range'],
            0
        )
        
        # Returns
        result_df['return_1'] = df['close'].pct_change(1)
        result_df['return_3'] = df['close'].pct_change(3)
        result_df['return_5'] = df['close'].pct_change(5)
        
        # Log returns
        result_df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        result_df['volume_ma_5'] = df['volume'].rolling(5).mean()
        result_df['volume_ratio'] = np.where(
            result_df['volume_ma_5'] > 0,
            df['volume'] / result_df['volume_ma_5'],
            1
        )
        
        # Price position features
        result_df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        result_df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])
        
        return result_df
    
    @staticmethod
    def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with all engineered features
        """
        result_df = df.copy()
        
        # Price features
        result_df = FeatureEngineer.create_price_features(result_df)
        
        # Time features
        result_df = FeatureEngineer.create_time_features(result_df)
        
        # Lagged features for key indicators
        key_features = ['close', 'volume', 'return_1']
        lags = [1, 2, 3, 5]
        result_df = FeatureEngineer.create_lagged_features(result_df, key_features, lags)
        
        # Rolling features
        windows = [5, 10, 20]
        result_df = FeatureEngineer.create_rolling_features(
            result_df, ['close', 'volume'], windows
        )
        
        # Interaction features
        interactions = [
            ('volume', 'return_1'),
            ('price_range', 'volume'),
            ('body_size', 'volume')
        ]
        result_df = FeatureEngineer.create_interaction_features(result_df, interactions)
        
        logger.info(f"Created {len(result_df.columns) - len(df.columns)} engineered features")
        
        return result_df 