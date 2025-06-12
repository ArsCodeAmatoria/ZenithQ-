"""
Data Preprocessor

Handles data cleaning, normalization, and feature engineering for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing for ML trading models."""
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
        """
        self.scaling_method = scaling_method
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame, 
                        target_column: str = 'future_return',
                        lookback_window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for ML training.
        
        Args:
            df: DataFrame with OHLCV and indicators
            target_column: Name of target variable
            lookback_window: Number of historical periods to include
            
        Returns:
            Tuple of (features, targets)
        """
        # Create target variable (future returns)
        df = df.copy()
        df[target_column] = df['close'].pct_change().shift(-1)
        
        # Remove non-numeric columns and NaN values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.dropna()
        
        if len(numeric_df) < lookback_window:
            logger.warning("Insufficient data after preprocessing")
            return np.array([]), np.array([])
        
        # Separate features and target
        feature_columns = [col for col in numeric_df.columns if col != target_column]
        self.feature_columns = feature_columns
        
        # Create sequences for time series
        X, y = self._create_sequences(
            numeric_df[feature_columns].values,
            numeric_df[target_column].values,
            lookback_window
        )
        
        # Scale features
        if len(X) > 0:
            X = self._scale_features(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return X, y
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray, 
                         window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(window_size, len(features)):
            X.append(features[i-window_size:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using the specified method."""
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.scaling_method}")
            return X
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler for later use
        self.scalers['features'] = scaler
        
        return X_scaled
    
    def create_labels(self, df: pd.DataFrame, 
                     method: str = "return_threshold",
                     threshold: float = 0.01) -> pd.Series:
        """
        Create trading labels for supervised learning.
        
        Args:
            df: DataFrame with price data
            method: Labeling method ('return_threshold', 'triple_barrier')
            threshold: Return threshold for classification
            
        Returns:
            Series with labels (1: buy, 0: hold, -1: sell)
        """
        if method == "return_threshold":
            return self._create_return_threshold_labels(df, threshold)
        elif method == "triple_barrier":
            return self._create_triple_barrier_labels(df, threshold)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def _create_return_threshold_labels(self, df: pd.DataFrame, 
                                      threshold: float) -> pd.Series:
        """Create labels based on return thresholds."""
        future_returns = df['close'].pct_change().shift(-1)
        
        labels = pd.Series(0, index=df.index)  # Default: hold
        labels[future_returns > threshold] = 1   # Buy signal
        labels[future_returns < -threshold] = -1  # Sell signal
        
        return labels
    
    def _create_triple_barrier_labels(self, df: pd.DataFrame, 
                                    threshold: float) -> pd.Series:
        """Create labels using triple barrier method."""
        # Simplified triple barrier implementation
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - 5):  # Look 5 periods ahead
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+6]
            
            # Upper barrier (profit taking)
            upper_barrier = current_price * (1 + threshold)
            # Lower barrier (stop loss)
            lower_barrier = current_price * (1 - threshold)
            
            # Check which barrier is hit first
            upper_hit = (future_prices >= upper_barrier).any()
            lower_hit = (future_prices <= lower_barrier).any()
            
            if upper_hit and not lower_hit:
                labels.iloc[i] = 1  # Buy
            elif lower_hit and not upper_hit:
                labels.iloc[i] = -1  # Sell
            # Otherwise remains 0 (hold)
        
        return labels 