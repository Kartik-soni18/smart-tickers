import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class StockDataInterface:
    """
    A high-performance interface for loading and analyzing minute-level stock data.
    Provides data in NumPy formats optimized for quantitative analysis.
    """
    
    def __init__(self, data_dir: str = "minute"):
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory '{self.data_dir}' does not exist.")

    def list_symbols(self) -> List[str]:
        """Returns a list of all available stock symbols."""
        # Using scandir for better performance on large directories
        with os.scandir(self.data_dir) as entries:
            return sorted([
                e.name[:-4] for e in entries 
                if e.is_file() and e.name.endswith(".csv")
            ])

    def load_stock(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> np.ndarray:
        """
        Loads stock data into a structured NumPy array with optional date filtering.
        
        Args:
            symbol: The stock symbol to load.
            start_date: Filter data starting from this date (inclusive). Format: 'YYYY-MM-DD'
            end_date: Filter data up to this date (inclusive). Format: 'YYYY-MM-DD'
            
        Returns:
            np.ndarray: Structured array with fields: date, close, high, low, open, volume
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Symbol '{symbol}' not found in {self.data_dir}")

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
        if start_date:
            start_ts = pd.to_datetime(start_date).tz_localize(None)
            df = df[df['date'] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date).tz_localize(None)
            df = df[df['date'] <= end_ts]
            
        df['date'] = df['date'].values.astype('datetime64[ns]')
        return df.to_records(index=False)

    def load_columns(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Loads stock data as a dictionary of separate NumPy arrays with optional filtering.
        """
        data = self.load_stock(symbol, start_date, end_date)
        return {name: data[name] for name in data.dtype.names}

    @staticmethod
    def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
        """Calculates Simple Moving Average using NumPy."""
        if len(prices) < window:
            return np.full_like(prices, np.nan)
        weights = np.ones(window) / window
        return np.convolve(prices, weights, mode='valid')

    @staticmethod
    def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculates Relative Strength Index using NumPy."""
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:window+1] = 100. - 100. / (1. + rs)

        for i in range(window + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi

    @staticmethod
    def resample(data: np.ndarray, timeframe: str = '5min') -> np.ndarray:
        """
        Resamples minute data to a higher timeframe.
        
        Args:
            data: Structured array from load_stock.
            timeframe: Pandas timeframe string (e.g., '5min', '15min', '1H', '1D').
            
        Returns:
            np.ndarray: Resampled structured array.
        """
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled['date'] = resampled['date'].values.astype('datetime64[ns]')
        return resampled.to_records(index=False)

if __name__ == "__main__":
    # Quick demonstration
    try:
        interface = StockDataInterface()
        symbols = interface.list_symbols()
        print(f"Found {len(symbols)} stocks.")
        
        if symbols:
            sample = symbols[0]
            print(f"\nLoading data for: {sample}")
            
            # Example 1: Structured Array
            data = interface.load_stock(sample)
            print(f"Last 5 Close prices: {data['close'][-5:]}")
            
            # Example 2: Dictionary of Arrays
            cols = interface.load_columns(sample)
            avg_price = (cols['high'] + cols['low']) / 2
            print(f"Calculated average price for {len(avg_price)} rows.")
            
    except Exception as e:
        print(f"Error: {e}")
