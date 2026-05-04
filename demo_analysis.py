from interface import StockDataInterface
import numpy as np

def run_sample_analysis(symbol="20MICRONS"):
    # 1. Initialize Interface
    api = StockDataInterface()
    
    print(f"--- Analysis for {symbol} ---")
    
    # 2. Load Data (Filtering for a specific month for speed)
    # Using raw minute data
    data = api.load_stock(symbol, start_date="2020-01-01", end_date="2020-01-31")
    print(f"Loaded {len(data)} minutes of data.")
    
    # 3. Resample to 15-minute bars
    data_15m = api.resample(data, timeframe='15min')
    print(f"Resampled to {len(data_15m)} 15-minute bars.")
    
    # 4. Vectorized Calculations using NumPy
    closes = data_15m['close']
    
    # Calculate returns
    returns = np.diff(closes) / closes[:-1]
    
    # Calculate SMA
    sma_20 = api.calculate_sma(closes, window=20)
    
    # Calculate RSI
    rsi_14 = api.calculate_rsi(closes, window=14)
    
    # 5. Output some results
    print(f"\nAnalysis Summary (Last 5 periods):")
    print(f"{'Time':<20} | {'Close':<8} | {'SMA_20':<8} | {'RSI':<8}")
    print("-" * 55)
    
    # Align SMA and RSI with dates (they have different lengths if not padded)
    # SMA is shorter due to window
    for i in range(-5, 0):
        t = str(data_15m['date'][i])[:19]
        c = closes[i]
        # sma might have fewer elements than closes
        s = sma_20[i] if len(sma_20) >= abs(i) else np.nan
        r = rsi_14[i]
        print(f"{t:<20} | {c:<8.2f} | {s:<8.2f} | {r:<8.2f}")

    # 6. Quantitative Insights
    volatility = np.std(returns) * np.sqrt(252 * 6.5 * 4) # Annualized vol approx
    print(f"\nEstimated Annualized Volatility (based on Jan 2020): {volatility:.2%}")

if __name__ == "__main__":
    run_sample_analysis()
