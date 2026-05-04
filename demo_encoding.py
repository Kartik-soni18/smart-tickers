import os
import numpy as np
from interface import StockDataInterface
from image_encoder import TimeSeriesImageEncoder

def run_encoding_demo(symbol="20MICRONS"):
    # 1. Initialize Interface and Encoder
    api = StockDataInterface()
    encoder = TimeSeriesImageEncoder(image_size=64)
    
    print(f"--- Encoding Analysis for {symbol} ---")
    
    # 2. Load a chunk of data (e.g., 500 minutes)
    # We want a sequence to encode
    data = api.load_stock(symbol, start_date="2021-01-01", end_date="2021-01-10")
    
    if len(data) < 128:
        print("Not enough data for the requested image size.")
        return
        
    # We'll take the first 500 points for the demo
    prices = data['close'][:500]
    
    # 3. Generate Encodings
    print("Generating GAF and MTF encodings...")
    gasf = encoder.to_gaf(prices, method='summation')
    gadf = encoder.to_gaf(prices, method='difference')
    mtf = encoder.to_mtf(prices)
    
    # 4. Sliding Window Batch Example (Preparation for CNN)
    output_dir = f"encoded_images/{symbol}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nDemonstrating sliding window batch generation...")
    window_size = 64
    windows = encoder.create_sliding_windows(prices, window_size=window_size, step=10)
    print(f"Created {len(windows)} windows of size {window_size}.")
    
    batch_gasf = encoder.generate_batch_gaf(windows)
    print(f"Generated batch of GASF images with shape: {batch_gasf.shape}")
    
    # Save a few from the batch
    batch_dir = f"{output_dir}/batch_samples"
    os.makedirs(batch_dir, exist_ok=True)
    for i in range(min(5, len(batch_gasf))):
        encoder.save_image(batch_gasf[i], f"{batch_dir}/sample_{i}.png", cmap='viridis')

    print(f"\nSuccess! Check the '{output_dir}' folder for the results.")

if __name__ == "__main__":
    run_encoding_demo()
