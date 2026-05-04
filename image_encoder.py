import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField
import os

class TimeSeriesImageEncoder:
    """
    Encodes 1D time series data into 2D images using GAF and MTF techniques.
    """
    
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        
    def to_gaf(self, data: np.ndarray, method: str = 'summation') -> np.ndarray:
        """
        Transforms data to Gramian Angular Field.
        method: 'summation' (GASF) or 'difference' (GADF)
        """
        # Data needs to be in shape (n_samples, n_timestamps)
        # We assume 1 sample for now
        X = data.reshape(1, -1)
        
        # If data is too long, we might need to piecewise aggregate approximate (PAA)
        # But pyts handles the resizing if we specify image_size in GAF? 
        # Actually GAF doesn't resize by default, it creates an N x N matrix where N is len(data).
        # We usually want a fixed size for CNNs.
        
        gaf = GramianAngularField(image_size=self.image_size, method=method)
        return gaf.fit_transform(X)[0]

    def to_mtf(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms data to Markov Transition Field.
        """
        X = data.reshape(1, -1)
        mtf = MarkovTransitionField(image_size=self.image_size)
        return mtf.fit_transform(X)[0]

    def create_sliding_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        """
        Creates sliding windows from a 1D array.
        """
        n_windows = (len(data) - window_size) // step + 1
        windows = np.zeros((n_windows, window_size))
        for i in range(n_windows):
            start = i * step
            windows[i] = data[start : start + window_size]
        return windows

    def generate_batch_gaf(self, windows: np.ndarray, method: str = 'summation') -> np.ndarray:
        """
        Transforms a batch of windows into GAF images.
        windows shape: (n_samples, window_size)
        """
        gaf = GramianAngularField(image_size=self.image_size, method=method)
        return gaf.fit_transform(windows)

    def save_image(self, image_data: np.ndarray, filename: str, cmap: str = 'rainbow'):
        """Saves the 2D array as an image file."""
        plt.figure(figsize=(5, 5))
        plt.imshow(image_data, cmap=cmap, origin='lower')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Image saved to {filename}")

if __name__ == "__main__":
    # Test with synthetic data
    encoder = TimeSeriesImageEncoder(image_size=64)
    test_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    
    gasf = encoder.to_gaf(test_data, method='summation')
    gadf = encoder.to_gaf(test_data, method='difference')
    mtf = encoder.to_mtf(test_data)
    
    os.makedirs("encoded_images", exist_ok=True)
    encoder.save_image(gasf, "encoded_images/test_gasf.png")
    encoder.save_image(gadf, "encoded_images/test_gadf.png")
    encoder.save_image(mtf, "encoded_images/test_mtf.png")
