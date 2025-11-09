import os
import numpy as np
from PIL import Image

def check_outliers_on_folder(image_dir, percentile=1):
    """
    Scan all images in folder and check pixel outlier distance distribution.
    """
    all_pixels = []

    # iterate over all images in folder recursively
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    # convert image to grayscale to simplify pixel range evaluation
                    img = Image.open(path).convert("L")
                    arr = np.array(img, dtype=np.float32).flatten()
                    all_pixels.append(arr)
                except Exception as e:
                    print(f"⚠ Error loading {path}: {e}")

    if not all_pixels:
        print("⚠ No valid images found.")
        return

    # merge all pixels from all images
    vec = np.concatenate(all_pixels)
    mean = np.mean(vec)
    std = np.std(vec)

    # avoid division by zero case
    if std == 0:
        print("⚠ Std=0 (all pixels identical).")
        return

    # compute most extreme deviation (z-score)
    max_dev = np.max(np.abs(vec - mean)) / std

    # percentile low/high bounds
    low_bound = np.percentile(vec, percentile)
    high_bound = np.percentile(vec, 100 - percentile)
    low_dev = abs(low_bound - mean) / std
    high_dev = abs(high_bound - mean) / std

    print(f"Analyzed {len(vec):,} pixels from {image_dir}")
    print(f"Mean = {mean:.2f}, Std = {std:.2f}")
    print(f"Most extreme pixel: {max_dev:.2f}σ from mean")
    print(f"{percentile}% lowest pixels: ~{low_dev:.2f}σ from mean")
    print(f"{percentile}% highest pixels: ~{high_dev:.2f}σ from mean")


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    image_dir = os.path.join(base, "animals_dataset")
    check_outliers_on_folder(image_dir, percentile=1)
