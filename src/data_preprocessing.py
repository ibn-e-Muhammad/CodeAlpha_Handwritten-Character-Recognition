import os
import numpy as np
import pandas as pd
import json
import config

def load_emnist_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label_idx, ascii_val = int(parts[0]), int(parts[1])
                mapping[label_idx] = chr(ascii_val)
    return mapping

def process_csv(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    
    # Isolate label column 0
    y = df.iloc[:, 0].values
    
    # Convert remaining 784 pixel attributes into (28, 28, 1) matrices
    x = df.iloc[:, 1:].values.astype('float32')
    
    num_samples = x.shape[0]
    x = x.reshape(num_samples, config.IMG_ROWS, config.IMG_COLS)
    
    # The EMNIST Anomaly Solution
    # EMNIST is rotated 90 degrees counter-clockwise and mirrored horizontally.
    # To fix, we can transpose the spatial dimensions (equivalent index slicing).
    x = np.transpose(x, (0, 2, 1))
    
    # Reshape to include channel dimension
    x = x.reshape(num_samples, config.IMG_ROWS, config.IMG_COLS, config.CHANNELS)
    
    # Normalize pixel array to [0.0, 1.0]
    x /= 255.0
    
    return x, y

def main():
    print("Starting Phase 2: Data Preprocessing...")
    
    # Ingest mapping
    mapping = load_emnist_mapping(config.MAPPING_TXT)
    with open(config.MAPPING_JSON, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved mapping dictionary to {config.MAPPING_JSON}")
    
    # Process train
    x_train, y_train = process_csv(config.TRAIN_CSV)
    print(f"Processed x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_train value range: [{x_train.min()}, {x_train.max()}]")
    
    # Process test
    x_test, y_test = process_csv(config.TEST_CSV)
    print(f"Processed x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"x_test value range: [{x_test.min()}, {x_test.max()}]")
    
    # Save arrays as binary .npy configurations
    np.save(config.X_TRAIN_NPY, x_train)
    np.save(config.Y_TRAIN_NPY, y_train)
    np.save(config.X_TEST_NPY, x_test)
    np.save(config.Y_TEST_NPY, y_test)
    
    print("Saved finalized cleaned arrays to disk for lightning-fast access.")
    print("Preprocessing completed successfully.")

if __name__ == '__main__':
    main()
