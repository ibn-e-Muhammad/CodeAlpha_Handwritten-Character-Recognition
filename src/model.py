import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import config

def build_model(input_shape=(28, 28, 1), num_classes=47):
    model = Sequential([
        # Feature Extraction Stack
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Regularization to penalize complexity and prevent high-dimensional overfitting
        Dropout(0.25),
        
        # Dense Classification Output
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == '__main__':
    model = build_model(input_shape=(config.IMG_ROWS, config.IMG_COLS, config.CHANNELS), 
                        num_classes=config.NUM_CLASSES)
    
    # Print summary to console
    model.summary()
    
    # Save the summary layout to pipeline_vision/data/model_summary.txt
    summary_path = os.path.join(config.DATA_DIR, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        # We use a lambda to capture the summary string
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print(f"Model structural topology exported to {summary_path}")
