import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import config

# =====================================================================
# V1 Architecture (Baseline — 229,807 params — 88.37% Val Acc)
# =====================================================================
def build_model_v1(input_shape=(28, 28, 1), num_classes=47):
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

# =====================================================================
# V2 Architecture (VGG-Style Double-Conv Blocks + BatchNorm + LR Decay)
# Target: 91%+ Val Acc on 47-class EMNIST Balanced
# =====================================================================
def build_model_v2(input_shape=(28, 28, 1), num_classes=47):
    model = Sequential([
        # --- Convolutional Stack 1 ---
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # --- Convolutional Stack 2 ---
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # --- Classification Head ---
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Default build_model points to V2
def build_model(input_shape=(28, 28, 1), num_classes=47):
    return build_model_v2(input_shape, num_classes)

if __name__ == '__main__':
    model = build_model_v2(input_shape=(config.IMG_ROWS, config.IMG_COLS, config.CHANNELS), 
                           num_classes=config.NUM_CLASSES)
    
    # Print summary to console
    model.summary()
    
    # Save the V2 summary layout to pipeline_vision/data/model_summary_v2.txt
    summary_path = os.path.join(config.DATA_DIR, 'model_summary_v2.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print(f"V2 Model structural topology exported to {summary_path}")
