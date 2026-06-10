import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# KAGGLE CLOUD CONFIGURATION & PATHS
# ==============================================================================
KAGGLE_INPUT_DIR = "/kaggle/input/emnist"
KAGGLE_WORKING_DIR = "/kaggle/working"

TRAIN_CSV = os.path.join(KAGGLE_INPUT_DIR, "emnist-balanced-train.csv")
TEST_CSV = os.path.join(KAGGLE_INPUT_DIR, "emnist-balanced-test.csv")
MAPPING_TXT = os.path.join(KAGGLE_INPUT_DIR, "emnist-balanced-mapping.txt")

IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
NUM_CLASSES = 47
BATCH_SIZE = 256
EPOCHS = 25  # Scaled to 25 with EarlyStopping guards per requirement

# ==============================================================================
# HARDWARE CONFIGURATION
# ==============================================================================
def configure_hardware():
    tf.keras.backend.set_floatx('float32')
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] VRAM Memory Growth Enabled on {len(physical_devices)} physical GPUs.")
        except RuntimeError as e:
            print(e)
    else:
        print("[WARNING] No Physical GPUs found. Running on CPU.")

# ==============================================================================
# DATA INGESTION & TRANSFORMATION
# ==============================================================================
def load_mapping():
    print(f"[INFO] Loading EMNIST mapping from {MAPPING_TXT}...")
    mapping = {}
    with open(MAPPING_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label_idx, ascii_val = int(parts[0]), int(parts[1])
                mapping[label_idx] = chr(ascii_val)
                
    mapping_json_path = os.path.join(KAGGLE_WORKING_DIR, 'mapping.json')
    # Save mapping for future reference in deployment
    with open(mapping_json_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"[INFO] Mapping serialized to {mapping_json_path}")
    return mapping

def process_csv(csv_path):
    print(f"[INFO] Ingesting & transforming {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1:].values.astype('float32')
    
    num_samples = x.shape[0]
    x = x.reshape(num_samples, IMG_ROWS, IMG_COLS)
    
    # Structural Transposition: 90-degree CCW & flip fix for EMNIST
    x = np.transpose(x, (0, 2, 1))
    
    x = x.reshape(num_samples, IMG_ROWS, IMG_COLS, CHANNELS)
    # Scale matrix attributes to [0.0, 1.0]
    x /= 255.0
    return x, y

# ==============================================================================
# CLOUD-TIER V3 MODEL ARCHITECTURE
# ==============================================================================
def build_model_v3(input_shape=(28, 28, 1), num_classes=47):
    print("[INFO] Constructing 3-Stage VGG-Style CNN Topology...")
    model = Sequential([
        # Stage 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Stage 2
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Stage 3 (Cloud Tier Upgrade)
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),
        
        # Dense Head
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.50),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    configure_hardware()
    
    mapping = load_mapping()
    x_train, y_train = process_csv(TRAIN_CSV)
    x_test, y_test = process_csv(TEST_CSV)
    
    print(f"[INFO] Training Tensor: {x_train.shape} | Testing Tensor: {x_test.shape}")
    
    # Real-Time Augmentation Engine
    print("[INFO] Initializing Stroke-Invariant ImageDataGenerator...")
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(x_train)
    
    model = build_model_v3(input_shape=(IMG_ROWS, IMG_COLS, CHANNELS), num_classes=NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # Cloud Telemetry Callbacks
    model_path = os.path.join(KAGGLE_WORKING_DIR, 'handwritten_character_cnn_v3.keras')
    log_path = os.path.join(KAGGLE_WORKING_DIR, 'training_log_v3.csv')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1)
    csv_logger = CSVLogger(filename=log_path, append=False)
    
    print(f"\n[INFO] Commencing V3 Cloud Optimization Pass (Batch: {BATCH_SIZE}, Max Epochs: {EPOCHS})...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stopping, model_checkpoint, csv_logger]
    )
    
    print("\n[INFO] Training Completed. Extracting Final Validation Telemetry...")
    
    # Evaluation Pass
    print("[INFO] Generating Full Classification Report...")
    y_pred_probs = model.predict(x_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    labels = [int(k) for k in sorted(mapping.keys(), key=int)]
    target_names = [mapping[k] for k in labels]
    
    report = classification_report(y_test, y_pred, labels=labels, target_names=target_names)
    print("\n" + "="*60)
    print("V3 CLOUD EVALUATION TELEMETRY")
    print("="*60)
    print(report)
    print("="*60)
    
    print(f"[INFO] Pipeline Execution Complete. Model artifact saved to: {model_path}")
    print(f"[INFO] Telemetry matrix saved to: {log_path}")

if __name__ == '__main__':
    main()
