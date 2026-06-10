import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import config
from model import build_model_v2

def configure_hardware():
    # Enforce standard float32 calculation loops (No mixed precision)
    tf.keras.backend.set_floatx('float32')
    
    # Hardware Defence Constraints (CRITICAL): Enable VRAM Memory Growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"VRAM Memory Growth Enabled on {len(physical_devices)} physical GPUs.")
        except RuntimeError as e:
            print(e)
    else:
        print("No Physical GPUs found. Running on CPU.")

def load_data():
    print("Loading preprocessed .npy arrays...")
    x_train = np.load(config.X_TRAIN_NPY)
    y_train = np.load(config.Y_TRAIN_NPY)
    x_test = np.load(config.X_TEST_NPY)
    y_test = np.load(config.Y_TEST_NPY)
    return (x_train, y_train), (x_test, y_test)

def main():
    configure_hardware()
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(config.BASE_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    model = build_model_v2(input_shape=(config.IMG_ROWS, config.IMG_COLS, config.CHANNELS),
                           num_classes=config.NUM_CLASSES)
    
    # Compilation with explicit Adam learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks setup — V2 paths
    model_path = os.path.join(models_dir, 'handwritten_character_cnn_v2.keras')
    log_path = os.path.join(config.DATA_DIR, 'training_log_v2.csv')
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=False, verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1
    )
    
    csv_logger = CSVLogger(filename=log_path, append=False)
    
    # Dynamic Learning Rate Scheduler — halve LR when val_loss plateaus for 3 epochs
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    
    print("Starting V2 network training & optimization loop...")
    print(f"  Epochs: {config.EPOCHS} | Batch Size: {config.BATCH_SIZE}")
    print(f"  Model checkpoint: {model_path}")
    print(f"  CSV telemetry: {log_path}")
    
    history = model.fit(
        x_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, model_checkpoint, csv_logger, reduce_lr]
    )
    
    print(f"Training completed. Best model artifact serialized to {model_path}")
    print(f"Execution telemetry logged to {log_path}")

if __name__ == '__main__':
    main()
