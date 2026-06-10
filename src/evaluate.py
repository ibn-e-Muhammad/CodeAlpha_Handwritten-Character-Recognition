import os
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config

def main():
    print("Loading preprocessed test data...")
    x_test = np.load(config.X_TEST_NPY)
    y_test = np.load(config.Y_TEST_NPY)
    
    with open(config.MAPPING_JSON, 'r') as f:
        mapping = json.load(f)
        
    model_path = os.path.join(config.BASE_DIR, 'models', 'handwritten_character_cnn.keras')
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Running full-set inference on validation samples...")
    y_pred_probs = model.predict(x_test, batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    labels = [int(k) for k in sorted(mapping.keys(), key=int)]
    target_names = [mapping[str(k)] for k in labels]
    
    report = classification_report(y_test, y_pred, labels=labels, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)
    
    # Generate Confusion Matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plots_dir = os.path.join(config.DATA_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - 47 Classes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved confusion matrix plot to {plot_path}")
    
    report_path = os.path.join(plots_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
        
if __name__ == '__main__':
    main()
