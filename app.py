import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import sys

# Load config path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config

from model import build_model_v2

# Cache the model to prevent reload on every UI interaction
@st.cache_resource
def load_assets():
    model_path = os.path.join(config.BASE_DIR, 'models', 'handwritten_character_cnn_v2.keras')
    # Bypass Keras 3 serialization bugs by explicitly building architecture and loading weights
    model = build_model_v2()
    model.load_weights(model_path)
    
    with open(config.MAPPING_JSON, 'r') as f:
        mapping = json.load(f)
    return model, mapping

model, mapping = load_assets()

def process_and_predict(image_array):
    # Convert to PIL Image for robust preprocessing without OpenCV
    img = Image.fromarray(image_array.astype('uint8'))
    
    # Convert to grayscale
    gray_img = img.convert("L")
        
    # Resize down to exactly 28x28 pixels
    resized_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)
    resized = np.array(resized_img)
    
    # Normalize pixel intensity down to float32 range [0.0, 1.0]
    normalized = resized.astype('float32') / 255.0
    
    # CRITICAL NUANCE: Ensure upright bright strokes on dark background
    # If the image is mostly light (mean > 0.5), automatically invert it
    if np.mean(normalized) > 0.5:
        normalized = 1.0 - normalized
        
    # Reshape for structural input metric (1, 28, 28, 1)
    input_tensor = normalized.reshape(1, 28, 28, 1)
    
    # Live MLOps Inference
    preds = model.predict(input_tensor)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    
    # Parse output using mapping.json translation key
    predicted_char = mapping[str(pred_idx)]
    
    return predicted_char, confidence, normalized

st.title("Handwritten Character Recognition")
st.write("Interactive multi-class alphanumeric visualization dashboard.")

tab1, tab2 = st.tabs(["Drawing Canvas", "File Uploader"])

with tab1:
    st.write("Draw a character (0-9, a-z, A-Z)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#000000",
        background_color="#ffffff",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Predict from Canvas"):
        if canvas_result.image_data is not None:
            char, conf, processed_img = process_and_predict(canvas_result.image_data)
            st.success(f"Model Prediction: **{char}**")
            st.info(f"Confidence Level: {conf:.2f}%")
            st.image(processed_img, caption="Preprocessed Normalized Tensor (28x28)", width=150)

with tab2:
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=280)
        
        if st.button("Predict from Upload"):
            img_array = np.array(image)
            char, conf, processed_img = process_and_predict(img_array)
            st.success(f"Model Prediction: **{char}**")
            st.info(f"Confidence Level: {conf:.2f}%")
            st.image(processed_img, caption="Preprocessed Normalized Tensor (28x28)", width=150)
