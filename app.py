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

from model import build_model_v1, build_model_v2, build_model_v3

# Cache the models to prevent reload on every UI interaction
@st.cache_resource
def load_assets():
    # Load V1
    model_v1_path = os.path.join(config.BASE_DIR, 'models', 'handwritten_character_cnn.keras')
    model_v1 = build_model_v1()
    model_v1.load_weights(model_v1_path)

    # Load V2
    model_v2_path = os.path.join(config.BASE_DIR, 'models', 'handwritten_character_cnn_v2.keras')
    model_v2 = build_model_v2()
    model_v2.load_weights(model_v2_path)

    # Load V3
    model_v3_path = os.path.join(config.BASE_DIR, 'models', 'handwritten_character_cnn_v3.keras')
    model_v3 = build_model_v3()
    model_v3.load_weights(model_v3_path)
    
    with open(config.MAPPING_JSON, 'r') as f:
        mapping = json.load(f)
        
    return model_v1, model_v2, model_v3, mapping

model_v1, model_v2, model_v3, mapping = load_assets()

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
    
    # Live MLOps Inference for all 3 models
    preds_v1 = model_v1.predict(input_tensor)
    pred_idx_v1 = np.argmax(preds_v1[0])
    conf_v1 = preds_v1[0][pred_idx_v1] * 100
    char_v1 = mapping[str(pred_idx_v1)]

    preds_v2 = model_v2.predict(input_tensor)
    pred_idx_v2 = np.argmax(preds_v2[0])
    conf_v2 = preds_v2[0][pred_idx_v2] * 100
    char_v2 = mapping[str(pred_idx_v2)]

    preds_v3 = model_v3.predict(input_tensor)
    pred_idx_v3 = np.argmax(preds_v3[0])
    conf_v3 = preds_v3[0][pred_idx_v3] * 100
    char_v3 = mapping[str(pred_idx_v3)]
    
    return {
        "v1": {"char": char_v1, "conf": conf_v1},
        "v2": {"char": char_v2, "conf": conf_v2},
        "v3": {"char": char_v3, "conf": conf_v3}
    }, normalized

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
            results, processed_img = process_and_predict(canvas_result.image_data)
            
            st.image(processed_img, caption="Preprocessed Normalized Tensor (28x28)", width=150)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("### V1 (Baseline)")
                st.success(f"**{results['v1']['char']}**")
                st.write(f"{results['v1']['conf']:.2f}%")
            with col2:
                st.info("### V2 (VGG-Style)")
                st.success(f"**{results['v2']['char']}**")
                st.write(f"{results['v2']['conf']:.2f}%")
            with col3:
                st.info("### V3 (Cloud-Tier)")
                st.success(f"**{results['v3']['char']}**")
                st.write(f"{results['v3']['conf']:.2f}%")

with tab2:
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=280)
        
        if st.button("Predict from Upload"):
            img_array = np.array(image)
            results, processed_img = process_and_predict(img_array)
            
            st.image(processed_img, caption="Preprocessed Normalized Tensor (28x28)", width=150)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("### V1 (Baseline)")
                st.success(f"**{results['v1']['char']}**")
                st.write(f"{results['v1']['conf']:.2f}%")
            with col2:
                st.info("### V2 (VGG-Style)")
                st.success(f"**{results['v2']['char']}**")
                st.write(f"{results['v2']['conf']:.2f}%")
            with col3:
                st.info("### V3 (Cloud-Tier)")
                st.success(f"**{results['v3']['char']}**")
                st.write(f"{results['v3']['conf']:.2f}%")
