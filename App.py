import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
from pathlib import Path
import time

# ========================================
# MODEL CONFIGURATION
# ========================================
os.makedirs("models", exist_ok=True)

# Try production model first, fallback to legacy
PRODUCTION_MODEL_PATH = "models/export/saved_model"
ALT_MODEL_PATH = "models/efficientnet_final.keras"
LEGACY_MODEL_PATH = "models/sipakmed_best_2.keras"
HISTORY_PATH = "models/training_history.pkl"
CM_PATH = "models/confusion_matrix.pkl"
METADATA_PATH = "models/export/model_metadata.json"

# Determine which model to use
def find_model():
    if os.path.exists(PRODUCTION_MODEL_PATH):
        return PRODUCTION_MODEL_PATH, 224, "Production Model (EfficientNet)"
    elif os.path.exists(ALT_MODEL_PATH):
        return ALT_MODEL_PATH, 224, "Production Model (Keras)"
    elif os.path.exists(LEGACY_MODEL_PATH):
        return LEGACY_MODEL_PATH, 64, "Legacy Model"
    else:
        return None, None, None

MODEL_PATH, IMG_SIZE, MODEL_NAME = find_model()

# Load model with caching
@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        st.error("❌ No model found! Please train a model first using: python ml/train.py")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

# Load metadata if available
@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return None

model = load_model()
metadata = load_metadata()

CLASS_LABELS = metadata.get('class_names', [
    "Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"
]) if metadata else ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# ========================================
# Helper Functions
# ========================================
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def grad_cam(model, img_array, layer_name="conv2d_15"):
    target_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if sub_layer.name == layer_name:
                    target_layer = sub_layer
                    break
        if target_layer:
            break
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
    return superimposed

# ========================================
# Streamlit UI
# ========================================
st.set_page_config(page_title="Cervical Cytology Classifier", layout="wide")
st.title("Automated Cervical Cytology Classification")
st.markdown("Upload Pap smear cell images to classify into 5 categories using a deep learning model.")

# Sidebar
st.sidebar.header("About this App")
st.sidebar.info("""
Developed for the **16th KEMRI Annual Scientific & Health Conference**  
**Authors:** Mikenickson Wanjohi et al.  
**Dataset:** SIPaKMeD  
**Model:** Custom CNN (TensorFlow/Keras)  
**Test Accuracy:** 91.79% | **Validation Accuracy:** 92.27%
""")

tab1, tab2 = st.tabs(["Classifier", "Dashboard"])

# -----------------------------
# Tab 1: Classifier
# -----------------------------
with tab1:
    st.success("Model loaded successfully! Ready for predictions.")
    uploaded_files = st.file_uploader("Upload Pap Smear Images", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption="Uploaded Image", width=700)
            processed_img = preprocess_image(img)
            preds = model.predict(processed_img, verbose=0)
            pred_idx = np.argmax(preds)
            pred_class = CLASS_LABELS[pred_idx]
            confidence_score = float(preds[0][pred_idx])        # ← Python float
            confidence_percent = confidence_score * 100

            st.subheader(f"Prediction: **{pred_class}**")
            st.progress(confidence_score)
            st.write(f"Confidence: **{confidence_percent:.2f}%**")

            # Grad-CAM
            try:
                heatmap = grad_cam(model, processed_img)
                overlay = overlay_gradcam(img, heatmap)
                st.markdown("**Grad-CAM Heatmap (regions the model focused on):**")
                st.image(overlay, width=400)
            except Exception as e:
                st.warning(f"Grad-CAM temporarily unavailable: {e}")

# Tab 2: Dashboard

with tab2:
    st.subheader("Model Performance Dashboard")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history['accuracy'], label='Train Accuracy')
        ax[0].plot(history['val_accuracy'], label='Val Accuracy')
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].legend()
        ax[1].plot(history['loss'], label='Train Loss')
        ax[1].plot(history['val_loss'], label='Val Loss')
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.info("Training history (history.pkl) not found yet. Upload it to your GitHub release to display curves.")

    if os.path.exists(CM_PATH):
        with open(CM_PATH, "rb") as f:
            cm, labels = pickle.load(f)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)
    else:
        st.info("Confusion matrix (confusion_matrix.pkl) not found yet. Upload it to the release to show here.")




