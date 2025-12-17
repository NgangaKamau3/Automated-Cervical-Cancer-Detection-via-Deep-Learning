import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Constants
IMG_SIZE = 224
CLASS_NAMES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
MODEL_PATH = "models/efficientnet_final.keras"

# Load model
# We load with compile=False because we only need inference, 
# so we don't need the custom focal loss function definition here.
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model load failed: {e}")
    print(f"   Please ensure '{MODEL_PATH}' exists.")
    model = None

def preprocess_image(image):
    """
    Preprocess image for model inference.
    """
    if image is None:
        return None
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Rescale to [0, 1] (as done in ml/data_loader.py)
    # Note: Keras img_to_array returns float32 0-255 range
    img_array = img_array / 255.0
    
    return img_array

def predict_fn(image):
    """
    Prediction function for Gradio.
    """
    if model is None:
        return {"Error": "Model not found. Please train the model explicitly."}
    
    try:
        # Preprocess
        processed_img = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_img)
        scores = predictions[0]
        
        # Format results for Gradio (Label -> Probability)
        results = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
        
        return results
    except Exception as e:
        return {f"Error: {str(e)}": 0.0}

# Create Gradio Interface
with gr.Blocks(title="Cervical Cancer Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 Cervical Cancer Cell Classification AI
        
        This AI model analyzes cytology images to detect cervical cancer precursor lesions.
        
        **Classes:**
        - **Dyskeratotic**: Abnormal cells (squamous intraepithelial lesion)
        - **Koilocytotic**: HPV-infected cells
        - **Metaplastic**: Normal metaplastic cells
        - **Parabasal**: Normal parabasal cells
        - **Superficial-Intermediate**: Normal superficial/intermediate cells
        
        **Medical Disclaimer**: This tool is for research purposes only and not for clinical diagnosis.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Cytology Image")
            predict_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Prediction Result")
            
    # Example images (if you have them)
    # gr.Examples(examples=["test_image.jpg"], inputs=input_image)
    
    predict_btn.click(
        fn=predict_fn,
        inputs=input_image,
        outputs=output_label
    )
    
    gr.Markdown(
        """
        ---
        **Target Performance:** Sensitivity ≥95%, Specificity ≥94%
        """
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
