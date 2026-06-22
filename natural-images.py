import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Dense   # Needed to subclass Dense

# -------------------------------------------------------
# 1. Set up page configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Natural Image Classifier",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Natural Image Classification System")
st.write("Upload an image, and our fine-tuned VGG16 model will instantly predict its category.")

# -------------------------------------------------------
# 2. Custom Dense layer to ignore 'quantization_config'
#    (fixes deserialisation error with older TensorFlow)
# -------------------------------------------------------
class CustomDense(Dense):
    @classmethod
    def from_config(cls, config):
        # Remove the problematic key introduced in newer TF
        config.pop('quantization_config', None)
        return super().from_config(config)

# -------------------------------------------------------
# 3. Model downloader & loader (cached)
# -------------------------------------------------------
@st.cache_resource
def load_champion_model():
    model_path = 'best_natural_images_model.keras'
    
    # Download model from Google Drive if not present
    if not os.path.exists(model_path):
        status_text = st.empty()
        status_text.info("📥 Initial setup: Downloading the deep learning model from Google Drive... (This takes about 10-15 seconds)")
        
        # 🚨 REPLACE THE STRING BELOW WITH YOUR ACTUAL GOOGLE DRIVE FILE ID 🚨
        file_id = '1qB52CTPBBVDUGuYbgPuhSHncWKE_62jw'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            gdown.download(url, model_path, quiet=False)
            status_text.empty()  # Clear download text
        except Exception as e:
            st.error(f"Failed to download model from Google Drive. Error: {e}")
            return None
    
    # Load the model with our custom Dense class and skip compilation
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'Dense': CustomDense},
        compile=False          # Avoids optimizer/loss version mismatches
    )

model = load_champion_model()

# -------------------------------------------------------
# 4. Define the exact 8 classes from the Natural Images Dataset
# -------------------------------------------------------
CLASS_NAMES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# -------------------------------------------------------
# 5. File uploader & inference
# -------------------------------------------------------
uploaded_file = st.file_uploader("Drop an image here or click to browse...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("⚙️ Processing image and running inference...")
    
    # Preprocess the image exactly as during training
    if image.mode != "RGB":
        image = image.convert("RGB")  # handle PNG alpha channels
    
    resized_image = image.resize((224, 224))
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    
    # Run prediction
    predictions = model.predict(img_array)
    probabilities = predictions[0]
    
    best_class_idx = np.argmax(probabilities)
    predicted_label = CLASS_NAMES[best_class_idx]
    confidence_score = probabilities[best_class_idx] * 100
    
    # Display results
    st.success(f"**Prediction:** {predicted_label.upper()}")
    st.info(f"**Confidence Level:** {confidence_score:.2f}%")
    
    # Show confidence distribution breakdown
    st.write("### Class Probability Breakdown:")
    chart_data = dict(zip(CLASS_NAMES, [float(p) for p in probabilities]))
    st.bar_chart(chart_data)