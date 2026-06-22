import os
import gdown
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
import numpy as np

# -------------------------------------------------------
# 1. Safe Monkey-Patch for Keras Dense Layer
#    Uses a guard flag to prevent infinite recursion on Streamlit reruns.
# -------------------------------------------------------
for dense_class in [keras.layers.Dense, tf.keras.layers.Dense]:
    if not hasattr(dense_class, '_patched_for_quantization'):
        original_init = dense_class.__init__
        
        # Using a helper function to cleanly bind the original init method
        def make_patched_init(orig_init):
            def patched_init(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                return orig_init(self, *args, **kwargs)
            return patched_init
            
        dense_class.__init__ = make_patched_init(original_init)
        dense_class._patched_for_quantization = True

# -------------------------------------------------------
# 2. Set up page configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Natural Image Classifier",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Natural Image Classification System")
st.write("Upload an image, and our fine-tuned VGG16 model will instantly predict its category.")

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
        
        # Your verified Google Drive file ID
        file_id = '1qB52CTPBBVDUGuYbgPuhSHncWKE_62jw'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            gdown.download(url, model_path, quiet=False)
            status_text.empty()
        except Exception as e:
            st.error(f"Failed to download model from Google Drive. Error: {e}")
            return None

    # Load the model cleanly
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Failed to parse model file. Error: {e}")
        return None

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