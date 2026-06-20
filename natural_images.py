import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Set up page configuration
st.set_page_config(
    page_title="Natural Image Classifier",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Natural Image Classification System")
st.write("Upload an image, and our fine-tuned VGG16 model will instantly predict its category.")

# 2. Load the trained model (Cached so it only loads once into memory)
@st.cache_resource
def load_champion_model():
    return tf.keras.models.load_model('best_natural_images_model.keras')

with st.spinner("Loading deep learning model... Please wait."):
    model = load_champion_model()

# 3. Define the exact 8 classes from the Natural Images Dataset
CLASS_NAMES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# 4. File Uploader UI Element
uploaded_file = st.file_uploader("Drop an image here or click to browse...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("⚙️ Processing image and running inference...")
    
    # 5. Image Preprocessing matching our training pipeline
    if image.mode != "RGB":
        image = image.convert("RGB") # Ensures PNG alpha channels don't break the model
        
    resized_image = image.resize((224, 224))
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0) # Changes shape to (1, 224, 224, 3)
    
    # 6. Run Prediction
    # (Note: VGG16 preprocessing steps are safely embedded inside the saved model layers)
    predictions = model.predict(img_array)
    probabilities = predictions[0]
    
    best_class_idx = np.argmax(probabilities)
    predicted_label = CLASS_NAMES[best_class_idx]
    confidence_score = probabilities[best_class_idx] * 100
    
    # 7. Display Results cleanly
    st.success(f"**Prediction:** {predicted_label.upper()}")
    st.info(f"**Confidence Level:** {confidence_score:.2f}%")
    
    # Show confidence distribution breakdown
    st.write("### Class Probability Breakdown:")
    chart_data = dict(zip(CLASS_NAMES, [float(p) for p in probabilities]))
    st.bar_chart(chart_data)