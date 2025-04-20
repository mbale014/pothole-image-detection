import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

def load_model(path="best_model.h5"):
    model = tf.keras.models.load_model(path)
    return model

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)

def show_prediction():
    st.title("Pothole Detection")
    st.write("Upload an image of a road or choose a sample image to predict if it contains a pothole.")

    # Option to upload image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    # OR choose from sample images
    st.markdown("**Or select a sample image:**")
    sample_dir = "sample_images"
    sample_options = os.listdir(sample_dir)
    selected_sample = st.selectbox("Choose a sample image", ["None"] + sample_options)

    # Load image based on input method
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif selected_sample != "None":
        image_path = os.path.join(sample_dir, selected_sample)
        image = Image.open(image_path)
        st.image(image, caption=f"Sample: {selected_sample}", use_container_width=True)

    # Prediction
    if image:
        with st.spinner("Making prediction..."):
            model = load_model("best_model.h5")
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0][0]

            if prediction > 0.5:
                st.error("Pothole Detected!")
            else:
                st.success("Normal Road")

            st.write(f"Prediction Confidence: {prediction*100:.2f}%")
