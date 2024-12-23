import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Users\user\Downloads\facial_expression_model_fine_tuned.keras")  # Update with your file path

# Class labels
class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']  # Modify as per your model

# Streamlit App
st.set_page_config(page_title="Facial Expression Recognizer", page_icon="😊", layout="centered")

st.title("🤖 Facial Expression Recognizer")
st.write("Upload or drag & drop an image, and the model will predict the facial expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.write("Uploaded Image:")
    img = Image.open(uploaded_file)

    # Preprocess the image for display
    display_img = img.copy()
    st.image(display_img, caption='Uploaded Image', use_column_width=False, width=300)  # Resize the displayed image

    # Preprocess the image for the model
    img_resized = img.resize((128, 128))  # Resize to model input size
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    predicted_label = class_labels[class_idx]
    confidence = predictions[0][class_idx]

    # Display prediction above the image
    st.write("Prediction:")
    st.markdown(f"<h3 style='text-align: center; color: green;'>Prediction: {predicted_label} (Confidence: {confidence:.2f})</h3>", unsafe_allow_html=True)

    # Annotate prediction on the image
    draw = ImageDraw.Draw(display_img)
    font = ImageFont.load_default()  # You can specify a custom font file for better styling
    text = f"{predicted_label} ({confidence:.2f})"

    # Calculate text size and draw background
    text_size = draw.textbbox((0, 0), text, font=font)  # Get bounding box for the text
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    text_position = (10, 10)
    draw.rectangle(
        [text_position, (text_position[0] + text_width + 5, text_position[1] + text_height + 5)],
        fill="white"
    )
    draw.text(text_position, text, fill="black", font=font)

    # Display annotated image with reduced size
    st.image(display_img, caption="Image with Prediction", use_column_width=False, width=300)
