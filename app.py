import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
import os

# Load the pre-trained model
model = load_model("caption_model_new.keras", compile=False)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define the maximum length of the caption
max_length = 37

# Load ResNet50 CNN model
cnn_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

# Function to extract image features
def extract_normalized_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array, verbose=0)
    normalized = tf.nn.l2_normalize(features, axis=1)
    return normalized.numpy().flatten()

# Convert index to word using tokenizer
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict caption for an image
def predict_caption(caption_model, image_vector, tokenizer, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        yhat = caption_model.predict([image_vector, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break
        if word in ['<end>', 'end']:
            break

        in_text += ' ' + word

    return in_text.replace('<start>', '').replace('<end>', '').strip()

# Streamlit App
def main():
    st.title("🖼️ Image Caption Generator")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save to temp location
        img_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Feature Extraction and Prediction
        vector = extract_normalized_features(img_path).reshape(1, -1)
        caption = predict_caption(model, vector, tokenizer, max_length)
        st.markdown(f"**Predicted Caption:** {caption}")

if __name__ == "__main__":
    main()
