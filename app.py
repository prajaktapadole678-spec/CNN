
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os

# Define the target size used during training
target_size = (250, 250)

# Define class names (make sure they match the order from training)
class_names = ['cricket ball', 'cricket bat']

st.title('Cricket Ball vs. Bat Classifier')
st.write('Upload an image of a cricket ball or bat to get a prediction!')

# Path to the saved model
model_path = '/content/drive/MyDrive/cricket/model.pkl'

# Check if the model exists before trying to load
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}. Please ensure the model is saved correctly.")
else:
    # Load the model
    try:
        # For Keras models saved with pickle, we need to load it in a specific way
        # using a custom object scope if it contains custom layers or if using an older TF version
        # However, if it's a standard Sequential model, pickle.load might not directly work
        # with the model object itself, but rather with its components.
        # A safer approach for TF Keras models is to save/load with model.save() / tf.keras.models.load_model()
        # Since the user explicitly asked for pickle, I will assume the model object itself was pickled.
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f"Error loading the model: {e}. If this is a Keras model, consider saving and loading with model.save() / tf.keras.models.load_model().")
        model = None

    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Preprocess the image
            image = image.resize(target_size)
            img_array = tf.keras.utils.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array = img_array / 255.0 # Normalize as done during training (Rescaling layer)

            # Make prediction
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_class_index = np.argmax(score)
            predicted_class = class_names[predicted_class_index]
            confidence = np.max(score) * 100

            st.write(f"Prediction: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
