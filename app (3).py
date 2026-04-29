
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os

# Define the target size used during training
target_size = (250, 250)

# Define class names (make sure they match the order from training)
class_names = ['cricket ball', 'cricket bat'] # Assuming these are your class names

st.title('Cricket Ball vs. Bat Classifier (Quantized TFLite)')
st.write('Upload an image of a cricket ball or bat to get a prediction!')

# Path to the saved quantized TFLite model bytes in .pkl format
model_path = '/content/drive/MyDrive/cricket/quantized_model.pkl'

interpreter = None
input_details = None
output_details = None

# Check if the model exists before trying to load
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}. Please ensure the quantized model is saved correctly.")
else:
    # Load the TFLite model bytes from the .pkl file
    try:
        with open(model_path, 'rb') as file:
            tflite_model_bytes = pickle.load(file)
        
        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        st.success('Quantized TFLite model loaded successfully!')
    except Exception as e:
        st.error(f"Error loading or initializing TFLite model: {e}.")
        interpreter = None

if interpreter:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = image.resize(target_size)
        img_array = np.array(image, dtype=np.float32) # TFLite models often expect float32
        img_array = np.expand_dims(img_array, axis=0) # Create a batch
        img_array = img_array / 255.0 # Normalize

        # Set the tensor to be the image
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax if needed (TFLite models might not include it)
        # If the model output is logits, apply softmax to get probabilities
        if output_details[0]['dtype'] == np.float32:
            predictions = tf.nn.softmax(output_data[0]).numpy()
        else:
            # If output is already probabilities or integer quantized output
            predictions = output_data[0]

        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(predictions) * 100

        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
