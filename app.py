import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Config
MODEL_PATH = "quantized_model.tflite"
TARGET_SIZE = (250, 250)
CLASS_NAMES = ['cricket ball', 'cricket bat']

st.title("🏏 Cricket Classifier")

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    # Preprocess
    image = image.resize(TARGET_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Softmax using numpy
    exp_preds = np.exp(output_data[0] - np.max(output_data[0]))
    predictions = exp_preds / np.sum(exp_preds)

    pred_index = np.argmax(predictions)
    confidence = predictions[pred_index] * 100

    st.write(f"Prediction: {CLASS_NAMES[pred_index]}")
    st.write(f"Confidence: {confidence:.2f}%")
