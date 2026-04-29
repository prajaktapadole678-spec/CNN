import streamlit as st
import numpy as np
from PIL import Image
import os

# Try lightweight runtime first (better for Streamlit Cloud)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Page config
st.set_page_config(page_title="Cricket Classifier", layout="centered")

# Constants
MODEL_PATH = "quantized_model.tflite"
TARGET_SIZE = (250, 250)
CLASS_NAMES = ['cricket ball', 'cricket bat']

st.title("🏏 Cricket Ball vs Bat Classifier")
st.write("Upload an image to classify it as a cricket ball or bat.")

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

if interpreter is None:
    st.error("❌ Model file not found. Make sure 'quantized_model.tflite' is in the repo.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(TARGET_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Handle quantization
    input_scale, input_zero_point = input_details[0]['quantization']

    if input_scale > 0:
        img_array = img_array / input_scale + input_zero_point
        img_array = img_array.astype(input_details[0]['dtype'])
    else:
        img_array = img_array.astype(np.float32) / 255.0

    # Run inference
    with st.spinner("🔍 Classifying..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    output_scale, output_zero_point = output_details[0]['quantization']

    if output_scale > 0:
        predictions = (output_data[0] - output_zero_point) * output_scale
    else:
        predictions = output_data[0]

    # Softmax (for safety)
    exp_preds = np.exp(predictions - np.max(predictions))
    predictions = exp_preds / np.sum(exp_preds)

    # Results
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Class Probabilities")
    for i, prob in enumerate(predictions):
        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

    # Bar chart
    st.bar_chart(predictions)
