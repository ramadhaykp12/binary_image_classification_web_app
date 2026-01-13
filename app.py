import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# Replace these with your actual class names
CLASS_NAMES = ["Cat", "Dog"] 
MODEL_PATH = "trained_model.onnx"

st.set_page_config(page_title="ONNX Image Classifier", layout="centered")

# --- FUNCTIONS ---

@st.cache_resource
def load_session(model_path):
    """Loads the ONNX model and caches it to improve performance."""
    try:
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # Ensure image is RGB
    img = image.convert("RGB")
    # Resize to match model input
    img = img.resize((64, 64))
    # Convert to numpy and normalize
    img_array = np.array(img).astype('float32') / 255.0
    # Add batch dimension (Batch, Height, Width, Channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- USER INTERFACE ---

st.title("Binary Image Classifier")
st.info("Upload an image to see the ONNX model's prediction.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 2. Load model
    session = load_session(MODEL_PATH)
    
    if session:
        # 3. Predict Button
        if st.button("Classify Image"):
            with st.spinner("Analyzing..."):
                # Preprocess
                input_tensor = preprocess_image(image)
                input_name = session.get_inputs()[0].name
                
                # Run Inference
                # output is a list of arrays; we take the first element
                outputs = session.run(None, {input_name: input_tensor})
                raw_prediction = outputs[0][0][0] # Sigmoid value (0.0 to 1.0)
                
                # 4. Process Results
                # Probability > 0.5 is Class 1, else Class 0
                is_class_1 = raw_prediction > 0.5
                prediction_label = CLASS_NAMES[1] if is_class_1 else CLASS_NAMES[0]
                confidence = raw_prediction if is_class_1 else (1.0 - raw_prediction)
                
                # 5. Display Results
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", prediction_label)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Visual probability bar
                st.write(f"Raw Model Output (Sigmoid): `{raw_prediction:.4f}`")
                st.progress(float(raw_prediction))
    else:
        st.warning(f"Please ensure '{MODEL_PATH}' is in the same folder as this script.")

# --- FOOTER ---
st.markdown("---")