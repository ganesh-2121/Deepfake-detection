import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('deepfake_detection_model.h5')

# Preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (96,96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict if the image is fake or real
def predict_image(image):
    processed_image = preprocess_image(image)
    probability = model.predict(processed_image)[0][0]
    label = "Fake" if probability < 0.5 else "Real"
    return label, probability

# Streamlit interface
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: grey;'>🔍 Deepfake Detection in Social Media Content</h1>", unsafe_allow_html=True)
st.image("coverpage.png", use_column_width=True)

st.header("🧠 About Deepfakes")
st.write("""
Deepfakes are AI-generated images and videos that look incredibly realistic but are completely fake.
They can be used maliciously to impersonate people, spread misinformation, and erode trust in digital media.
Our deep learning model helps detect whether a given image is real or fake with high accuracy.
""")

# --- Model Performance Summary Section ---
with st.expander("📊 Model Performance Summary", expanded=True):
    st.markdown("""
    <div style='font-size:16px'>
    ✅ <b>Training Dataset:</b> 80,000 images (40k Real, 40k Fake)<br>
    ✅ <b>Validation Dataset:</b> 20,000 images (10k Real, 10k Fake)<br><br>
    🧠 <b>Model:</b> Convolutional Neural Network (CNN)<br>
    📈 <b>Validation Accuracy:</b> 93.98%<br>
    📉 <b>Validation Loss:</b> 0.1577<br>
    🕐 <b>Epochs:</b> 10<br>
    💡 <b>Optimizer:</b> Adam (lr=0.0001)<br>
    </div>
    """, unsafe_allow_html=True)


# Upload image
uploaded_file = st.file_uploader("📤 Upload an image for analysis", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # ✅ Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width = 300)

    # Run prediction
    label, confidence = predict_image(image)

    # Set style
    color = "green" if label == "Real" else "red"
    st.markdown(f"<h2 style='color:{color};'>🧾 Prediction: {label}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'>📈 Confidence: <b>{confidence:.4f}</b></p>", unsafe_allow_html=True)

    # Extra detail
    if label == "Fake":
        st.write("The image is likely fake due to subtle irregularities like uneven skin texture, blurred facial features, or unnatural lighting. These signs are common in deepfakes. The model detects such inconsistencies, comparing them with real examples, and flags the image as manipulated based on learned deepfake patterns. ")
    else:
        st.write("✅ The image appears real due to natural facial features, proper lighting, and consistent symmetry. There are no visible distortions, and textures like skin, eyes, and shadows look authentic. Our model recognizes these as traits of genuine images based on patterns learned during training.")

