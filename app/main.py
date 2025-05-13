
import streamlit as st
from PIL import Image
from utils import predict_image


st.set_page_config(page_title="Smart Waste Bin", layout="centered")
st.title("♻️ Smart Waste Bin")
st.write("Upload a waste image to classify it into categories like recyclable, organic, or general waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    label = predict_image(image)
    st.success(f"Predicted Waste Category: **{label}**")
