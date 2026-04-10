import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
from tf_keras_vis.attentions import EigenCAM
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# ---------- CONFIGURATION ----------
st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom background color and style
page_bg_img = '''
    <style>
    body {
        background-color: #3e2723;
        color: #fbe9e7;
    }
    .stApp {
        background-color: #3e2723;
        color: #fbe9e7;
    }
    .css-1v0mbdj.edgvbvh3 {
        background-color: #3e2723;
    }
    .stButton>button {
        background-color: #5d4037;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stMetric {
        color: #ffccbc;
    }
    </style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- LOADING MODEL ----------
class_names = ["Healthy", "Doubtful", "Mild", "Moderate", "Severe"]
model = tf.keras.models.load_model(r"H:\19thAPr2025\IISTproject\EfficientNetV2S_checkpoint.hdf5")
target_size = (300, 300)

# ---------- IMAGE UTILITY ----------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB").resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.expand_dims(img_array, axis=0)
    input_array = tf.keras.applications.efficientnet_v2.preprocess_input(input_array.astype(np.float32))
    return image, input_array, img_array

# ---------- EigenCAM ----------
def get_eigen_cam(model, img_array, predicted_class_index):
    score = CategoricalScore([predicted_class_index])
    replace2linear = ReplaceToLinear()
    cam = EigenCAM(model, model_modifier=replace2linear, clone=True)
    cam_mask = cam(score, img_array, penultimate_layer="top_conv")[0]  # Replace layer if needed
    return cam_mask

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# ---------- SIDEBAR ----------
st.sidebar.title("Final Project - MDC013")
st.sidebar.image("https://i.ibb.co/zQmXD2m/knee.png", width=200)
st.sidebar.markdown("**:orange[By: Fernanda Rodriguez]**")
uploaded_file = st.sidebar.file_uploader("📤 Upload a Knee X-ray Image", type=["jpg", "jpeg", "png"])

# ---------- MAIN ----------
st.title(":orange[Severity Analysis of Arthrosis in the Knee Using EigenCAM]")

if uploaded_file is not None:
    # Process input image
    display_img, input_array, raw_img_array = preprocess_image(uploaded_file)

    # Predict
    preds = model.predict(input_array)
    predicted_class_index = np.argmax(preds)
    predicted_class = class_names[predicted_class_index]
    probability = preds[0][predicted_class_index] * 100

    # Display original image
    st.subheader("🖼️ Original Image")
    st.image(display_img, width=300)

    # Display prediction
    st.subheader("📌 Prediction Result")
    st.metric(
        label=":orange[Severity Grade]",
        value=f"{predicted_class} - {probability:.2f}%",
    )

    # Generate EigenCAM heatmap
    st.subheader("🔍 EigenCAM Visualization")
    heatmap = get_eigen_cam(model, input_array, predicted_class_index)
    cam_img = overlay_heatmap(raw_img_array, heatmap)
    st.image(cam_img, caption="EigenCAM Overlay", width=300)
else:
    st.info("👈 Upload a knee X-ray image from the sidebar to get started.")


    "C:\Users\ROHAN\AppData\Local\Microsoft\WindowsApps\python.exe"-m pip install tf-keras-vis
