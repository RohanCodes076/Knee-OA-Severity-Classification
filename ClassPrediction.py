import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

# === MUST BE THE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="Severity Analysis of Knee Osteoarthritis",
    page_icon="🦵",
    layout="wide",
)

# === Set Background Color to Lavender ===
st.markdown("""
    <style>
    .main {
        background-color: #E6E6FA;
    }
    </style>
    """, unsafe_allow_html=True)

# === GradCAM Heatmap Generation ===
def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

# === Overlay Heatmap on Image ===
def overlay_heatmap(img, heatmap, alpha=0.6):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

# === Class Labels and Model ===
class_names = ["Healthy", "Doubtful", "Mild", "Moderate", "Severe"]
model_path = r"H:\2nd_year_practice\knee_OA_detection_withApp\saved_model.h5"
model = tf.keras.models.load_model(model_path)
last_conv_layer_name = "global_average_pooling2d"  # change based on model architecture

# === Sidebar ===
with st.sidebar:
    st.image(r"H:\2nd_year_practice\knee_OA_detection_withApp\logo1.png", width=50)
    st.title("Knee Osteoarthritis Severity Analysis")
    uploaded_file = st.file_uploader("Upload X-ray Image")

# === Main Section ===
st.title("🦵 Knee Osteoarthritis Severity Detection using Deep Learning")

if uploaded_file is not None:
    # === Load Image ===
    img = Image.open(uploaded_file)
    st.subheader("📷 Input Image")

    # === Extract True Class from Folder Name (e.g., 0, 1, 2, 3, 4) ===
    folder_name = os.path.basename(os.path.dirname(uploaded_file.name))
    try:
        true_class_index = int(folder_name)
        true_class = class_names[true_class_index]
    except:
        true_class = "Unknown"

    # === Preprocessing ===
    img_resized = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    original_img = img_array.copy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # === Prediction ===
    with st.spinner("🧠 Analyzing..."):
        preds = model.predict(img_array)
    pred_class_index = np.argmax(preds)
    pred_class = class_names[pred_class_index]
    pred_confidence = preds[0][pred_class_index] * 100

    # === GradCAM Heatmap ===
    heatmap = make_gradcam_heatmap(model, img_array, last_conv_layer_name)
    gradcam_img = overlay_heatmap(original_img, heatmap)

    # === Side-by-Side Display ===
    st.subheader("🩺 Prediction and GradCAM Output")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_img.astype("uint8"), caption="Original X-ray", use_column_width=True)
        st.markdown(f"**Predicted:** `{pred_class}` ({pred_confidence:.2f}%)  \n**True:** `{true_class}`")

    with col2:
        st.image(gradcam_img, caption="GradCAM Overlay", use_column_width=True)
        st.markdown(f"**Activation Layer:** `{last_conv_layer_name}`")

    # === Horizontal Confidence Bar ===
    st.subheader("📊 Class Confidence Scores")
    score_cols = st.columns(len(class_names))
    for i, col in enumerate(score_cols):
        col.metric(label=class_names[i], value=f"{preds[0][i]*100:.2f}%")
