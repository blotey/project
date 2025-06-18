import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os
import pickle
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from segmentation.preprocess import preprocess_image as preprocess_seg

# ---- Load Captioning Components ----
@st.cache_resource
def load_captioning_components():
    caption_model = load_model("captioning/caption_model.h5", compile=False)
    with open("captioning/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    encoder = InceptionV3(weights='imagenet')
    encoder = tf.keras.Model(inputs=encoder.input, outputs=encoder.layers[-2].output)
    return caption_model, tokenizer, encoder

# ---- Load Segmentation Model ----
@st.cache_resource
def load_segmentation_model():
    return load_model("segmentation/segmentation_model.h5", compile=False)

caption_model, tokenizer, encoder = load_captioning_components()
segmentation_model = load_segmentation_model()

MAX_LENGTH = 34
os.makedirs("output/masks", exist_ok=True)

# ---- Feature Extraction ----
def extract_image_features(image, encoder):
    img = image.resize((299, 299)).convert('RGB')
    img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
    return encoder.predict(img_array, verbose=0).flatten()

# ---- Generate Caption using Beam Search ----
def generate_caption(feature, tokenizer, model, max_length=34, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([np.array([feature]), padded], verbose=0)
            top_preds = np.argsort(yhat[0, -1])[-beam_index:]

            for word in top_preds:
                new_seq = seq + [word]
                new_score = score + np.log(yhat[0, -1][word] + 1e-9)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_index]

    final = sequences[0][0]
    caption = [tokenizer.index_word.get(i, '') for i in final if i > 0]
    return ' '.join(caption).replace('startseq', '').replace('endseq', '').strip()

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Image Analyzer", layout="wide")
st.title("🧠 Unified Image Captioning + Segmentation App")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)

    with st.spinner("Analyzing image..."):

        # ---- Captioning ----
        features = extract_image_features(image, encoder)
        caption = generate_caption(features, tokenizer, caption_model)

        st.success("📝 Generated Caption:")
        st.markdown(f"<b>{caption}</b>", unsafe_allow_html=True)

        # Save caption
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        with open("output/captions_log.txt", "a") as f:
            f.write(f"{timestamp} | {uploaded_file.name} | {caption}\n")

        # ---- Segmentation ----
        img_array = preprocess_seg(uploaded_file, target_size=(128, 128))
        pred_mask = segmentation_model.predict(np.expand_dims(img_array, axis=0))[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()
        pred_mask_img = Image.fromarray(pred_mask * 255).resize(image.size)
        overlay = ImageEnhance.Brightness(pred_mask_img.convert("RGB")).enhance(0.6)
        blended = Image.blend(image, overlay, alpha=0.4)

        with col2:
            st.image(blended, caption="🟣 Overlay: Segmentation Mask", use_column_width=True)

        # Save mask
        mask_path = f"output/masks/{uploaded_file.name.split('.')[0]}_mask.png"
        pred_mask_img.save(mask_path)
        st.info(f"💾 Mask saved to: {mask_path}")
