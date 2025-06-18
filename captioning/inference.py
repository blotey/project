import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import sys

# --- SETTINGS ---
MAX_LENGTH = 34
VOCAB_SIZE = 10000
MODEL_PATH = 'caption_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'

# --- Load Tokenizer ---
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# --- Load Transformer Model ---
model = load_model(MODEL_PATH, compile=False)

# --- Load CNN Encoder (InceptionV3) ---
def build_encoder():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

encoder = build_encoder()

# --- Image Preprocessing ---
def preprocess_image(img_path):
    img = Image.open(img_path).resize((299, 299))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# --- Feature Extraction ---
def extract_features(image_path):
    img = preprocess_image(image_path)
    feature = encoder.predict(img, verbose=0)
    return feature.flatten()

# --- Generate Caption (Greedy) ---
def generate_caption_beam_search(feature, tokenizer, max_length, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            sequence = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([np.array([feature]), sequence], verbose=0)
            top_preds = np.argsort(yhat[0, -1])[-beam_index:]

            for word in top_preds:
                next_seq = seq + [word]
                next_score = score + np.log(yhat[0, -1][word] + 1e-9)
                all_candidates.append([next_seq, next_score])

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_index]

    final = sequences[0][0]
    result = [tokenizer.index_word.get(i, '') for i in final if i > 0]
    return ' '.join(result).replace('startseq', '').replace('endseq', '').strip()


# --- Main Execution ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        return

    image_path = sys.argv[1]
    print(f"Generating caption for: {image_path}")

    feature = extract_features(image_path)
    caption = generate_caption_beam_search(feature, tokenizer, max_length, beam_index=3):
    print("\nGenerated Caption:", caption)

if __name__ == '__main__':
    main()
