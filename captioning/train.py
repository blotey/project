import os
import numpy as np
import string
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# --- SETTINGS ---
MAX_LENGTH = 34
VOCAB_SIZE = 10000
EPOCHS = 20
BATCH_SIZE = 32
CAPTIONS_FILE = 'data/captions.txt'
IMAGES_PATH = 'data/images/'

# --- LOAD AND CLEAN DATA ---
def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        lines = f.readlines()
    captions = {}
    for line in lines:
        img, cap = line.strip().split('|')
        img = img.strip()
        cap = cap.strip().lower().translate(str.maketrans('', '', string.punctuation))
        cap = f"startseq {cap} endseq"
        if img not in captions:
            captions[img] = []
        captions[img].append(cap)
    return captions

# --- IMAGE PREPROCESSING ---
def preprocess_image(img_path):
    img = Image.open(img_path).resize((299, 299))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = preprocess_input(img)
    return img

def extract_features(images_dir):
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    features = {}
    for img_name in tqdm(os.listdir(images_dir)):
        path = os.path.join(images_dir, img_name)
        img = preprocess_image(path)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature.flatten()
    return features

# --- TOKENIZER ---
def create_tokenizer(captions_dict):
    all_captions = []
    for caps in captions_dict.values():
        all_captions.extend(caps)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# --- DATA GENERATOR ---
def data_generator(captions, photos, tokenizer, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []
        for key, desc_list in captions.items():
            photo = photos[key]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = np.array(out_seq)

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

                    if len(X1) == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = [], [], []

# --- TRANSFORMER COMPONENTS ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({"position": self.pos_encoding.shape[0], "d_model": self.pos_encoding.shape[1]})
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha(x, x, x, attention_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2 = self.mha(out1, enc_output, enc_output, attention_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn_output = self.ffn(out2)
        return LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

# --- BUILD TRANSFORMER MODEL ---
def build_transformer_model(vocab_size, max_length, d_model=256, num_heads=4, dff=512):
    image_input = Input(shape=(2048,))
    img_features = Dense(d_model, activation='relu')(image_input)
    img_features = Reshape((1, d_model))(img_features)  # (batch, 1, d_model)

    caption_input = Input(shape=(max_length,))
    x = Embedding(vocab_size, d_model, mask_zero=True)(caption_input)
    x = PositionalEncoding(max_length, d_model)(x)

    transformer_block = TransformerDecoderBlock(d_model, num_heads, dff)
    x = transformer_block(x, img_features, training=True)

    x = Dropout(0.5)(x)
    x = Dense(d_model, activation='relu')(x)
    output = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# --- MAIN ---
def main():
    print("Loading captions...")
    captions_dict = load_captions(CAPTIONS_FILE)

    print("Extracting features...")
    features = extract_features(IMAGES_PATH)

    print("Preparing tokenizer...")
    tokenizer = create_tokenizer(captions_dict)
    max_length = MAX_LENGTH
    vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

    print("Building model...")
    model = build_transformer_model(vocab_size, max_length)

    print("Preparing data generator...")
    steps = sum(len(c) for c in captions_dict.values())
    generator = data_generator(captions_dict, features, tokenizer, max_length, vocab_size, BATCH_SIZE)

    print("Training model...")
    checkpoint = ModelCheckpoint('caption_model.h5', monitor='loss', save_best_only=True)
    model.fit(generator, epochs=EPOCHS, steps_per_epoch=steps // BATCH_SIZE, callbacks=[checkpoint])

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Training complete. Model and tokenizer saved.")

if __name__ == '__main__':
    main()
