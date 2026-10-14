import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


# paths relative to this file so it works on streamlit cloud too
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# these get np.log1p in training (big counts squashed), so we do the same here
LOG_COLS = ['price', 'positive', 'negative', 'peak_ccu', 'average_playtime_forever', 'median_playtime_forever',
            'average_playtime_2weeks', 'median_playtime_2weeks', 'achievements', 'dlc_count', 'owners_mid', 'n_languages']


# same model class as in the training notebook
class GenCatMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024, p=0.5):
        super().__init__()

        def block(ins, outs):
            return nn.Sequential(nn.Linear(ins, outs), nn.ReLU(), nn.Dropout(p))

        self.net = nn.Sequential(
            block(in_dim, hidden),
            block(hidden, hidden // 2),
            block(hidden // 2, hidden // 4),
            nn.Linear(hidden // 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# the model was trained on scaled data, so we load the saved scaler
def get_scaler():
    return joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))


# the labels csv has a header row first, the names are in the second row
def get_labels(path):
    return pd.read_csv(path, header=None, skiprows=1).iloc[0].tolist()


# cache so streamlit doesnt reload everything every time we click something
@st.cache_resource
def load_everything():
    # fail early if something is missing
    required_files = [
        os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2", "config_sentence_transformers.json"),
        os.path.join(BASE_DIR, "models", "scaler.pkl"),
        os.path.join(BASE_DIR, "labels", "feature_columns.csv"),
        os.path.join(BASE_DIR, "labels", "labels_genres.csv"),
        os.path.join(BASE_DIR, "labels", "labels_gencat.csv"),
        os.path.join(BASE_DIR, "models", "best_genres.pth"),
        os.path.join(BASE_DIR, "models", "best_gencat.pth"),
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    encoder = SentenceTransformer(os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2"))
    feature_columns = pd.read_csv(
        os.path.join(BASE_DIR, "labels", "feature_columns.csv"), header=None
    )[0].tolist()
    scaler = get_scaler()

    # name embedding + description embedding + the tabular stuff
    in_dim = 384 + 384 + len(feature_columns)

    genres_labels = get_labels(os.path.join(BASE_DIR, "labels", "labels_genres.csv"))
    model_genres = GenCatMLP(in_dim, len(genres_labels))
    model_genres.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "models", "best_genres.pth"), map_location="cpu")
    )
    model_genres.eval()

    gencat_labels = get_labels(os.path.join(BASE_DIR, "labels", "labels_gencat.csv"))
    model_gencat = GenCatMLP(in_dim, len(gencat_labels))
    model_gencat.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "models", "best_gencat.pth"), map_location="cpu")
    )
    model_gencat.eval()

    return encoder, feature_columns, scaler, model_genres, genres_labels, model_gencat, gencat_labels


try:
    (encoder, feature_columns, scaler,
     model_genres, genres_labels,
     model_gencat, gencat_labels) = load_everything()
except FileNotFoundError as e:
    st.error(f"Missing required file: {e}")
    st.stop()


# ---------- page ----------
st.title("Steam game genre predictor")

# toggle between the two models
choice = st.radio("Which model?", ["Genres only", "Genres + categories"], horizontal=True)

name = st.text_input("Game name")
description = st.text_area("Short description (optional, but it helps a lot)",
                           placeholder="A roguelike dungeon crawler where you ...")
release_year = st.number_input("Release year", 1997, 2030, 2024)
release_month = st.slider("Release month", 1, 12, 6)
price = st.number_input("Price (USD)", 0.0, 200.0, 19.99)
price_status = st.selectbox("Price status", ["free", "paid", "unavailable"])
owners_mid = st.number_input("Estimated owners", 0, 500_000_000, 50_000)
positive = st.number_input("Positive reviews", 0, 10_000_000, 0)
negative = st.number_input("Negative reviews", 0, 10_000_000, 0)
peak_ccu = st.number_input("Peak concurrent users", 0, 1_000_000, 0)
metacritic_score = st.slider("Metacritic score", -1, 100, 70)
average_playtime_forever = st.number_input("Avg playtime forever", 0, 100_000, 0)
median_playtime_forever = st.number_input("Median playtime forever", 0, 100_000, 0)
average_playtime_2weeks = st.number_input("Avg playtime 2 weeks", 0, 10_000, 0)
median_playtime_2weeks = st.number_input("Median playtime 2 weeks", 0, 10_000, 0)
achievements = st.number_input("Achievements", 0, 10_000, 0)
dlc_count = st.number_input("DLC count", 0, 1_000, 0)
n_languages = st.number_input("Supported languages", 0, 100, 1)
windows = int(st.checkbox("Windows", value=True))
mac = int(st.checkbox("macOS"))
linux = int(st.checkbox("Linux"))


if st.button("Predict"):
    if name.strip() == "":
        st.warning("Please type a game name first")
        st.stop()

    # turn the name into a vector
    # same lowercase/strip as the cleaning notebook
    name_vec = encoder.encode(name.strip().lower(), normalize_embeddings=True)
    desc_vec = encoder.encode(description.strip(), normalize_embeddings=True)

    # put all the inputs in one row
    row = pd.DataFrame([{
        "release_month": f"{release_month:02d}",
        "release_year": release_year,
        "price": price,
        "price_status": price_status,
        "owners_mid": owners_mid,
        "positive": positive,
        "negative": negative,
        "peak_ccu": peak_ccu,
        "metacritic_score": metacritic_score,
        "average_playtime_forever": average_playtime_forever,
        "median_playtime_forever": median_playtime_forever,
        "average_playtime_2weeks": average_playtime_2weeks,
        "median_playtime_2weeks": median_playtime_2weeks,
        "achievements": achievements,
        "dlc_count": dlc_count,
        "windows": windows,
        "mac": mac,
        "linux": linux,
        "n_languages": n_languages,
    }])
    row[LOG_COLS] = np.log1p(row[LOG_COLS].astype("float32"))

    # one hot encode like in training, then make sure columns are in the same order
    row = pd.get_dummies(row, columns=["release_month", "price_status"],
                         prefix=["m_", "status_"])
    row = row.reindex(columns=feature_columns, fill_value=0)

    # embeddings first (name, then description), then the other features (same as training)
    x = np.concatenate([name_vec, desc_vec, row.to_numpy(dtype="float32").flatten()])
    x = scaler.transform(x.reshape(1, -1)).astype("float32")
    x = torch.from_numpy(x)

    # pick the model based on the toggle
    if choice == "Genres only":
        model = model_genres
        labels = genres_labels
    else:
        model = model_gencat
        labels = gencat_labels

    with torch.inference_mode():
        probs = torch.sigmoid(model(x)).squeeze(0).numpy()

    # top 10
    top = np.argsort(probs)[::-1][:10]
    st.subheader("Top 10 predictions")
    for i in top:
        label = labels[i].replace("genres:", "").replace("categories:", "")
        st.write(f"{label}: {probs[i]:.3f}")