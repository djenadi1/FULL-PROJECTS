# Name: Mohammed Djenadi

import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import streamlit as st


# -------------------------
# MODEL
# -------------------------

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=64
            ),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# -------------------------
# CONFIG
# -------------------------

DEVICE = "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "assets", "weights.pth")

classes = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}

rev_class = {v: k for k, v in classes.items()}


# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource  # streamlit reruns the script on every interaction, so load the model only once
def load_model():
    model = CNN(num_classes=10)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    _ = model.load_state_dict(state_dict)
    _ = model.eval()
    return model


model = load_model()


# -------------------------
# TRANSFORMS
# -------------------------

transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# -------------------------
# UI
# -------------------------

st.set_page_config(
    page_title="Satellite Image Classifier",
    layout="wide"
)

st.header("Satellite Image Classification")

file = st.file_uploader(
    "Upload a 64x64 satellite image",
    type=["jpg", "jpeg", "png"]
)


# -------------------------
# PREDICTION
# -------------------------

def probability_table():

    if not file:
        return pd.DataFrame()

    image = Image.open(file).convert("RGB")

    x = transformer(image)
    x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    df = pd.DataFrame({
        "Class": [rev_class[i] for i in range(10)],
        "Probability": probs.numpy()
    })

    df = df.sort_values(
        by="Probability",
        ascending=False
    )

    return df


def prediction_result():

    if not file:
        st.write("Upload an image to see predictions.")
        return

    image = Image.open(file).convert("RGB")

    x = transformer(image)
    x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1))
    pred_class = rev_class[pred_idx]
    confidence = float(probs[0][pred_idx]) * 100

    st.subheader("Prediction")
    st.divider()
    st.write(f"Predicted Class: {pred_class}")
    st.write(f"Confidence: {confidence:.2f}%")


def uploaded_image():

    if not file:
        return

    st.image(file, width=300)


st.dataframe(probability_table(), hide_index=True)
prediction_result()
uploaded_image()
