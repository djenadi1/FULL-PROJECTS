# Satellite Image Classifier

This was my final coding project for **Programming in Python II** at JKU. The task was to train a CNN that classifies 64x64 satellite images into 10 land use classes and then build a small app around it.

The code is written in a student style since it was a course project: it follows the course template, has some repeated cells and leftover comments, and isn't cleaned up like my other projects. I left it mostly as I handed it in.

## What it does

- Loads 10,000 labeled satellite images (1,000 per class) from `data/`
- Plots some samples, the average pixel distribution and the brightness per class
- Splits 80/20 per class and trains a small CNN in PyTorch with augmentation and early stopping
- Best validation accuracy: **97.30%**
- Predicts the unlabeled images in `test_data/` and writes `submission.csv` for the course challenge server
- A Streamlit app where you upload an image and get the predicted class and the probability for every class

Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Files

```
Satellite/
├── satellite_classifier.ipynb   # data, plots, training, evaluation, submission
├── app.py                       # Streamlit app
├── assets/
│   ├── weights.pth              # trained model weights
│   └── *.png                    # plots from the notebook
├── data/                        # training images, one folder per class
├── test_data/                   # unlabeled images for the submission
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The notebook expects to be started from this folder so `data/`, `test_data/` and `assets/` are found.
