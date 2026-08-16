# FULL-PROJECTS


# Superstore Discount Optimizer

An end-to-end machine learning project that uses historical retail data to recommend discounts based on a desired profitability level.

The project covers the full workflow from **SQL data cleaning and analysis** to **feature engineering, PyTorch model training, evaluation, and deployment with Streamlit**.

## Overview

Discounts can increase sales but can also significantly reduce profitability.

This project trains a neural network to classify an order into one of three profitability levels:

* **Loss**
* **Medium Profit**
* **High Profit**

The trained model is then used inside an interactive application to determine which discount ranges are associated with each predicted profit level.

Instead of only answering:

> *What profit class will this order have?*

the final application answers:

> *What discount should I use if I want this order to remain within a certain profit class?*

## How It Works

The project follows this pipeline:

```text
Raw Superstore Data
        ↓
SQL Cleaning & Analysis
        ↓
Feature Engineering
        ↓
One-Hot Encoding
        ↓
PyTorch Neural Network
        ↓
Profit Classification
        ↓
Discount Range Search
        ↓
Streamlit Application
```

For a user-defined order, the application tests discounts from **0% to 80%** in increments of **0.1 percentage points**.

Each discount is passed through the trained model and classified as:

```text
Loss | Medium Profit | High Profit
```

The predictions are grouped into continuous ranges.

For example:

```text
0.0% – 18.7%   → High Profit
18.8% – 34.2%  → Medium Profit
34.3% – 80.0%  → Loss
```

The user chooses their desired profit level and the application recommends the **midpoint of the widest matching discount range**.

## Model

The classifier is implemented in **PyTorch**.

### Architecture

```text
95 Input Features
      ↓
Linear(95 → 128)
BatchNorm + ReLU + Dropout
      ↓
Linear(128 → 64)
BatchNorm + ReLU + Dropout
      ↓
Linear(64 → 32)
ReLU
      ↓
Linear(32 → 3)
      ↓
Loss / Medium Profit / High Profit
```

The model uses:

* Cross-Entropy Loss
* Adam optimizer
* Batch Normalization
* Dropout
* Early Stopping
* Best-model checkpointing

Model performance is evaluated using accuracy, precision, recall, F1-score, macro averages, and a confusion matrix.

## Features

The model uses order information including:

* Ship Mode
* Segment
* State
* Region
* Category
* Sub-Category
* Sales
* Quantity
* Discount
* Order Month

Categorical features are one-hot encoded, resulting in **95 model input features**.

## Streamlit App

The interactive application allows the user to enter the characteristics of an order and select a desired profit level.

The application then displays:

* Predicted discount ranges
* Predicted profit class for each range
* Recommended discount



## Project Structure

```text
Superstore/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── ...
│
├── notebooks/
│   ├── experiments.ipynb
│   ├── best_model.pth
│   ├── categories.json
│   └── feature_columns.json
│
├── sql/
│   └── ...
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/djenadi1/FULL-PROJECTS.git
cd FULL-PROJECTS/Superstore
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will then open in your browser.

## Technologies

* Python
* SQL
* Pandas
* NumPy
* PyTorch
* Scikit-learn
* Matplotlib
* Streamlit
* Jupyter Notebook
* Git

## Limitations

The recommended discounts are based on patterns learned from historical Superstore data.

The application is intended as a **machine learning and portfolio project**, not as a production pricing system. Real-world pricing decisions would require additional factors such as product costs, inventory, demand, competition, customer behavior, and business constraints.

## Future Improvements

* Compare the neural network with tree-based models such as XGBoost
* Add prediction confidence visualization
* Add model explainability
* Improve hyperparameter optimization
* Introduce additional business constraints into discount recommendations
* Train on larger and more recent retail datasets

## Author

**Mohammed Djenadi**

BSc Artificial Intelligence
Johannes Kepler University Linz

GitHub: `djenadi1`
