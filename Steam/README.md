# Steam Game Genre Predictor

This project is a machine learning system that predicts game genres and categories for Steam games using a neural network model. It combines numerical features with text embeddings of the game's name and short description.

## Live Demo

Try the live Steam Game Genre Predictor web application:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://full-projects-8iblvu4aktnyyaairiralk.streamlit.app/)


## Project Overview

The system takes game metadata as input and predicts the most likely genres and categories for a given game. This is achieved through:
- Training a neural network model on Steam game data
- Using sentence transformers to create embeddings from game names and short descriptions
- Combining those embeddings with numerical features for prediction
- Supporting both genre-only and genre+category predictions

## Dataset

The project uses the Steam games dataset containing information about thousands of games including:
- Game titles and metadata
- Release dates and platform support (Windows, Mac, Linux)
- Pricing information
- User ratings and reviews
- Genre and category tags
- Achievement data
- Metacritic scores
- Estimated player ownership ranges

## Project Structure

```
steam-game-predictor/
├── 00_data_understanding_cleaning.ipynb    # Data exploration and preprocessing
├── 01_model_train_eval.ipynb              # Model training and evaluation
├── app.py                                 # Streamlit web application
├── steam_games.csv                        # Main Steam games dataset
├── labels/                                # Label files for genre/category predictions
│   ├── feature_columns.csv                # Feature column names used in model
│   ├── labels_genres.csv                  # Genre label mapping
│   └── labels_gencat.csv                  # Genre+category label mapping
├── models/                                # Trained model files
│   ├── all-MiniLM-L6-v2                   # Sentence transformer model for embeddings
│   ├── best_genres.pth                    # Best performing genre prediction model
│   ├── best_gencat.pth                    # Best performing genre+category prediction model
│   └── scaler.pkl                         # StandardScaler fitted on the training set
├── processed/                             # Processed data files
│   └── df_staging_2.parquet               # Cleaned dataset + embeddings
├── requirements.txt                       # Project dependencies
├── README.md                              # This file
└── data_source.txt                        # Data source information
```

## Key Features

- **Neural Network Model**: Implements a multi-layer perceptron (MLP) for genre and category prediction
- **Text Embeddings**: Uses Sentence Transformers to convert game names and descriptions into numerical vectors
- **Feature Engineering**: Combines numerical features with text embeddings for comprehensive predictions
- **Web Application**: Streamlit interface for real-time predictions
- **Multiple Prediction Modes**: Supports both genres-only and genres+categories predictions

## Getting Started

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Web Application

1. Ensure all model files are in their respective directories
2. Run the Streamlit application:
```bash
streamlit run app.py
```
3. Access the application at `http://localhost:8501`

## Model Architecture

The neural network model uses a multi-layer perceptron with the following structure:
- Input layer: name embedding (384) + description embedding (384) + 32 tabular features (log-scaled counts, release year/month, platforms, price status)
- Hidden layers: 3 fully connected layers with ReLU activation and dropout
- Output layer: Linear layer with sigmoid activation for multi-label classification

## Data Processing Pipeline

1. **Data Cleaning**: Removes unnecessary columns, handles missing values, standardizes text data, merges genres/categories that appear in several languages (e.g. `solo` / `un jugador` / `シングルプレイヤー` → `single-player`)
2. **Feature Engineering**: Creates embeddings for game names and short descriptions, counts supported languages, extracts release year
3. **Categorical Encoding**: Converts genre/category tags to binary features using MultiLabelBinarizer
4. **Numerical Processing**: Log-scales heavy-tailed counts (reviews, owners, playtime...), then standardizes everything; estimated ownership ranges become their midpoint
5. **Model Training**: Drops games with no genre and labels with fewer than 100 training examples, then trains the MLPs with early stopping on validation mAP

## Usage

The system provides two prediction modes:
1. **Genres only**: Predicts game genres (24 labels)
2. **Genres + categories**: Predicts both genres and store categories (78 labels total)

Input parameters include:
- Game name
- Short description (optional, but it improves predictions a lot)
- Release year and month
- Price information
- User ratings and reviews
- Platform support
- Playtime statistics
- Achievement count
- DLC count
- Number of supported languages
- Estimated player ownership

## Output

Predictions are returned as top 10 probabilities for each category, with confidence scores between 0 and 1.

## Results

Test set (10% hold-out):

| Model | macro mAP | micro mAP | top-1 prediction is a real label |
|---|---|---|---|
| Genres only (before) | 0.294 | — | — |
| Genres only (now) | 0.496 | 0.767 | 83.9% |
| Genres + categories (before) | 0.180 | — | — |
| Genres + categories (now) | 0.386 | 0.806 | 98.6% |

Most of the improvement came from adding the short description embedding; cleaning the labels (merging translated
categories, dropping labels with too few examples) and log-scaling the counts accounted for the rest.

## Related Project: Superstore Discount Optimizer

This repository also contains another machine learning project:

**Superstore Discount Optimizer** - An end-to-end machine learning project that uses historical retail data to recommend discounts based on a desired profitability level.

Key features:
- PyTorch neural network for profit classification (Loss, Medium Profit, High Profit)
- SQL data cleaning and analysis
- Feature engineering with one-hot encoding (95 input features)
- Interactive Streamlit application for discount recommendation
- Discount range search algorithm

Project structure:
```
Superstore/
├── app.py
├── requirements.txt
├── data/
│   └── ...
├── notebooks/
│   ├── experiments.ipynb
│   ├── best_model.pth
│   ├── categories.json
│   └── feature_columns.json
├── sql/
│   └── ...
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Steam Database for providing the game data
- Hugging Face and Sentence Transformers for text embedding capabilities
- The open-source community for the libraries used in this project