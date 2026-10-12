# Steam Game Genre Predictor

This project is a machine learning system that predicts game genres and categories for Steam games using a neural network model. It combines traditional numerical features with text embeddings from game names to provide accurate genre predictions.

## Project Overview

The system takes game metadata as input and predicts the most likely genres and categories for a given game. This is achieved through:
- Training a neural network model on Steam game data
- Using sentence transformers to create embeddings from game names
- Combining name embeddings with numerical features for prediction
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
│   ├── scaler.npz                         # Data scaler for feature normalization
│   └── scaler.pkl                         # Alternative scaler format
├── processed/                             # Processed data files
│   ├── df_staging_2.csv                   # Cleaned CSV dataset
│   └── df_staging_2.parquet               # Cleaned Parquet dataset
├── requirements.txt                       # Project dependencies
├── README.md                              # This file
└── data_source.txt                        # Data source information
```

## Key Features

- **Neural Network Model**: Implements a multi-layer perceptron (MLP) for genre and category prediction
- **Text Embeddings**: Uses Sentence Transformers to convert game names into numerical vectors
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
- Input layer: Concatenation of name embeddings (384 dimensions) and numerical features
- Hidden layers: 3 fully connected layers with ReLU activation and dropout
- Output layer: Linear layer with sigmoid activation for multi-label classification

## Data Processing Pipeline

1. **Data Cleaning**: Removes unnecessary columns, handles missing values, standardizes text data
2. **Feature Engineering**: Creates embeddings using Sentence Transformers for game names
3. **Categorical Encoding**: Converts genre/category tags to binary features using MultiLabelBinarizer
4. **Numerical Processing**: Normalizes numerical features and handles estimated ownership ranges
5. **Model Training**: Trains neural networks on the processed dataset

## Usage

The system provides two prediction modes:
1. **Genres only**: Predicts individual game genres (32 categories)
2. **Genres + categories**: Predicts both genres and categories (46 categories total)

Input parameters include:
- Game name
- Release month
- Price information
- User ratings and reviews
- Platform support
- Playtime statistics
- Achievement count
- DLC count
- Estimated player ownership

## Output

Predictions are returned as top 10 probabilities for each category, with confidence scores between 0 and 1.

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