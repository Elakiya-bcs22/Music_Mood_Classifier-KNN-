# 🎵 Music Mood Classifier (K-Nearest Neighbors)

A machine learning project that classifies songs into mood categories based on their audio features using the K-Nearest Neighbors (KNN) algorithm. This project demonstrates the application of ML in music information retrieval and audio analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-green) ![Status](https://img.shields.io/badge/Status-Complete-success)

## 📖 Project Overview

This project addresses the interesting challenge of automatically classifying music into mood categories based on quantitative audio features. Using the **K-Nearest Neighbors (KNN)** algorithm, the model learns patterns from features like danceability, energy, acousticness, and valence to predict whether a song is **Happy**, **Sad**, **Energetic**, or **Calm**.

## ✨ Features

*   **Audio Feature Analysis:** Utilizes standard audio features from Spotify's API or similar datasets (e.g., acousticness, danceability, energy, instrumentalness, valence, tempo).
*   **KNN Algorithm:** Implements the K-Nearest Neighbors classifier, a simple yet powerful instance-based learning algorithm perfect for multi-class classification.
*   **Data Preprocessing:** Includes feature scaling using Standardization/Normalization, which is crucial for distance-based algorithms like KNN.
*   **Model Evaluation:** Comprehensive evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and a detailed Classification Report.
*   **Hyperparameter Tuning:** Finds the optimal value for `K` (number of neighbors) to maximize model performance.
*   **Interactive Prediction:** Allows for predicting the mood of a new song based on its feature values.

## 🗂️ Dataset

The model is trained on a dataset containing audio features of songs labeled with mood categories.

*   **Possible Sources:**
    *   **Spotify API:** Using the `spotipy` library to fetch audio features for a curated list of songs.
    *   **Public Datasets:** Such as the "Emotion in Music" dataset from Kaggle or similar collections.
*   **Key Features:**
    *   `acousticness`, `danceability`, `energy`, `instrumentalness`
    *   `liveness`, `loudness`, `speechiness`, `valence`
    *   `tempo`, `key`, `mode`, `duration_ms`
*   **Target Variable:** `mood` or `label` (e.g., 'happy', 'sad', 'energetic', 'calm').

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Elakiya-bcs22/Music_Mood_Classifier-KNN-.git
    cd Music_Mood_Classifier-KNN-
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is not present, install the core libraries manually:*
    ```bash
    pip install scikit-learn pandas numpy matplotlib seaborn
    ```
    *Optional: If using the Spotify API, install `spotipy`:*
    ```bash
    pip install spotipy
    ```

## 🚀 Usage

### 1. Running the Main Script
The primary workflow is contained in the Jupyter Notebook or Python script (e.g., `music_mood_knn.ipynb`).

Execute the code to:
*   Load and explore the dataset.
*   Preprocess the data (handle missing values, scale features).
*   Train the KNN classifier.
*   Find the best `K` value using a error rate plot or grid search.
*   Evaluate the final model on a test set.

### 2. Making a Prediction on a New Song
After training, you can use the model to predict the mood of a new song by providing its audio features.

```python
# Example: Predict mood for a new set of features
# [danceability, energy, loudness, valence, acousticness, instrumentalness, tempo]
new_song_features = [[0.85, 0.9, -4.5, 0.95, 0.1, 0.0, 120]]  # Example values

# Scale the new features using the same scaler from training
new_song_scaled = scaler.transform(new_song_features)

# Predict the mood
prediction = knn_model.predict(new_song_scaled)
mood_mapping = {0: 'Calm', 1: 'Energetic', 2: 'Happy', 3: 'Sad'}
print(f"Predicted Mood: {mood_mapping[prediction[0]]}")