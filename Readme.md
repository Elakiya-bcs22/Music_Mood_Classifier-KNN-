# 🎬 Movie Review Sentiment Classifier (Naive Bayes)

A machine learning project that classifies movie reviews as either **positive** or **negative** using a Naive Bayes algorithm. This project demonstrates a classic NLP text classification task from end-to-end, including data preprocessing, model training, evaluation, and prediction.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![NLTK](https://img.shields.io/badge/NLTK-Data%20Processing-green) ![Status](https://img.shields.io/badge/Status-Complete-success)

## 📖 Project Overview

This project tackles the fundamental Natural Language Processing (NLP) problem of sentiment analysis. The goal is to automatically predict the sentiment (positive or negative) expressed in a piece of text about a movie. The classifier is built using a **Multinomial Naive Bayes** model, a popular and effective algorithm for text classification due to its simplicity and efficiency with discrete data (like word counts).

## ✨ Features

*   **Text Preprocessing:** Cleans raw text data using techniques like lowercasing, tokenization, and stopword removal.
*   **Feature Extraction:** Transforms text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
*   **Naive Bayes Model:** Implements a probabilistic classifier that works well with high-dimensional text data.
*   **Model Evaluation:** Thoroughly evaluates performance using standard metrics like Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
*   **Interactive Prediction:** Allows users to input their own movie review text and get an instant sentiment prediction (Positive/Negative).

## 🗂️ Dataset

The model is trained on the **IMDb Movie Reviews dataset**, a benchmark dataset for sentiment analysis. This dataset contains 50,000 reviews, evenly split between 25,000 positive and 25,000 negative sentiments.

*   **Source:** The dataset is typically fetched internally via libraries like `nltk` or `tensorflow`.
*   **Size:** 50,000 reviews (25,000 positive, 25,000 negative).

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Elakiya-bcs22/Movie_Review_Classifier-NB-.git
    cd Movie_Review_Classifier-NB-
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
    pip install scikit-learn nltk pandas numpy matplotlib seaborn
    ```

## 🚀 Usage

### 1. Running the Main Script
The primary workflow is contained in `movie_review_naive_bayes.ipynb` (Jupyter Notebook) or the corresponding `.py` file.

Run the notebook cell-by-cell or execute the Python script to:
*   Load and preprocess the data.
*   Train the Naive Bayes classifier.
*   Evaluate the model on a test set.
*   Print out performance metrics and visualizations.

### 2. Making a Prediction on a New Review
After training, the script will likely save the model and vectorizer, or you can use it to predict directly. You can input a custom review like this:

```python
# Example code from the project
new_review = ["This movie was an absolute masterpiece! The acting was superb and the story was captivating."]
prediction = loaded_model.predict(new_review)
print("Positive" if prediction[0] == 1 else "Negative")