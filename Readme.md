# 🎵 Music Mood Classifier (K-Nearest Neighbors)

A **machine learning project** that classifies songs into **mood categories** based on their audio features using the **K-Nearest Neighbors (KNN)** algorithm.  
This project demonstrates the application of ML in **music information retrieval** and **audio analysis**.

---

## 📌 Project Overview
This project addresses the challenge of **automatically classifying music** into mood categories based on quantitative audio features.  
Using **KNN**, the model learns patterns from features like *danceability, energy, acousticness,* and *valence* to predict whether a song is:
- 😊 Happy  
- 😔 Sad  
- ⚡ Energetic  
- 🌙 Calm  

---

## ✨ Features
- 🎧 **Audio Feature Analysis** – Extracts features like energy, tempo, valence, etc.  
- 🤖 **KNN Classifier** – Simple yet powerful for multi-class mood classification.  
- 🔄 **Data Preprocessing** – Normalization/Standardization applied.  
- 📊 **Evaluation Metrics** – Accuracy, Precision, Recall, F1-Score.  
- ⚙️ **Hyperparameter Tuning** – Optimizes `K` value for best accuracy.  
- 🎶 **Interactive Prediction** – Predict mood of a new song by inputting its features.  

---

## 🎼 Dataset
The dataset contains **audio features of songs** with mood labels.

**Features include:**
- `danceability`, `energy`, `acousticness`, `valence`, `tempo`, `instrumentalness`, `speechiness`, etc.  

**Target Variable:** `mood` → (Happy, Sad, Energetic, Calm)  

---

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Elakiya-bcs22/Music_Mood_Classifier-KNN-.git
cd Music_Mood_Classifier-KNN-

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If requirements.txt missing, install manually
pip install scikit-learn pandas numpy matplotlib seaborn
```

👉 Optional: For Spotify API
```bash
pip install spotipy
```

---

## 🚀 Usage

### 1️⃣ Train & Test the Model
Run the notebook/script (`music_mood_knn.ipynb`) to:
- Load and preprocess dataset  
- Train **KNN model**  
- Tune best `K` value  
- Evaluate performance  

### 2️⃣ Predict Mood of a New Song
```python
# Example new song features
new_song_features = [[0.85, 0.9, -4.5, 0.95, 0.1, 0.0, 120]]

# Scale features
new_song_scaled = scaler.transform(new_song_features)

# Predict mood
prediction = knn_model.predict(new_song_scaled)
mood_mapping = {0: 'Calm', 1: 'Energetic', 2: 'Happy', 3: 'Sad'}
print(f"Predicted Mood: {mood_mapping[prediction[0]]}")
```

---

## 📊 Example Results
- Best Accuracy: **~85%**  
- Confusion Matrix & Classification Report included.  

---

## 👩‍💻 Author
**Elakiya Kalimuthu**  
📌 [GitHub Repository](https://github.com/Elakiya-bcs22/Music_Mood_Classifier-KNN-)  

---
