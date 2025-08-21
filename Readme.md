🎵 Music Mood Classifier (KNN)

📌 Project Overview
The Music Mood Classifier is a machine learning project that predicts the mood of a music track 
(e.g., Happy, Sad, Energetic, Calm) based on extracted audio features. 
It uses the K-Nearest Neighbors (KNN) algorithm for classification.

🧠 Problem Statement
With millions of songs available on streaming platforms, 
automatically categorizing music by mood can help users discover songs 
that match their emotions. Manual tagging is inconsistent and time-consuming, 
so a machine learning model can assist in automatic mood labeling.

🎯 Objective
- Data Preparation: Collect a dataset of music tracks with features and mood labels.
- Model Training: Implement and train a KNN classifier.
- Model Evaluation: Assess the accuracy and effectiveness of the classifier.
- Deployment: Provide a Flask web app for real-time mood prediction.

🔧 Technologies & Libraries
- Programming Language: Python
- Libraries:
  - scikit-learn – for KNN model
  - pandas – for dataset handling
  - numpy – for feature processing
  - flask – for web deployment

📁 Project Structure
Music_Mood_Classifier-KNN-/
├── data.csv                   # Dataset of music features & moods
├── train_model.py             # Script to train KNN model
├── app.py                     # Flask app for prediction
├── model.pkl                  # Trained KNN model
├── scaler.pkl                 # Feature scaler
├── requirements.txt           # Dependencies
└── templates/
    └── index.html             # Web form for input

🧪 Dataset
The dataset contains extracted features from music tracks such as:
- Tempo
- Beats per Minute (BPM)
- Spectral Features
- Key
- Loudness
- Danceability
Target: Mood (Happy, Sad, Energetic, Calm, etc.)

⚙️ Setup & Installation
1. Clone the repository:
   git clone https://github.com/Elakiya-bcs22/Music_Mood_Classifier-KNN-
   cd Music_Mood_Classifier-KNN-

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python train_model.py

4. Run the app:
   python app.py

🚀 Usage
- Open browser at http://127.0.0.1:5000/
- Enter music features
- Get predicted mood of the song 🎶

📊 Results
The KNN model achieved ~75–80% accuracy in predicting moods, 
demonstrating its usefulness for music classification tasks.

🛠️ Future Enhancements
- Use advanced algorithms (SVM, Random Forest, Deep Learning)
- Extract features directly from audio files
- Improve user interface with visualizations
- Integrate with Spotify/YouTube APIs for live mood classification

📄 License
This project is licensed under the MIT License.

📬 Contact
For any inquiries, please contact Elakiya BCS22 via GitHub.

🔗 References
- Scikit-learn Documentation
- Music Information Retrieval Studies
- Kaggle Datasets

🏆 Acknowledgements
Thanks to open-source contributors, dataset providers, and ML community 
for inspiration and resources.