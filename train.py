import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load data
df = pd.read_csv("music_mood_data.csv")

# Encode key and mood
le_key = LabelEncoder()
le_mood = LabelEncoder()

df['key'] = le_key.fit_transform(df['key'])
df['mood'] = le_mood.fit_transform(df['mood'])

X = df.drop('mood', axis=1)
y = df['mood']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# Save model, scaler, encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_mood, open("label_encoder.pkl", "wb"))
pickle.dump(le_key, open("key_encoder.pkl", "wb"))
