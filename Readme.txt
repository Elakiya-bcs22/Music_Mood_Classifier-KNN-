# Employee Promotion Prediction (Logistic Regression)

## 📌 Project Overview
This project predicts whether an employee will get a promotion or not using **Logistic Regression**.  
It is an end-to-end **Machine Learning + Flask Web App** project.

---

## 📂 Project Structure
```
employee_promotion_lr/
│── employee_promotion_data.csv   # Dataset
│── train_model.py                # Trains the model and saves model.pkl & scaler.pkl
│── app.py                        # Flask app for prediction
│── model.pkl                     # Trained Logistic Regression model
│── scaler.pkl                    # StandardScaler for normalization
│── requirements.txt              # Dependencies
│── templates/
│    └── index.html               # Frontend (web form)
```

---

## ⚙️ How to Run the Project

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the Model
```
python train_model.py
```
This will generate `model.pkl` and `scaler.pkl`.

### 3. Run the Flask App
```
python app.py
```
Open your browser at **http://127.0.0.1:5000/**

---

## 🎯 Features
- Logistic Regression ML model
- Predicts employee promotion based on experience, training score, performance rating, and previous promotions
- Web interface with company-themed design

---

## 📊 Input Fields
- **Experience (Years)**
- **Training Score (0-100)**
- **Performance Rating (1-5)**
- **Previous Promotions**

---

## ✅ Output
- "Employee will be Promoted ✅"
- "Employee will NOT be Promoted ❌"

---

## 🛠 Tech Stack
- Python
- Flask
- Pandas, NumPy, Scikit-learn
- HTML, CSS (for frontend)
