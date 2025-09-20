# Titanic ML Model Deployment on AWS EC2

## Project Overview
This project demonstrates how to train a Machine Learning model, save it, and deploy it as a REST API using **Flask** on an **AWS EC2 instance**.  
It’s part of my **100 Days of Machine Learning + AWS + DevOps Challenge**.

---

## Tech Stack
- Python 3
- Pandas, NumPy, Scikit-learn
- Joblib (for model persistence)
- Flask (for API)
- AWS EC2 (for deployment)

---

## Steps

### 1. Train & Save Model
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("titanic.csv")

# Basic preprocessing
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare']].fillna(0)
X['Sex'] = X['Sex'].map({'male':0, 'female':1})
y = data['Survived']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.pkl")
