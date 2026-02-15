import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Wine Quality Prediction App")

# Load dataset
data = pd.read_csv("dataset.csv", delimiter=";")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest (you can swap with other models)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Input Wine Features")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)
st.subheader("Prediction")
st.write(f"Predicted Wine Quality: {prediction[0]}")