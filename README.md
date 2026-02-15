# ML-ASS2
a.Problem Statement
To build and evaluate multiple machine learning models for predicting wine quality based on physicochemical properties, and deploy an interactive application using Streamlit.
b. Dataset Description
	• Dataset: Wine Quality Dataset (dataset.csv)
	• Samples: 1599 rows
	• Features (11): fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
	• Target: quality (integer score of wine quality)
c. Models Used 
The following models were implemented and evaluated with metrics:
📊 Comparison Table

| ML Model Name            | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|--------------------------|----------|------|-----------|--------|------|------|
| Logistic Regression      | 0.72     | 0.74 | 0.71      | 0.70   | 0.70 | 0.38 |
| Decision Tree            | 0.68     | 0.69 | 0.67      | 0.66   | 0.66 | 0.32 |
| kNN                      | 0.70     | 0.71 | 0.69      | 0.68   | 0.68 | 0.35 |
| Naive Bayes              | 0.65     | 0.66 | 0.64      | 0.63   | 0.63 | 0.28 |
| Random Forest (Ensemble) | 0.78     | 0.81 | 0.77      | 0.76   | 0.76 | 0.50 |
| XGBoost (Ensemble)       | 0.80     | 0.83 | 0.79      | 0.78   | 0.78 | 0.53 |

Fill in the metric values from your notebook outputs here.
d. Observations 

| ML Model Name            | Observation about model performance |
|--------------------------|--------------------------------------|
| Logistic Regression      | Performs moderately well, stable baseline, but limited in capturing complex feature interactions. |
| Decision Tree            | Easy to interpret, but prone to overfitting; accuracy lower compared to ensemble methods. |
| kNN                      | Sensitive to scaling and dataset size; moderate performance but less robust. |
| Naive Bayes              | Very fast and simple; weaker accuracy due to independence assumption among features. |
| Random Forest (Ensemble) | Strong performance, robust against overfitting, consistently high accuracy and AUC. |
| XGBoost (Ensemble)       | Best overall performance; handles feature interactions effectively, highest accuracy and MCC. |<img width="1460" height="1258" alt="image" src="https://github.com/user-attachments/assets/29486408-4919-49cb-9fe3-0354252d9d99" />


Wine Quality Prediction – ML Assignment 2
📌 Project Overview
This project applies Machine Learning classification models to predict wine quality based on physicochemical properties.
The workflow includes:
- Data preprocessing
- Training multiple classifiers
- Evaluating performance metrics
- Deploying an interactive app using Streamlit

📊 Dataset
- Source: dataset.csv (Wine Quality dataset)
- Features:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- Target: quality (integer score of wine quality)

⚙️ Models Implemented
- Logistic Regression
- Decision Tree Classifier
- K‑Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

📈 Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC‑AUC (multi‑class)
- Confusion Matrix
(Insert your actual metric values here for each model, e.g. Logistic Regression: Accuracy 0.72, Random Forest: Accuracy 0.78, XGBoost: Accuracy 0.80)

🚀 Deployment
The project includes a Streamlit app (app.py) for interactive predictions.
Run Locally
pip install -r requirements.txt
streamlit run app.py


Requirements
pandas
numpy
scikit-learn
streamlit
xgboost


Streamlit Cloud
The app is deployed on Streamlit Cloud.
👉 [Live Demo Link] (insert your Streamlit Cloud URL here after deployment)

📂 Repository Structure
ML-ASS2/
│── dataset.csv
│── app.py
│── requirements.txt
│── README.md
│── LogisticRegression.ipynb
│── DecisionTree.ipynb
│── KNN.ipynb
│── NaiveBayes.ipynb
│── RandomForest.ipynb
│── XGBoost.ipynb


✅ How to Use
- Clone the repo:
git clone https://github.com/Aparna-Rath/ML-ASS2.git
cd ML-ASS2
- Install dependencies:
pip install -r requirements.txt
- Run the app:
streamlit run app.py
