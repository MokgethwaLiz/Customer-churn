### Project Overview
This project is an end-to-end machine learning system developed to predict customer churn. The system combines data preprocessing, feature engineering, machine learning model training, hyperparameter optimization, experiment tracking, API deployment, and dashboard visualization.

The main objective of the project is to help businesses identify customers who are likely to churn so that retention strategies can be implemented more effectively.

### Features
Customer churn prediction using an ensembled machine learning model
Data preprocessing and feature engineering pipeline
Hyperparameter tuning using Optuna
Experiment tracking with MLflow
REST API built with FastAPI
Docker support for containerization
GitHub Actions CI/CD integration
Model evaluation using classification metrics

### Technologies Used

## Programming Language
Python

## Machine Learning Libraries
Scikit-learn
XGBoost Classifier
Random Forest Classifier
Logisttic Regression
Ensemble / Stacking Classifier
Optuna

## Data Analysis & Visualization
Pandas
NumPy
Matplotlib
Seaborn
Plotly

## Deployment & APIs
FastAPI
Uvicorn
Streamlit
Docker

## Experiment Tracking
MLflow

## Feature Selection
RFECV (Recursive Feature Elimination with Cross Validation)

## Version Control & CI/CD
GitHub
GitHub Actions

### Project Structure
customer_churn/
│
├── artifacts/                 # Saved trained models and artifacts
├── data/                      # Dataset files
├── mlartifacts/               # MLflow artifacts
├── notebooks/                 # Jupyter notebooks
├── src/
│   ├── inference/             # Prediction pipeline
│   ├── preprocess/            # Data preprocessing scripts
│   ├── train/                 # Model training scripts
│   └── app/
│       └── main.py            # FastAPI application
│
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD workflow
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── mlflow.db
└── README.md

### Dataset

The project uses the Telco Customer Churn dataset obtained from Kaggle. The dataset contains information about customers of a fictional telecommunications company providing home phone and internet services.

The main objective of the dataset is to predict whether a customer is likely to leave the company (churn), allowing the business to take proactive retention measures before customers discontinue their services.

Customer churn is a major challenge in the telecommunications industry because losing customers directly affects revenue and long-term profitability. Even a small reduction in churn rates can lead to significant business value and customer retention improvements.

## The dataset contains:

7,043 customer records
21 columns (features)
17 categorical variables
4 numerical variables

Each row represents a customer, while each column represents customer-related information such as demographics, account details, service subscriptions, billing information, and churn status.

## Independent Variables (X)
The independent variables include customer attributes such as:
Gender
SeniorCitizen
Partner
Dependents
Tenure
PhoneService
InternetService
Contract
PaymentMethod
MonthlyCharges
TotalCharges

## Dependent Variable (y)
The target variable is:
Churn

This variable indicates whether a customer left the company or stayed with the service provider.

### Machine Learning Workflow

Data Cleaning
Feature Engineering
Encoding Categorical Features
Feature Selection
Model Training
Hyperparameter Tuning
Model Evaluation
Ensemble Learning
Experiment Tracking with MLflow
Model Deployment with FastAPI
Containerization with Docker

### Data Preprocessing

The preprocessing pipeline includes:

Handling missing values
Cleaning categorical values
Encoding binary categorical variables
Encoding Yes/No categorical variables
Encoding multi-class categorical variables
Feature scaling
Feature selection
Train-test splitting

### Models Used

Logistic Regression
Random Forest
XGBoost
Ensemble / Stacking Model

### Evaluation Metrics

The models were evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix

Special emphasis was placed on optimizing Recall to improve the model’s ability to correctly identify customers likely to churn, reducing false negatives and improving customer retention targeting.

### MLflow Experiment Tracking

MLflow was used to:
Track experiments
Log parameters
Store trained models
Monitor metrics

### Run MLflow 

Bash

mlflow ui 
or 
mlflow server --host 127.0.0.1 --port 5000

### FastAPI Deployment 

Bash

uvicorn src.app.main:app --reload

Example API response

{
  "prediction": 1,
  "probability": 0.5733
}

### Docker Support

Build Docker image 

docker build -t customer-churn .

Run Docker Container

docker run -p 8000:8000 customer-churn

### GitHub Actions CI/CD

The project includes a GitHub Actions workflow that:

Builds the Docker image automatically
Pushes the Docker image to Docker Hub
Automates deployment workflows on pushes to the main branch

Clone repository

git clone <https://github.com/MokgethwaLiz/Customer-churn>
cd customer_churn

### Create Virtual Environment 

Windows 

Bash 

python -m venv venv
venv\Scripts\activate

Linux/macOS 

Bash 

python3 -m venv venv
source venv/bin/activate

Install Dependencies

Bash

pip install -r requirements.txt

### Workflow

Load and clean the dataset
Perform exploratory data analysis
Preprocess and encode features
Split data into training and testing sets
Train machine learning models
Optimize models using Optuna
Evaluate model performance
Track experiments using MLflow
Deploy models using FastAPI
Containerize the application using Docker
Automate builds using GitHub Actions

### Author

Developed by Mokgethwa Molongoana


