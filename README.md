# Project Setup

## Clone the Repository

Open Git Bash or Command Prompt and run:

```bash
git clone https://github.com/MokgethwaLiz/Customer-churn.git
```

Move into the project folder:

```bash
cd Customer-churn
```

---

## Ensure Python 3.10 is Installed

Check your Python version:

```bash
python --version
```

or

```bash
py --version
```

The project requires **Python 3.10**.

If Python 3.10 is not installed, download and install it from:

https://www.python.org/downloads/release/python-3100/

During installation:

* Check **“Add Python to PATH”**
* Check **“Install launcher for all users”**

After installation, restart your terminal.

---

## Create a Virtual Environment

Create the virtual environment using Python 3.10:

```bash
py -3.10 -m venv venv
```

---

## Activate the Virtual Environment

### Git Bash

```bash
source venv/Scripts/activate
```

### Command Prompt

```bash
venv\Scripts\activate
```

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, your terminal should display:

```bash
(venv)
```

---

## Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Install Project Requirements

```bash
pip install -r requirements.txt
```

---

## Start MLflow Before Training

Before training the model, ensure MLflow is running.

Start the MLflow UI:

```bash
mlflow ui
```

MLflow will run on:

```text
http://127.0.0.1:5000
```

Keep this terminal running.

Open a new terminal, activate the virtual environment again, and continue with model training.

# Model Training

Ensure that:

* The virtual environment is activated
* MLflow is running before training starts

Run the training script:

```bash
python src/train.py
```

If training is successful:

* The trained model will be saved
* MLflow experiments will be logged
* Artifacts such as selected features and preprocessing objects will be generated

---

# Run the FastAPI Application

Start the FastAPI server:

```bash
uvicorn src.app.main:app --reload
```

The API will run on:

```text 
http://127.0.0.1:8000
```

Swagger documentation:

```text 
http://127.0.0.1:8000/docs
```

Example API response

{
  "prediction": 1,
  "probability": 0.5733
}

---

# Docker Setup

## Build the Docker Image

Ensure Docker Desktop is installed and running.

Build the Docker image:

```bash
docker build -t customer-churn .
```

---

## Run the Docker Container

```bash
docker run -p 8000:8000 customer-churn
```

The application will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Stop the Docker Container

Press:

```text
CTRL + C
```

or stop the container from Docker Desktop.


# Project Overview
This project is an end-to-end machine learning system developed to predict customer churn. The system combines data preprocessing, feature engineering, machine learning model training, hyperparameter optimization, experiment tracking, API deployment, and dashboard visualization.

The main objective of the project is to help businesses identify customers who are likely to churn so that retention strategies can be implemented more effectively.

# Features
Customer churn prediction using an ensembled machine learning model
Data preprocessing and feature engineering pipeline
Hyperparameter tuning using Optuna
Experiment tracking with MLflow
REST API built with FastAPI
Docker support for containerization
GitHub Actions CI/CD integration
Model evaluation using classification metrics

# Technologies Used

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

# Project Structure
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

# Dataset

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

# Machine Learning Workflow

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

# Data Preprocessing

The preprocessing pipeline includes:

Handling missing values
Cleaning categorical values
Encoding binary categorical variables
Encoding Yes/No categorical variables
Encoding multi-class categorical variables
Feature scaling
Feature selection
Train-test splitting

# Models Used

Logistic Regression
Random Forest
XGBoost
Ensemble / Stacking Model

# Evaluation Metrics

The models were evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix

Special emphasis was placed on optimizing Recall to improve the model’s ability to correctly identify customers likely to churn, reducing false negatives and improving customer retention targeting.

# MLflow Experiment Tracking

MLflow was used to:
Track experiments
Log parameters
Store trained models
Monitor metrics

# GitHub Actions CI/CD

The project includes a GitHub Actions workflow that:

Builds the Docker image automatically
Pushes the Docker image to Docker Hub
Automates deployment workflows on pushes to the main branch

# Workflow

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

# Author

Developed by Mokgethwa Molongoana


