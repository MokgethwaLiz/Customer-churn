import optuna
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold,cross_val_score
from preprocess import preprocessed_data
from xgboost.sklearn import XGBClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (confusion_matrix,classification_report, recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score )

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Customer_Churn_Stacking_Model")

THRESHOLD = 0.25 #lower than 0.5 to boost recall

# load transformed data
X, y = preprocessed_data()

# [4] train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('--------------------------------------------------------')
print('[4] Dataset split 70/30')
print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')
print(f'Train shape: {y_train.shape}')
print(f'Test shape: {y_test.shape}')

      
# [5] RFECV using Random Forest
# Create a Random Forest classifier
estimator = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)

# Use RFE with cross-validation to 
# find the optimal number of features
selector = RFECV(estimator, cv=5, scoring= 'roc_auc', n_jobs=-1)
selector = selector.fit(X_train, y_train)

# selected feature names 
selected_features = X_train.columns[selector.support_]

# dropped feature names 
dropped_features = X_train.columns[~selector.support_]

# transform train and test data
X_train = X_train[selected_features]
X_test = X_test[selected_features]

print('--------------------------------------------------------')
print('[5] Feature selection process completed')
# Print the optimal number of features
print("Optimal number of features: %d" % selector.n_features_)

# Save selected features
FEATURE_PATH = (Path.cwd().parent/ "artifacts"/ "selected_features.pkl")
joblib.dump(selected_features.tolist(),FEATURE_PATH)
print('Selected features saved')

# features selected 
print(f'Selected features: {selected_features}')

# dropped features
print(f'Dropped features: {dropped_features}')

# feature rankings
print(f'Selector ranking: {selector.ranking_}')


#[6] Optuna objective function

def objective(trial):

    xgb_params = {

        "objective": "binary:logistic",
        "verbosity": 0,

        "n_estimators": trial.suggest_int("n_estimators",300,800),
        "max_depth": trial.suggest_int("max_depth",3,10),
        "learning_rate": trial.suggest_float("learning_rate",0.001,0.3,log=True),
        "subsample": trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),
        "gamma": trial.suggest_float("gamma",0,5),
        "min_child_weight": trial.suggest_int("min_child_weight",1,10),
        "reg_alpha": trial.suggest_float("reg_alpha",1e-5,10,log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",1e-5,10,log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight",1,10),
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1
    }

    # Create model

    xgb_model = XGBClassifier(**xgb_params)

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Evaluate
    scores = cross_val_score(
        xgb_model,
        X_train,
        y_train,
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )

    return scores.mean()


# create study
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42)
)

with mlflow.start_run():

    # run optuna
    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )

    print('[6]Optuna ran successfully')

    #best results
    print("Best Recall Score:")
    print(study.best_value)
    print("Best Parameters:")

    mlflow.log_params(study.best_params)

    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring_metric", "recall")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")


    # [7] train ensembled model

    print('Model training loading...')

    # base models 
    base_models = [
        (
            'random_forest',
            RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=1)
        ),

        (
            'xgboost',
            XGBClassifier(**study.best_params, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1)
        )
    ]

    # metadata
    meta_model = LogisticRegression(max_iter=1000)

    # Train stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5, n_jobs=-1
    )

    stacking_clf.fit(
        X_train,
        y_train
    )

    print('[7] Ensembled model trained')

    #save model
    mlflow.sklearn.log_model(
    stacking_clf,
    artifact_path="stacking_model"
    )

    MODEL_PATH = Path.cwd().parent / "artifacts" / "stacking_churn_model.pkl"

    joblib.dump(
        stacking_clf,
        MODEL_PATH
    )

    mlflow.log_artifact(
        MODEL_PATH
    )

    # [8] Model Predictions
    proba = stacking_clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    #calculate metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)

    #log metrics
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    print(
        f'[8] Confusion metrics: \n '
        f'{confusion_matrix(y_test, y_pred)}'
    )

    print(
        f'[9] Classification report: \n '
        f'{classification_report(y_test, y_pred)}'
    )
