# Step 1: Data Cleaning & Preprocessing

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# Load dataset
df = pd.read_csv(r'D:\Desktop\Internship at VRTECHSOL\creditcard.csv')

# Check structure
print("Shape of data:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['Class'].value_counts())

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split before applying SMOTE (to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Show resampled class distribution
print("\nClass distribution after SMOTE:\n", Counter(y_train_resampled))
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Check class balance
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0 = Non-Fraud, 1 = Fraud)')
plt.xlabel('Transaction Class')
plt.ylabel('Count')
plt.show()

# 2. Visualize transaction amounts by class
plt.figure(figsize=(10, 5))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=100, color='green', label='Non-Fraud', stat='density')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=100, color='red', label='Fraud', stat='density')
plt.title('Transaction Amounts: Fraud vs Non-Fraud')
plt.xlabel('Amount')
plt.legend()
plt.show()

# 3. Boxplots to compare key features (like V1 to V10)
features = ['V1', 'V2', 'V3', 'V4', 'V5']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='Class', y=col, data=df)
    plt.title(f'{col} by Class')
plt.tight_layout()
plt.show()

# 4. Correlation heatmap of top features
plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, cmap='coolwarm_r', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Time vs Fraud (optional)
if 'Time' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[df['Class'] == 1]['Time'], bins=100, color='red', label='Fraud')
    plt.title('When Do Frauds Occur? (Time Feature)')
    plt.xlabel('Time (seconds since first transaction)')
    plt.legend()
    plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# For evaluation in Step 4
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ------------------ LOGISTIC REGRESSION ------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_resampled, y_train_resampled)
log_pred = logreg.predict(X_test)
print("📘 Logistic Regression Results:")
print(classification_report(y_test, log_pred))
print("AUC Score:", roc_auc_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))


# ------------------ RANDOM FOREST ------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
rf_pred = rf.predict(X_test)
print("\n🌲 Random Forest Results:")
print(classification_report(y_test, rf_pred))
print("AUC Score:", roc_auc_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))


# ------------------ XGBOOST ------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_resampled, y_train_resampled)
xgb_pred = xgb.predict(X_test)
print("\n🔥 XGBoost Results:")
print(classification_report(y_test, xgb_pred))
print("AUC Score:", roc_auc_score(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Ground truth
y_true = y_test

# ------------------ 1. Confusion Matrices ------------------

models = {
    'Logistic Regression': log_pred,
    'Random Forest': rf_pred,
    'XGBoost': xgb_pred
}

for name, pred in models.items():
    cm = confusion_matrix(y_true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb_grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

xgb_grid_search.fit(X_train_resampled, y_train_resampled)

print("🔥 Best XGBoost Parameters:")
print(xgb_grid_search.best_params_)

# Evaluate tuned XGBoost model
xgb_best = xgb_grid_search.best_estimator_
xgb_tuned_pred = xgb_best.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define parameter grid
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42)

rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_params,
    n_iter=20,  # Number of random combinations
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train_resampled, y_train_resampled)

print("🌲 Best Random Forest Parameters:")
print(rf_random_search.best_params_)

# Evaluate tuned model
rf_best = rf_random_search.best_estimator_
rf_tuned_pred = rf_best.predict(X_test)
from sklearn.metrics import classification_report, roc_auc_score

print("\n🌲 Tuned Random Forest Performance:")
print(classification_report(y_test, rf_tuned_pred))
print("AUC:", roc_auc_score(y_test, rf_tuned_pred))

print("\n🔥 Tuned XGBoost Performance:")
print(classification_report(y_test, xgb_tuned_pred))
print("AUC:", roc_auc_score(y_test, xgb_tuned_pred))
