"""model_student.py
Model training logic for classification.
Replace 'student' with your full name on submission.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_knn_optimized(X_train, y_train):
    """Train KNN with hyperparameter tuning."""
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='f1', n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_xgb(X_train, y_train, random_state=42):
    """Train an XGBoost model."""
    model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model
