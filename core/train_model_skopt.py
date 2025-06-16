# backend/core/train_model_skopt.py

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocess import load_data, build_pipeline
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

def train_and_save_model():
    # Load raw data and split
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define base model
    base_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=None,
)


    # Wrap preprocessing + model into a single pipeline
    pipeline = build_pipeline(base_model)

    # Define hyperparameter space (only for model part)
    search_space = {
        'model__max_depth': Integer(3, 10),
        'model__n_estimators': Integer(50, 300),
        'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'model__subsample': Real(0.6, 1.0),
        'model__colsample_bytree': Real(0.6, 1.0),
    }

    # BayesSearchCV on pipeline
    opt = BayesSearchCV(
        pipeline,
        search_spaces=search_space,
        n_iter=25,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit and evaluate
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("logs/roc_curve.png")
    plt.close()

    print(f"✅ Best Accuracy: {acc:.4f}")

    # Save pipeline model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/lead_model.pkl")
    print("✅ Pipeline model saved to models/lead_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
