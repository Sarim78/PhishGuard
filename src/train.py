import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import preprocess
from features import extract_features


def train(data_path, model_output_path='../models/classifier.pkl'):
    """
    Full training pipeline for PhishGuard.
    Steps:
      1. Load and preprocess data
      2. Extract security features
      3. Train TF-IDF + Logistic Regression model
      4. Evaluate on holdout test set
      5. Save model to disk
    """

    # Step 1: Load and preprocess
    df = preprocess(data_path)

    # Step 2: Extract features
    df = extract_features(df)

    # Step 3: Split data
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}\n")

    # Step 4: Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),     # unigrams + bigrams
            sublinear_tf=True,      # apply log normalization to term frequency
            min_df=2                # ignore terms that appear in fewer than 2 emails
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            class_weight='balanced' # handles slight class imbalance
        ))
    ])

    # Step 5: Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # Step 6: Evaluate
    y_pred = pipeline.predict(X_test)

    print("=" * 45)
    print("         PHISHGUARD MODEL RESULTS")
    print("=" * 45)
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives  (Safe correctly flagged)    : {cm[0][0]}")
    print(f"  False Positives (Safe wrongly flagged)      : {cm[0][1]}")
    print(f"  False Negatives (Phishing missed)           : {cm[1][0]}")
    print(f"  True Positives  (Phishing correctly caught) : {cm[1][1]}")
    print("=" * 45)

    # Step 7: Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {model_output_path}")


if __name__ == '__main__':
    train('../data/Phishing_Email.csv')