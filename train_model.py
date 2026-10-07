# train_model.py
import os
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure nltk resources
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\S+','', text)
    # remove non alphabets
    text = re.sub(r'[^a-z\s]',' ', text)
    # remove extra spaces
    text = re.sub(r'\s+',' ', text).strip()
    # remove stopwords + stemming
    tokens = [stemmer.stem(word) for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def load_data():
    # Try common filenames: Fake.csv and True.csv (Kaggle dataset)
    if os.path.exists('data/Fake.csv') and os.path.exists('data/True.csv'):
        fake = pd.read_csv('data/Fake.csv')
        true = pd.read_csv('data/True.csv')
        # Many Kaggle datasets have 'text' and 'title'. We'll combine title+text.
        fake['label'] = 0
        true['label'] = 1
        fake['content'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
        true['content'] = true['title'].fillna('') + ' ' + true['text'].fillna('')
        df = pd.concat([fake[['content','label']], true[['content','label']]], ignore_index=True)
        return df
    # Else try a single CSV
    elif os.path.exists('data/news.csv'):
        df = pd.read_csv('data/news.csv')  # expect columns 'text' and 'label' or 'content' and 'label'
        if 'text' in df.columns:
            df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('') if 'title' in df.columns else df['text']
        elif 'content' not in df.columns:
            raise Exception("news.csv missing 'text' or 'content' column")
        return df[['content','label']]
    else:
        raise FileNotFoundError("Put dataset in data/ folder: Fake.csv and True.csv (Kaggle).")

def main():
    print("Loading data...")
    df = load_data()
    print("Total samples:", len(df))
    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Cleaning text (may take time)...")
    df['content_clean'] = df['content'].astype(str).apply(clean_text)

    X = df['content_clean'].values
    y = df['label'].values

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    print("Model and vectorizer saved to models/")

if __name__ == '__main__':
    main()
