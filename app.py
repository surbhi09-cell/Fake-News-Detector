# app.py
from flask import Flask, request, render_template
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+','', text)
    text = re.sub(r'[^a-z\s]',' ', text)
    text = re.sub(r'\s+',' ', text).strip()
    tokens = [stemmer.stem(word) for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

app = Flask(__name__)

MODEL_PATH = 'models/model.joblib'
VECT_PATH = 'models/vectorizer.joblib'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError("Please run train_model.py first to create models/")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    prob = None
    if request.method == "POST":
        news = request.form.get("news", "")
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = max(model.predict_proba(vec)[0]) if hasattr(model, "predict_proba") else None
        result = "REAL" if pred==1 else "FAKE"
        prob = round(float(proba), 3) if proba is not None else None
    return render_template("index.html", result=result, prob=prob)
    
if __name__ == "__main__":
    app.run(debug=True)
