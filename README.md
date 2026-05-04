<div align="center">

# 🎣 PhishGuard

**NLP-powered phishing email classifier using TF-IDF + Logistic Regression**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikitlearn-1.3+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

</div>

---

## 📌 Overview

PhishGuard is a machine learning classifier that detects phishing emails using natural language processing. It combines TF-IDF vectorization with hand-crafted security features: urgency keywords, URL counts, and spoofed domain patterns to accurately flag malicious emails before they reach users.

Built as a cybersecurity portfolio project simulating a real SOC (Security Operations Center) detection tool.

---

## ✨ Features

- 🧹 **Text Preprocessing**: cleans raw email bodies, replaces URLs and sender addresses with tokens
- 🔍 **Feature Engineering**: extracts urgency signals, URL counts, exclamation marks, and spoofed domain patterns
- 🤖 **TF-IDF + Logistic Regression Pipeline**: bigram-aware vectorization with a fast, interpretable classifier
- 📦 **Model Persistence**: trained model saved as `.pkl` for reuse
- 🔮 **Predict on New Emails**: feed any raw email text and get an instant phishing/legit classification

---

## 📊 Model Performance

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~97%  |
| Precision | ~96%  |
| Recall    | ~95%  |
| F1 Score  | ~96%  |

> Evaluated on a 20% holdout split from the Phishing Email Detection dataset by subhajournal (Kaggle).

---

## 📁 Project Structure

```
phishguard/
├── data/
│   └── Phishing_Email.csv
├── src/
│   ├── preprocess.py       # Text cleaning and normalization
│   ├── features.py         # Hand-crafted security feature extraction
│   ├── train.py            # Model training and evaluation
│   └── predict.py          # Run predictions on new emails
├── models/
│   └── classifier.pkl      # Saved trained model
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sarim78/phishguard.git
cd phishguard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the dataset from Kaggle:
[Phishing Email Detection by subhajournal](https://www.kaggle.com/datasets/subhajournal/phishingemails)

Place `Phishing_Email.csv` inside the `data/` folder.

### 4. Train the model

```bash
cd src
python train.py
```

### 5. Predict on a new email

```bash
python predict.py
```

---

## 🧠 How It Works

### Step 1: Preprocessing (`preprocess.py`)
Raw email text is lowercased, URLs are replaced with `URL` tokens, email addresses with `EMAIL` tokens, and special characters are stripped.

### Step 2: Feature Extraction (`features.py`)
Hand-crafted SOC-relevant signals are extracted:
- Number of URLs in the email
- Presence of urgency keywords (e.g. `verify now`, `account suspended`, `click here`)
- Count of exclamation marks
- Detection of spoofed brand domains (e.g. `paypal.malicious.net`)

### Step 3: Model Training (`train.py`)
A `sklearn` Pipeline chains TF-IDF vectorization (5000 features, bigrams) with Logistic Regression. The model is evaluated with a classification report and saved to `models/classifier.pkl`.

### Step 4: Prediction (`predict.py`)
Load the saved model and pass any raw email string to get a `Phishing` or `Legit` label instantly.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data loading and manipulation |
| scikit-learn | TF-IDF, Logistic Regression, Pipeline |
| re (regex) | Text cleaning and pattern matching |
| pickle | Model serialization |

---

## 📦 Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🗺 Roadmap

- [ ] Add Naive Bayes model for comparison
- [ ] Build a Flask API to expose predictions via endpoint
- [ ] Add a simple web UI for pasting and scanning emails
- [ ] Integrate AbuseIPDB for IP reputation enrichment
- [ ] Add MITRE ATT&CK technique tagging to detections

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built for cybersecurity portfolio purposes, simulating real SOC detection workflows.</sub>
</div>
