import pandas as pd
import re


def load_data(path):
    """
    Load the Phishing_Email.csv dataset and map labels to binary.
    - 'Safe Email'     -> 0
    - 'Phishing Email' -> 1
    """
    df = pd.read_csv(path)

    # Drop the unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Rename columns to standard names
    df = df.rename(columns={
        'Email Text': 'text',
        'Email Type': 'label_raw'
    })

    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    # Map labels to binary
    df['label'] = df['label_raw'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })

    # Drop any rows where label mapping failed
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    return df[['text', 'label']]


def clean_text(text):
    """
    Clean raw email text for NLP processing.
    Steps:
      1. Lowercase everything
      2. Replace URLs with URL token
      3. Replace email addresses with EMAIL token
      4. Remove special characters and digits
      5. Collapse extra whitespace
    """
    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)

    # Replace email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)

    # Remove special characters and digits, keep letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess(path):
    """
    Full preprocessing pipeline.
    Loads data, cleans text, returns a ready-to-use DataFrame.
    """
    print(f"Loading data from: {path}")
    df = load_data(path)

    print(f"Total rows loaded: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().rename({0: 'Safe', 1: 'Phishing'})}\n")

    print("Cleaning email text...")
    df['clean_text'] = df['text'].apply(clean_text)

    print("Preprocessing complete.\n")
    return df


# Quick test when run directly
if __name__ == '__main__':
    df = preprocess('../data/Phishing_Email.csv')
    print(df[['clean_text', 'label']].head(3))