import re
import pandas as pd


# Keywords commonly found in phishing emails
URGENCY_KEYWORDS = [
    'urgent', 'immediate', 'action required', 'verify now', 'verify your',
    'account suspended', 'account locked', 'click here', 'confirm your',
    'limited time', 'act now', 'response required', 'update your',
    'unusual activity', 'suspicious activity', 'unauthorized access',
    'your account', 'password expired', 'billing information', 'payment failed'
]

# Brand names commonly spoofed in phishing emails
SPOOFED_BRANDS = [
    'paypal', 'amazon', 'google', 'apple', 'microsoft',
    'netflix', 'bank', 'chase', 'wellsfargo', 'ebay'
]


def count_urls(text):
    """Count the number of URLs in the raw email text."""
    return len(re.findall(r'http\S+|www\.\S+', text))


def has_urgency(text):
    """Return 1 if any urgency keyword is found, else 0."""
    text_lower = text.lower()
    return int(any(keyword in text_lower for keyword in URGENCY_KEYWORDS))


def count_urgency_hits(text):
    """Count how many urgency keywords appear in the text."""
    text_lower = text.lower()
    return sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)


def count_exclamations(text):
    """Count exclamation marks — common in phishing emails."""
    return text.count('!')


def has_spoofed_domain(text):
    """
    Detect spoofed brand domains.
    e.g. paypal.malicious.net or amazon-support.com
    """
    text_lower = text.lower()
    for brand in SPOOFED_BRANDS:
        # Match brand name followed by something other than .com/.org/.net at root
        matches = re.findall(rf'{brand}[\w\-]*\.(?!com|org|net|co)[\w]{{2,}}', text_lower)
        if matches:
            return 1
    return 0


def count_email_addresses(text):
    """Count how many email addresses appear in the raw text."""
    return len(re.findall(r'\S+@\S+', text))


def text_length(text):
    """Return total character length of the email."""
    return len(text)


def word_count(text):
    """Return total word count of the email."""
    return len(text.split())


def extract_features(df):
    """
    Add all hand-crafted security features to the DataFrame.
    Expects a 'text' column with raw email text.
    Returns the DataFrame with new feature columns appended.
    """
    print("Extracting security features...")

    df['url_count']         = df['text'].apply(count_urls)
    df['has_urgency']       = df['text'].apply(has_urgency)
    df['urgency_hits']      = df['text'].apply(count_urgency_hits)
    df['exclamations']      = df['text'].apply(count_exclamations)
    df['spoofed_domain']    = df['text'].apply(has_spoofed_domain)
    df['email_addr_count']  = df['text'].apply(count_email_addresses)
    df['text_length']       = df['text'].apply(text_length)
    df['word_count']        = df['text'].apply(word_count)

    print("Feature extraction complete.")
    print(f"Features added: url_count, has_urgency, urgency_hits, exclamations, spoofed_domain, email_addr_count, text_length, word_count\n")

    return df


# Quick test when run directly
if __name__ == '__main__':
    from preprocess import preprocess

    df = preprocess('../data/Phishing_Email.csv')
    df = extract_features(df)

    print(df[['has_urgency', 'url_count', 'exclamations', 'spoofed_domain', 'label']].head(10))
    print("\nFeature averages by label:")
    print(df.groupby('label')[['url_count', 'has_urgency', 'urgency_hits', 'exclamations']].mean())