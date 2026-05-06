import pickle
import sys
from preprocess import clean_text


def load_model(model_path='../models/classifier.pkl'):
    """Load the trained PhishGuard model from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {model_path}\n")
        return model
    except FileNotFoundError:
        print(f"Model not found at: {model_path}")
        print("Run train.py first to generate the model.")
        sys.exit(1)


def predict(model, email_text):
    """
    Predict whether a single email is phishing or safe.
    Returns a dict with label, confidence, and risk level.
    """
    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]

    phishing_confidence = round(probabilities[1] * 100, 2)
    safe_confidence = round(probabilities[0] * 100, 2)

    if phishing_confidence >= 80:
        risk = "HIGH"
    elif phishing_confidence >= 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        'label': 'Phishing Email' if prediction == 1 else 'Safe Email',
        'prediction': prediction,
        'phishing_confidence': phishing_confidence,
        'safe_confidence': safe_confidence,
        'risk_level': risk
    }


def print_result(result):
    """Pretty print the prediction result."""
    print("=" * 45)
    print("         PHISHGUARD SCAN RESULT")
    print("=" * 45)
    print(f"  Verdict          : {result['label']}")
    print(f"  Risk Level       : {result['risk_level']}")
    print(f"  Phishing Score   : {result['phishing_confidence']}%")
    print(f"  Safe Score       : {result['safe_confidence']}%")
    print("=" * 45)


def run_examples(model):
    """Run a few sample emails to demonstrate the classifier."""

    samples = [
        {
            'description': 'Phishing attempt',
            'text': (
                "URGENT: Your PayPal account has been suspended! "
                "Click here immediately to verify your account and avoid permanent closure. "
                "Action required within 24 hours. Visit http://paypal.secure-verify.net now."
            )
        },
        {
            'description': 'Safe email',
            'text': (
                "Hi team, just a reminder that our monthly sync is scheduled for Thursday at 2pm. "
                "Please review the agenda doc beforehand and come prepared with updates. "
                "Let me know if you have any conflicts. Thanks!"
            )
        },
        {
            'description': 'Suspicious email',
            'text': (
                "Dear Customer, your Amazon account shows unusual activity. "
                "Please confirm your billing information to restore access. "
                "Failure to respond will result in account suspension."
            )
        }
    ]

    for sample in samples:
        print(f"\nTest: {sample['description']}")
        print(f"Email: \"{sample['text'][:80]}...\"")
        result = predict(model, sample['text'])
        print_result(result)


if __name__ == '__main__':
    model = load_model('../models/classifier.pkl')

    # If an email is passed as a command line argument
    if len(sys.argv) > 1:
        email_input = ' '.join(sys.argv[1:])
        print(f"Scanning: \"{email_input[:80]}...\"\n")
        result = predict(model, email_input)
        print_result(result)
    else:
        # Run built-in examples
        run_examples(model)