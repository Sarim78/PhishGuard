import sys
import os
import unittest

# Add src to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_model, predict


class TestPhishGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')
        cls.model = load_model(model_path)

    # -------------------------
    # Phishing tests
    # -------------------------

    def test_obvious_phishing(self):
        """Classic phishing email should be flagged."""
        email = (
            "URGENT: Your PayPal account has been suspended! "
            "Click here immediately to verify your account. "
            "Visit http://paypal.secure-verify.net now."
        )
        result = predict(self.model, email)
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['label'], 'Phishing Email')

    def test_phishing_confidence_high(self):
        """Phishing email should have high phishing confidence score."""
        email = (
            "Action required: Your account will be closed. "
            "Confirm your billing information immediately to avoid suspension."
        )
        result = predict(self.model, email)
        self.assertGreater(result['phishing_confidence'], 50)

    def test_spoofed_amazon(self):
        """Spoofed Amazon email should be detected as phishing."""
        email = (
            "Dear Customer, your Amazon account shows unusual activity. "
            "Please confirm your billing information to restore access. "
            "Failure to respond will result in account suspension."
        )
        result = predict(self.model, email)
        self.assertEqual(result['prediction'], 1)

    # -------------------------
    # Safe email tests
    # -------------------------

    def test_safe_email(self):
        """Normal workplace email should be flagged as safe."""
        email = (
            "Hi team, just a reminder that our monthly sync is on Thursday at 2pm. "
            "Please review the agenda beforehand. Thanks!"
        )
        result = predict(self.model, email)
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['label'], 'Safe Email')

    def test_safe_confidence_high(self):
        """Safe email should have high safe confidence score."""
        email = (
            "Hey, attached is the project report for Q3. "
            "Let me know if you have any feedback before the Friday deadline."
        )
        result = predict(self.model, email)
        self.assertGreater(result['safe_confidence'], 50)

    # -------------------------
    # Output structure tests
    # -------------------------

    def test_result_has_required_keys(self):
        """Prediction result should contain all expected keys."""
        email = "Test email content."
        result = predict(self.model, email)
        self.assertIn('label', result)
        self.assertIn('prediction', result)
        self.assertIn('phishing_confidence', result)
        self.assertIn('safe_confidence', result)
        self.assertIn('risk_level', result)

    def test_confidence_sums_to_100(self):
        """Phishing and safe confidence scores should sum to 100."""
        email = "Please verify your account immediately."
        result = predict(self.model, email)
        total = result['phishing_confidence'] + result['safe_confidence']
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_risk_level_values(self):
        """Risk level should always be LOW, MEDIUM, or HIGH."""
        emails = [
            "Hi, hope you are well.",
            "Please verify your account now.",
            "URGENT: Your account is suspended click here immediately!"
        ]
        valid_levels = {'LOW', 'MEDIUM', 'HIGH'}
        for email in emails:
            result = predict(self.model, email)
            self.assertIn(result['risk_level'], valid_levels)

    def test_empty_string(self):
        """Empty email input should not crash the predictor."""
        try:
            result = predict(self.model, "")
            self.assertIn(result['label'], ['Safe Email', 'Phishing Email'])
        except Exception as e:
            self.fail(f"predict() raised an exception on empty input: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)