import unittest
import json
import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, EXPECTED_FEATURES

class DiaPredictBackendTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertTrue(data.get('metrics_loaded'))

    def test_metrics_endpoint(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('accuracy', data)
        self.assertIn('roc_auc', data)
        self.assertGreater(data['accuracy'], 0)

    def test_predict_endpoint_valid(self):
        payload = {
            "Pregnancies": 2,
            "Glucose": 130,
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 85,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 35
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], [0, 1])
        self.assertIn('probability', data)
        self.assertIn('probability_percentage', data)
        self.assertIn('disclaimer', data)

    def test_predict_endpoint_missing_fields(self):
        payload = {
            "Pregnancies": 2,
            "Glucose": 130
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('missing_fields', data)

    def test_predict_endpoint_invalid_values(self):
        payload = {
            "Pregnancies": -1,
            "Glucose": "invalid_number",
            "BloodPressure": 70,
            "SkinThickness": 25,
            "Insulin": 85,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 35
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('details', data)

if __name__ == '__main__':
    unittest.main()
