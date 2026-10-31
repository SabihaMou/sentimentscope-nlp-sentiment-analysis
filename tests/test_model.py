import unittest
from unittest.mock import patch
from model import classify

class TestModel(unittest.TestCase):
    def test_empty_input(self):
        with self.assertRaises(ValueError):
            classify('  ')

    def test_long_input(self):
        with self.assertRaises(ValueError):
            classify('x' * 4001)

    @patch('model.get_classifier')
    def test_predictions_sorted(self, mock_classifier):
        mock_classifier.return_value.return_value = [
            {'label':'negative','score':0.1},
            {'label':'positive','score':0.8},
            {'label':'neutral','score':0.1},
        ]
        result = classify('Great product')
        self.assertEqual(result['label'], 'positive')
        self.assertEqual(result['confidence'], 0.8)

if __name__ == '__main__':
    unittest.main()
