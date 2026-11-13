"""Check authorization ordering without needing real biometric artifacts."""
import unittest
import numpy as np
from main import AccessDenied, authenticate


class Model:
    def __init__(self, classes, probabilities):
        self.classes_ = classes
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, features):
        self.calls += 1
        return np.array([self.probabilities])


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.registry = {'maurice': '1', 'larissa': '2'}

    def verify(self, face, voice):
        return authenticate(face, voice, [[0]], [[0]], self.registry, .8, .8)

    def test_matching_enrolled_identities_pass(self):
        self.assertEqual(self.verify(Model(['maurice'], [1.]),
                                     Model(['maurice'], [1.])), 'maurice')

    def test_different_enrolled_voice_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.verify(Model(['maurice'], [1.]), Model(['larissa'], [1.]))

    def test_low_confidence_face_stops_before_voice(self):
        voice = Model(['maurice'], [1.])
        with self.assertRaises(AccessDenied):
            self.verify(Model(['maurice', 'larissa'], [.6, .4]), voice)
        self.assertEqual(voice.calls, 0)

    def test_unregistered_face_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.verify(Model(['unknown'], [1.]), Model(['unknown'], [1.]))

    def test_unregistered_voice_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.verify(Model(['maurice'], [1.]), Model(['unknown'], [1.]))

    def test_low_confidence_voice_is_denied(self):
        with self.assertRaises(AccessDenied):
            self.verify(Model(['maurice'], [1.]),
                        Model(['maurice', 'larissa'], [.6, .4]))

    def test_nonfinite_scores_are_denied(self):
        with self.assertRaises(AccessDenied):
            self.verify(Model(['maurice'], [float('nan')]), Model(['maurice'], [1.]))
