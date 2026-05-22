import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bionic_core import BionicClassifier


def test_bionic_initialization():
    classifier = BionicClassifier()
    assert classifier.sample_rate == 22050
    assert classifier.t_analysis == 2500


def test_smoothing():
    classifier = BionicClassifier(smoothing_window=3)
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = classifier._smooth_signal(sig)
    assert len(smoothed) == len(sig)


def test_extract_features_no_peaks():
    classifier = BionicClassifier()
    sig = np.zeros(10000)
    features, amplitudes = classifier.extract_features(sig)
    assert len(features) == 0
    assert len(amplitudes) == 0


def test_density_centers_empty():
    classifier = BionicClassifier()
    centers = classifier.find_density_centers(np.array([]))
    assert len(centers) == 0


def test_analyze_file_not_found():
    classifier = BionicClassifier()
    with pytest.raises(FileNotFoundError):
        classifier.analyze_file("non_existent.wav")
