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
    # Звичайний сигнал
    sig = np.array([1, 2, 3, 4, 5])
    # Очікуване: згортка з [1/3, 1/3, 1/3]
    smoothed = classifier._smooth_signal(sig)
    assert len(smoothed) == len(sig)

def test_extract_features_no_peaks():
    # Сигнал без піків
    classifier = BionicClassifier()
    sig = np.zeros(10000)
    features = classifier.extract_features(sig)
    assert len(features) == 0

def test_density_centers_empty():
    classifier = BionicClassifier()
    centers, _ = classifier.find_density_centers(np.array([]))
    assert len(centers) == 0

def test_calculate_distances_empty():
    classifier = BionicClassifier()
    dists = classifier.calculate_distances([])
    assert len(dists) == 0

def test_calculate_distances():
    classifier = BionicClassifier()
    # 3 точки, що утворюють трикутник (0,0), (0,3), (4,0)
    centers = [(0, 0), (0, 3), (4, 0)]
    dists = classifier.calculate_distances(centers)
    assert len(dists) == 3
    # Відстані мають бути 3, 4, 5
    assert np.isclose(dists[0], 3.0)
    assert np.isclose(dists[1], 4.0)
    assert np.isclose(dists[2], 5.0)
