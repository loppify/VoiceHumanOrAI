import os
import glob
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('src'))
from bionic_core import BionicClassifier

classifier = BionicClassifier()
ai_files = glob.glob("dataset/ai/*.wav")
human_files = glob.glob("dataset/human/*.wav")

ai_rs = []
for f in ai_files:
    res = classifier.analyze_file(f)
    if res['mean_r'] > 0:
        ai_rs.append(res['mean_r'])

human_rs = []
for f in human_files:
    res = classifier.analyze_file(f)
    if res['mean_r'] > 0:
        human_rs.append(res['mean_r'])

print(f"AI Mean R: {np.mean(ai_rs):.2f}, Std: {np.std(ai_rs):.2f}, Min: {np.min(ai_rs):.2f}, Max: {np.max(ai_rs):.2f}")
print(f"Human Mean R: {np.mean(human_rs):.2f}, Std: {np.std(human_rs):.2f}, Min: {np.min(human_rs):.2f}, Max: {np.max(human_rs):.2f}")
