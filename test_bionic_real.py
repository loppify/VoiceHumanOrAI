import sys
import os
import glob
sys.path.insert(0, os.path.abspath('src'))
from bionic_core import BionicClassifier

bionic_model = BionicClassifier()
print("--- Перевірка AI файлів ---")
for f in glob.glob("dataset/ai/*.wav")[:5]:
    res = bionic_model.analyze_file(f)
    print(f"{os.path.basename(f)}: {res['mean_r']:.2f} -> {res['verdict']}")

print("\n--- Перевірка Human файлів ---")
for f in glob.glob("dataset/human/*.wav")[:5]:
    res = bionic_model.analyze_file(f)
    print(f"{os.path.basename(f)}: {res['mean_r']:.2f} -> {res['verdict']}")
