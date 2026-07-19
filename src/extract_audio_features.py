import os
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = "data/audio"
OUT_PATH = "data/features/audio_features.csv"

def get_person(filename):
    name = filename.lower()
    if "david" in name:
        return "David"
    if "hamse" in name:
        return "Hamse"
    if "celine" in name:
        return "Celine"
    return "Unknown"

def get_phrase(filename):
    name = filename.lower()
    if "confirm" in name:
        return "confirm_transaction"
    if "yes" in name or "approve" in name:
        return "yes_approve"
    return "unknown"

def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return {
        "mfcc_mean": np.mean(mfcc),
        "mfcc_std": np.std(mfcc),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "energy_mean": np.mean(rms),
        "energy_std": np.std(rms)
    }

rows = []
for file in sorted(os.listdir(AUDIO_DIR)):
    if not file.endswith(".wav"):
        continue
    path = os.path.join(AUDIO_DIR, file)
    feats = extract_features(path)
    feats["file"] = file
    feats["person"] = get_person(file)
    feats["phrase"] = get_phrase(file)
    rows.append(feats)

df = pd.DataFrame(rows)
cols = ["file", "person", "phrase", "mfcc_mean", "mfcc_std", "rolloff_mean", "rolloff_std", "energy_mean", "energy_std"]
df = df[cols]
df.to_csv(OUT_PATH, index=False)
print(df)
print("Saved to", OUT_PATH)
