import os
import sys
import numpy as np
import pandas as pd
import joblib
import librosa
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

MODELS_DIR = "models"
FACE_MATCH_THRESHOLD = 0.5
VOICE_MATCH_THRESHOLD = 0.5

print("Loading models...")
face_model = joblib.load(os.path.join(MODELS_DIR, "face_model.pkl"))
face_encoder = joblib.load(os.path.join(MODELS_DIR, "face_label_encoder.pkl"))

voice_model = joblib.load(os.path.join(MODELS_DIR, "voice_model.pkl"))
voice_encoder = joblib.load(os.path.join(MODELS_DIR, "voice_label_encoder.pkl"))

product_model = joblib.load(os.path.join(MODELS_DIR, "product_model.pkl"))
product_encoder = joblib.load(os.path.join(MODELS_DIR, "product_label_encoder.pkl"))
sentiment_encoder = joblib.load(os.path.join(MODELS_DIR, "sentiment_encoder.pkl"))
engagement_encoder = joblib.load(os.path.join(MODELS_DIR, "engagement_encoder.pkl"))
platform_encoder = joblib.load(os.path.join(MODELS_DIR, "platform_encoder.pkl"))

print("Loading face embedding model (this can take a moment the first time)...")
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained="vggface2").eval()


def get_face_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding.squeeze().numpy()


def verify_face(image_path):
    embedding = get_face_embedding(image_path)
    if embedding is None:
        print("No face detected in the image.")
        return None

    probs = face_model.predict_proba([embedding])[0]
    best_index = np.argmax(probs)
    confidence = probs[best_index]
    person = face_encoder.inverse_transform([best_index])[0]

    print(f"Face check: closest match is {person}, confidence {confidence:.2f}")

    if confidence >= FACE_MATCH_THRESHOLD:
        return person
    else:
        return None


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    return [
        np.mean(mfcc),
        np.std(mfcc),
        np.mean(rolloff),
        np.std(rolloff),
        np.mean(rms),
        np.std(rms),
    ]


def verify_voice(audio_path):
    features = extract_audio_features(audio_path)
    probs = voice_model.predict_proba([features])[0]
    best_index = np.argmax(probs)
    confidence = probs[best_index]
    person = voice_encoder.inverse_transform([best_index])[0]

    print(f"Voice check: closest match is {person}, confidence {confidence:.2f}")

    if confidence >= VOICE_MATCH_THRESHOLD:
        return person
    else:
        return None


def predict_product(customer_row):
    features = [[
        platform_encoder.transform([customer_row["social_media_platform"]])[0],
        customer_row["engagement_score"],
        customer_row["purchase_interest_score"],
        sentiment_encoder.transform([customer_row["review_sentiment"]])[0],
        customer_row["purchase_amount"],
        customer_row["customer_rating"],
        customer_row["high_value_customer"],
        engagement_encoder.transform([customer_row["engagement_level"]])[0],
    ]]
    pred_index = product_model.predict(features)[0]
    return product_encoder.inverse_transform([pred_index])[0]


def run_transaction(face_image_path, voice_audio_path, customer_row):
    print("\n--- Starting Transaction ---")

    face_person = verify_face(face_image_path)
    if face_person is None:
        print("ACCESS DENIED: face not recognized.\n")
        return

    voice_person = verify_voice(voice_audio_path)
    if voice_person is None:
        print("ACCESS DENIED: voice not recognized.\n")
        return

    print(f"ACCESS GRANTED for {face_person}.")
    product = predict_product(customer_row)
    print(f"Recommended product category: {product}\n")


if __name__ == "__main__":
    sample_customer = {
        "social_media_platform": "Twitter",
        "engagement_score": 82,
        "purchase_interest_score": 4.8,
        "review_sentiment": "Neutral",
        "purchase_amount": 333,
        "customer_rating": 3.8,
        "high_value_customer": 1,
        "engagement_level": "High",
    }

    print("=== VALID USER SIMULATION ===")
    print("Enter the path to a face image for a known team member.")
    face_path = input("Face image path: ").strip()
    print("Enter the path to a matching voice sample.")
    audio_path = input("Voice audio path: ").strip()
    run_transaction(face_path, audio_path, sample_customer)

    print("=== UNAUTHORIZED ATTEMPT SIMULATION ===")
    print("Enter the path to a face image NOT in the training set, or a random photo.")
    bad_face_path = input("Face image path: ").strip()
    print("Enter the path to any voice sample.")
    bad_audio_path = input("Voice audio path: ").strip()
    run_transaction(bad_face_path, bad_audio_path, sample_customer)
