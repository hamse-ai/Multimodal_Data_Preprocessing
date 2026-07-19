import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

# ---------------------------------------------------------
# Model 1: Facial Recognition Model
# Trained on the embeddings extracted from each team member's photos.
# ---------------------------------------------------------

print("=== Facial Recognition Model ===")

face_df = pd.read_csv("data/features/face_embeddings.csv")

# clean up the person label, "David faces" should just be "David"
face_df["person"] = face_df["person"].str.replace(" faces", "", regex=False)

emb_cols = [c for c in face_df.columns if c.startswith("emb_")]
X_face = face_df[emb_cols]
y_face = face_df["person"]

face_encoder = LabelEncoder()
y_face_enc = face_encoder.fit_transform(y_face)

X_train, X_test, y_train, y_test = train_test_split(
    X_face, y_face_enc, test_size=0.25, random_state=42, stratify=y_face_enc
)

face_model = RandomForestClassifier(n_estimators=100, random_state=42)
face_model.fit(X_train, y_train)

face_preds = face_model.predict(X_test)
face_probs = face_model.predict_proba(X_test)

face_acc = accuracy_score(y_test, face_preds)
face_f1 = f1_score(y_test, face_preds, average="weighted")
face_loss = log_loss(y_test, face_probs, labels=face_model.classes_)

print("Accuracy:", round(face_acc, 3))
print("F1 Score:", round(face_f1, 3))
print("Loss:", round(face_loss, 3))
print()

# ---------------------------------------------------------
# Model 2: Voiceprint Verification Model
# Trained on the audio features extracted from each team member's recordings.
# ---------------------------------------------------------

print("=== Voiceprint Verification Model ===")

audio_df = pd.read_csv("data/features/audio_features.csv")

audio_feature_cols = ["mfcc_mean", "mfcc_std", "rolloff_mean", "rolloff_std", "energy_mean", "energy_std"]
X_audio = audio_df[audio_feature_cols]
y_audio = audio_df["person"]

audio_encoder = LabelEncoder()
y_audio_enc = audio_encoder.fit_transform(y_audio)

# small dataset, so we keep the test split small too
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_audio, y_audio_enc, test_size=0.34, random_state=42, stratify=y_audio_enc
)

voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
voice_model.fit(X_train_a, y_train_a)

voice_preds = voice_model.predict(X_test_a)
voice_probs = voice_model.predict_proba(X_test_a)

voice_acc = accuracy_score(y_test_a, voice_preds)
voice_f1 = f1_score(y_test_a, voice_preds, average="weighted")
voice_loss = log_loss(y_test_a, voice_probs, labels=voice_model.classes_)

print("Accuracy:", round(voice_acc, 3))
print("F1 Score:", round(voice_f1, 3))
print("Loss:", round(voice_loss, 3))
print()

# ---------------------------------------------------------
# Model 3: Product Recommendation Model
# Trained on the merged social profile + transaction dataset.
# ---------------------------------------------------------

print("=== Product Recommendation Model ===")

cust_df = pd.read_csv("data/raw/merged_customer_data.csv")

# turn text columns into numbers the model can use
sentiment_encoder = LabelEncoder()
cust_df["review_sentiment_enc"] = sentiment_encoder.fit_transform(cust_df["review_sentiment"])

engagement_encoder = LabelEncoder()
cust_df["engagement_level_enc"] = engagement_encoder.fit_transform(cust_df["engagement_level"])

platform_encoder = LabelEncoder()
cust_df["platform_enc"] = platform_encoder.fit_transform(cust_df["social_media_platform"])

feature_cols = [
    "platform_enc",
    "engagement_score",
    "purchase_interest_score",
    "review_sentiment_enc",
    "purchase_amount",
    "customer_rating",
    "high_value_customer",
    "engagement_level_enc",
]

X_prod = cust_df[feature_cols]
y_prod = cust_df["product_category"]

product_encoder = LabelEncoder()
y_prod_enc = product_encoder.fit_transform(y_prod)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prod, y_prod_enc, test_size=0.25, random_state=42, stratify=y_prod_enc
)

product_model = RandomForestClassifier(n_estimators=200, random_state=42)
product_model.fit(X_train_p, y_train_p)

product_preds = product_model.predict(X_test_p)
product_probs = product_model.predict_proba(X_test_p)

product_acc = accuracy_score(y_test_p, product_preds)
product_f1 = f1_score(y_test_p, product_preds, average="weighted")
product_loss = log_loss(y_test_p, product_probs, labels=product_model.classes_)

print("Accuracy:", round(product_acc, 3))
print("F1 Score:", round(product_f1, 3))
print("Loss:", round(product_loss, 3))
print()

# ---------------------------------------------------------
# Save all three models and their encoders so the CLI simulation
# script can load them later without retraining.
# ---------------------------------------------------------

import joblib

joblib.dump(face_model, "models/face_model.pkl")
joblib.dump(face_encoder, "models/face_label_encoder.pkl")

joblib.dump(voice_model, "models/voice_model.pkl")
joblib.dump(audio_encoder, "models/voice_label_encoder.pkl")

joblib.dump(product_model, "models/product_model.pkl")
joblib.dump(product_encoder, "models/product_label_encoder.pkl")
joblib.dump(sentiment_encoder, "models/sentiment_encoder.pkl")
joblib.dump(engagement_encoder, "models/engagement_encoder.pkl")
joblib.dump(platform_encoder, "models/platform_encoder.pkl")

print("All models saved to the models/ folder.")

results = {
    "model": ["Facial Recognition", "Voiceprint Verification", "Product Recommendation"],
    "accuracy": [face_acc, voice_acc, product_acc],
    "f1_score": [face_f1, voice_f1, product_f1],
    "loss": [face_loss, voice_loss, product_loss],
}
results_df = pd.DataFrame(results)
results_df.to_csv("models/evaluation_results.csv", index=False)
print(results_df)
