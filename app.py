"""
app.py (multi-platform)

One API, two models. The frontend sends a "platform" field ("instagram"
or "twitter") plus that platform's raw fields, and this routes to the
right trained model.

Folder layout expected:
  fake-account-detector/
    app.py                  <- this file
    instagram_model.joblib  <- copy of the Instagram model.joblib, renamed
    twitter_model.joblib    <- copy of the Twitter twitter_model.joblib

Run with:
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

instagram_model = joblib.load("instagram_model.joblib")
twitter_model = joblib.load("twitter_model.joblib")

INSTAGRAM_RAW_COLUMNS = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
]

TWITTER_RAW_COLUMNS = [
    "account_age_days",
    "followers_count",
    "friends_count",
    "statuses_count",
    "description_length",
    "username_digit_ratio",
    "average_tweets_per_day",
    "verified",
    "geo_enabled",
    "default_profile",
    "favourites_count",
    "has_profile_pic",
]


# These orders must exactly match FEATURE_COLUMNS in each train_model.py —
# scikit-learn requires the same column order at predict time as at fit time.
INSTAGRAM_MODEL_COLUMN_ORDER = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
    "follower_following_ratio",
]

TWITTER_MODEL_COLUMN_ORDER = [
    "account_age_days",
    "followers_count",
    "friends_count",
    "follower_following_ratio",
    "statuses_count",
    "has_profile_pic",
    "description_length",
    "username_digit_ratio",
    "average_tweets_per_day",
    "verified",
    "geo_enabled",
    "default_profile",
    "favourites_count",
]


def predict_instagram(data):
    missing = [f for f in INSTAGRAM_RAW_COLUMNS if f not in data]
    if missing:
        return None, missing
    row = {f: data[f] for f in INSTAGRAM_RAW_COLUMNS}
    row["follower_following_ratio"] = row["#followers"] / (row["#follows"] + 1)
    df = pd.DataFrame([row])[INSTAGRAM_MODEL_COLUMN_ORDER]  # enforce training column order
    pred = instagram_model.predict(df)[0]
    proba = instagram_model.predict_proba(df)[0]
    return pred, proba


def predict_twitter(data):
    missing = [f for f in TWITTER_RAW_COLUMNS if f not in data]
    if missing:
        return None, missing
    row = {f: data[f] for f in TWITTER_RAW_COLUMNS}
    row["follower_following_ratio"] = row["followers_count"] / (row["friends_count"] + 1)
    df = pd.DataFrame([row])[TWITTER_MODEL_COLUMN_ORDER]  # enforce training column order
    pred = twitter_model.predict(df)[0]
    proba = twitter_model.predict_proba(df)[0]
    return pred, proba


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    platform = data.get("platform", "").lower()

    if platform == "instagram":
        pred, result = predict_instagram(data)
    elif platform == "twitter":
        pred, result = predict_twitter(data)
    else:
        return jsonify({"error": "platform must be 'instagram' or 'twitter'"}), 400

    if pred is None:
        return jsonify({"error": f"Missing fields for {platform}: {result}"}), 400

    probabilities = result
    return jsonify({
        "platform": platform,
        "verdict": "fake" if pred == 1 else "real",
        "confidence": round(float(max(probabilities)), 3),
        "fake_probability": round(float(probabilities[1]), 3),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["instagram", "twitter"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
