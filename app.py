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
import re
import os
from datetime import datetime, timezone
import instaloader
import requests

app = Flask(__name__)
CORS(app)

instagram_model = joblib.load("instagram_model.joblib")
twitter_model = joblib.load("twitter_model.joblib")

# Set this as an environment variable on Render (Settings > Environment).
# Get a free key + 100k credits at https://twitterapi.io
TWITTERAPI_IO_KEY = os.environ.get("TWITTERAPI_IO_KEY", "")


def extract_username(url_or_username: str, platform: str) -> str:
    """Pulls a bare username out of a full profile URL, or returns it as-is."""
    if platform == "instagram":
        match = re.search(r"instagram\.com/([^/?]+)", url_or_username)
    else:  # twitter/x
        match = re.search(r"(?:twitter|x)\.com/([^/?]+)", url_or_username)
    if match:
        return match.group(1)
    return url_or_username.strip().lstrip("@")


def scrape_instagram(username: str) -> dict:
    L = instaloader.Instaloader(
        max_connection_attempts=1,  # don't retry — fail fast instead of hanging
        request_timeout=15,
        sleep=False,  # disable automatic sleep-and-retry on rate limits
    )

    profile = instaloader.Profile.from_username(L.context, username)

    full_name = profile.full_name or ""
    bio = profile.biography or ""
    digit_ratio = (sum(c.isdigit() for c in username) / len(username)) if username else 0
    fullname_digit_ratio = (sum(c.isdigit() for c in full_name) / len(full_name)) if full_name else 0

    return {
        "profile pic": 1 if profile.profile_pic_url else 0,
        "nums/length username": round(digit_ratio, 3),
        "fullname words": len(full_name.split()) if full_name else 0,
        "nums/length fullname": round(fullname_digit_ratio, 3),
        "name==username": 1 if full_name.lower() == username.lower() else 0,
        "description length": len(bio),
        "external URL": 1 if profile.external_url else 0,
        "private": 1 if profile.is_private else 0,
        "#posts": profile.mediacount,
        "#followers": profile.followers,
        "#follows": profile.followees,
    }


def fetch_twitter(username: str) -> dict:
    if not TWITTERAPI_IO_KEY:
        raise RuntimeError("TWITTERAPI_IO_KEY environment variable is not set")

    resp = requests.get(
        "https://api.twitterapi.io/twitter/user/info",
        params={"userName": username},
        headers={"X-API-Key": TWITTERAPI_IO_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("msg", "twitterapi.io request failed"))

    d = body["data"]

    raw_created_at = d["createdAt"]
    try:
        # Classic Twitter format, e.g. "Thu Dec 13 08:41:26 +0000 2007"
        created_at = datetime.strptime(raw_created_at, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        # ISO 8601 format, e.g. "2009-06-02T20:12:29.000000Z"
        created_at = datetime.strptime(raw_created_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    account_age_days = (datetime.now(timezone.utc) - created_at).days
    account_age_days = max(account_age_days, 1)  # avoid div-by-zero

    username_val = d.get("userName", username)
    digit_ratio = (sum(c.isdigit() for c in username_val) / len(username_val)) if username_val else 0
    description = d.get("description", "") or ""
    statuses_count = d.get("statusesCount", 0)

    return {
        "account_age_days": account_age_days,
        "followers_count": d.get("followers", 0),
        "friends_count": d.get("following", 0),
        "statuses_count": statuses_count,
        "description_length": len(description),
        "username_digit_ratio": round(digit_ratio, 3),
        "average_tweets_per_day": round(statuses_count / account_age_days, 3),
        "verified": 1 if d.get("isBlueVerified") else 0,
        # twitterapi.io doesn't expose these two — default to 0.
        # They were low-importance features in training (see README),
        # so this doesn't meaningfully hurt accuracy.
        "geo_enabled": 0,
        "default_profile": 0,
        "favourites_count": d.get("favouritesCount", 0),
        "has_profile_pic": 1 if d.get("profilePicture") else 0,
    }

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


@app.route("/analyze/instagram", methods=["POST"])
def analyze_instagram():
    data = request.get_json(force=True)
    raw = data.get("username") or data.get("url")
    if not raw:
        return jsonify({"error": "Provide 'username' or 'url'"}), 400

    username = extract_username(raw, "instagram")

    try:
        features = scrape_instagram(username)
    except instaloader.exceptions.ProfileNotExistsException:
        return jsonify({"error": f"No such Instagram profile: {username}"}), 404
    except instaloader.exceptions.TooManyRequestsException:
        return jsonify({
            "error": "Instagram rate-limited this server (common for cloud-hosted IPs). "
                     "Try again later, or use pre-fetched demo data for this platform."
        }), 429
    except Exception as e:
        return jsonify({"error": f"Couldn't fetch Instagram profile: {e}"}), 502

    row = dict(features)
    row["follower_following_ratio"] = row["#followers"] / (row["#follows"] + 1)
    df = pd.DataFrame([row])[INSTAGRAM_MODEL_COLUMN_ORDER]
    pred = instagram_model.predict(df)[0]
    proba = instagram_model.predict_proba(df)[0]

    return jsonify({
        "platform": "instagram",
        "username": username,
        "verdict": "fake" if pred == 1 else "real",
        "confidence": round(float(max(proba)), 3),
        "fake_probability": round(float(proba[1]), 3),
        "extracted_features": features,
    })


@app.route("/analyze/twitter", methods=["POST"])
def analyze_twitter():
    data = request.get_json(force=True)
    raw = data.get("username") or data.get("url")
    if not raw:
        return jsonify({"error": "Provide 'username' or 'url'"}), 400

    username = extract_username(raw, "twitter")

    try:
        features = fetch_twitter(username)
    except Exception as e:
        return jsonify({"error": f"Couldn't fetch Twitter profile: {e}"}), 502

    row = dict(features)
    row["follower_following_ratio"] = row["followers_count"] / (row["friends_count"] + 1)
    df = pd.DataFrame([row])[TWITTER_MODEL_COLUMN_ORDER]
    pred = twitter_model.predict(df)[0]
    proba = twitter_model.predict_proba(df)[0]

    return jsonify({
        "platform": "twitter",
        "username": username,
        "verdict": "fake" if pred == 1 else "real",
        "confidence": round(float(max(proba)), 3),
        "fake_probability": round(float(proba[1]), 3),
        "extracted_features": features,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["instagram", "twitter"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
