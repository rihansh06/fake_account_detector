
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

FILENAME = "twitter_human_bots_dataset.csv"  # change if your file is named differently

df = pd.read_csv(FILENAME)

# ---- 1. Clean up / engineer features --------------------------------------
df["description"] = df["description"].fillna("")
df["description_length"] = df["description"].str.len()

df["screen_name"] = df["screen_name"].fillna("")
df["username_digit_ratio"] = df["screen_name"].apply(
    lambda s: sum(c.isdigit() for c in s) / len(s) if len(s) > 0 else 0
)

df["follower_following_ratio"] = df["followers_count"] / (df["friends_count"] + 1)

# has_profile_pic: default_profile_image == 1 means NO custom picture (Twitter's default egg/avatar)
df["has_profile_pic"] = 1 - df["default_profile_image"].fillna(1).astype(int)

# Target: account_type is 'bot' or 'human' in this dataset
df["is_fake"] = (df["account_type"].str.lower() == "bot").astype(int)

FEATURE_COLUMNS = [
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
for col in ["verified", "geo_enabled", "default_profile"]:
    df[col] = df[col].fillna(False).astype(int)

# ---- 2. Split into train/test ourselves --------------------------------------
# (the dataset's own 'split' column turned out to use numeric codes rather
# than the words train/test, so we make our own 75/25 split instead)
X = df[FEATURE_COLUMNS]
y = df["is_fake"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---- 3. Train ----------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ---- 4. Evaluate ---------------------------------------------------------------
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds, target_names=["human", "bot"]))

importances = sorted(
    zip(FEATURE_COLUMNS, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)
print("\nFeature importance (most to least predictive):")
for name, score in importances:
    print(f"  {name:28s} {score:.3f}")

# ---- 5. Save ----------------------------------------------------------------
joblib.dump(model, "twitter_model.joblib")
print("\nSaved trained model to twitter_model.joblib")
