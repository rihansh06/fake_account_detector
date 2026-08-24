"""
train_model.py (v2 — real Instagram dataset)

Trains on the Kaggle "Instagram Fake/Spammer/Genuine Accounts" dataset.
This dataset ships with its own train.csv and test.csv, so we use their
split directly instead of making our own.

Put train.csv and test.csv in this same folder, then run:

    python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---- 1. Load the dataset's own train/test split -------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ---- 2. Feature engineering ----------------------------------------------
# Add a follower/following ratio — a strong signal the raw columns don't
# give us directly. +1 avoids divide-by-zero for accounts following no one.
for df in (train_df, test_df):
    df["follower_following_ratio"] = df["#followers"] / (df["#follows"] + 1)

FEATURE_COLUMNS = [
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
TARGET_COLUMN = "fake"  # 1 = fake, 0 = real

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]

# ---- 3. Train --------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ---- 4. Evaluate on the held-out test set -----------------------------------
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds, target_names=["real", "fake"]))

importances = sorted(
    zip(FEATURE_COLUMNS, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)
print("\nFeature importance (most to least predictive):")
for name, score in importances:
    print(f"  {name:28s} {score:.3f}")

# ---- 5. Save the model -------------------------------------------------------
joblib.dump(model, "model.joblib")
print("\nSaved trained model to model.joblib")
