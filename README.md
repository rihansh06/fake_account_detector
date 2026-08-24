# Fake Account Detector — starter kit

A working, minimal pipeline: CSV data → trained model → API → (your React frontend).
Everything here runs already with synthetic sample data, so you can demo it today
and improve it as you go.

## How to run it

```bash
pip install -r requirements.txt
python train_model.py     # trains the model, saves model.joblib, prints accuracy
python app.py              # starts the API on http://localhost:5000
```

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"account_age_days": 20, "follower_count": 5, "following_count": 1800, "follower_following_ratio": 0.003, "post_count": 0, "has_profile_pic": 0, "bio_length": 0, "username_digit_ratio": 0.6, "avg_posts_per_day": 0.0}'
```

## The 9 features (same meaning across Twitter/X, Instagram, Reddit)

| Feature | What it means | Why it matters |
|---|---|---|
| account_age_days | Days since account creation | Fresh accounts are riskier |
| follower_count | Number of followers | Fake accounts often have very few |
| following_count | Number of accounts followed | Fake accounts often follow thousands |
| follower_following_ratio | followers / following | Real accounts tend to be balanced or follower-heavy; bots follow way more than they're followed |
| post_count | Total posts/tweets | Fakes often have 0–3 posts |
| has_profile_pic | 1 if a real photo is set, else 0 | Bots frequently skip this |
| bio_length | Character count of bio | Empty bios are a signal |
| username_digit_ratio | Fraction of username that's digits (e.g. "john8827472" → high) | Auto-generated usernames lean fake |
| avg_posts_per_day | post_count / account_age_days | Either near-zero (abandoned bot) or unnaturally high (spam bot) is suspicious |

`is_fake`: 1 = fake, 0 = real — this is the label column your training data needs.

## Swapping in a real dataset

1. Search Kaggle for "Instagram fake account detection", "Twitter bot accounts dataset", or "Reddit bot detection".
2. Whatever raw columns it has, compute/rename them into the 9 columns above.
   Example: if a dataset has `followers` and `friends`, that's your
   `follower_count` and `following_count` — just compute
   `follower_following_ratio = followers / max(friends, 1)`.
3. Combine datasets from different platforms into one CSV with these
   shared columns — that's what lets one model score any platform.
4. Replace `sample_data.csv` with your real file and re-run `train_model.py`.

## Connecting your React frontend

From React, just POST to `/predict`:
```js
const res = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    account_age_days: 20,
    follower_count: 5,
    following_count: 1800,
    follower_following_ratio: 0.003,
    post_count: 0,
    has_profile_pic: 0,
    bio_length: 0,
    username_digit_ratio: 0.6,
    avg_posts_per_day: 0.0,
  }),
});
const result = await res.json();
// { verdict: "fake", confidence: 0.995, fake_probability: 0.995 }
```

## For your demo (important)

Live scraping Twitter/X, Instagram, and Reddit during a demo is risky —
X's API is paid/rate-limited and Instagram scraping breaks ToS quickly.
Only Reddit has a genuinely open API. Prepare a batch of sample profiles
(or a CSV upload feature) instead of promising live lookups on stage.

## Optional: rule-based baseline (no ML at all)

If you want a zero-ML fallback or a way to explain the model simply to
judges, the same 9 features can be combined into a manual weighted score
instead of a trained model — useful as a sanity check or a backup demo path.
