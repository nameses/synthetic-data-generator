#!/usr/bin/env python3
# prediction_real_only_scaling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ───── 1. CONFIG ─────
ORIGINAL_PATH = "../datasets/airline-passenger-satisfaction.csv"
SYNTH_PATH = "synthetic.csv"

NUM_FEATS = [
    "Age",
    #"Flight Distance",
    "Inflight wifi service",
    "Departure or Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]
CAT_FEATS = ["Gender", "Customer Type", "Type of Travel", "Class"]

# ───── 2. LOAD & SPLIT ORIGINAL DATA ─────
df_orig = pd.read_csv(ORIGINAL_PATH).dropna()
df_orig.describe(include="all").to_csv('df_orig.csv')
# map labels to 0/1
df_orig["satisfaction"] = df_orig["satisfaction"].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})

X = df_orig.drop("satisfaction", axis=1)
y = df_orig["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ───── 3. FIT PREPROCESSOR ON REAL DATA ONLY ─────
preproc = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])
preproc.fit(X_train)

# ───── 4. TRANSFORM REAL + SYNTHETIC ─────
# 4.1 real
X_train_real_scaled = preproc.transform(X_train)

# 4.2 synthetic
df_synth = pd.read_csv(SYNTH_PATH)
df_synth.describe(include="all").to_csv("df_synth.csv")

df_synth["satisfaction"] = df_synth["satisfaction"].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})
X_synth = df_synth.drop("satisfaction", axis=1)
y_synth = df_synth["satisfaction"]

X_synth_scaled = preproc.transform(X_synth)

# ───── 5. STACK & SHUFFLE ─────
X_aug = np.vstack([X_train_real_scaled, X_synth_scaled])
y_aug = np.concatenate([y_train.values, y_synth.values])

# optional: shuffle
from sklearn.utils import shuffle

X_aug, y_aug = shuffle(X_aug, y_aug, random_state=42)

# ───── 6. TRAIN CLASSIFIER ─────
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=100,
    random_state=42,
    verbose=False
)
mlp.fit(X_aug, y_aug)

# ───── 7. EVALUATE ON REAL TEST SET ─────
X_test_scaled = preproc.transform(X_test)
y_pred = mlp.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(
    y_test, y_pred, target_names=["dissatisfied", "satisfied"]
))
