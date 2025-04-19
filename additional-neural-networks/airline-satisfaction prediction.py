# %% [markdown]
# ## 1. Imports & Config

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# %%
# — configure your file paths here:
ORIGINAL_PATH = "../datasets/airline-passenger-satisfaction.csv"
SYNTH_PATHS = [
    "gan-synthetic-cutted.csv",
    # "gan-synthetic.csv",
    # "vae-synthetic.csv",
]

# numeric vs categorical columns
NUM_FEATS = [
    "Age",
    # "Flight Distance",
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
    # "Departure Delay in Minutes",
    # "Arrival Delay in Minutes"
]
CAT_FEATS = ["Gender", "Customer Type", "Type of Travel", "Class"]


# helper to build a fresh pipeline
def build_pipeline():
    preproc = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])
    return Pipeline([
        ("preproc", preproc),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=100,
            verbose=False,
            random_state=42
        ))
    ])


# %% [markdown]
# ## 2. Load & Split Original Data

# %%
df_orig = pd.read_csv(ORIGINAL_PATH).dropna()
df_orig["satisfaction"] = df_orig["satisfaction"].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})

X = df_orig.drop("satisfaction", axis=1)
y = df_orig["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% [markdown]
# ## 3. (A) Train on Original Only

# %%
print("### Training on ORIGINAL only\n")
pipe = build_pipeline()
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["dissatisfied", "satisfied"]))

# %% [markdown]
# ## 3. (B) Train on Original + Synthetic Sets

# %%
for synth_path in SYNTH_PATHS:
    name = synth_path.rsplit("/", 1)[-1]
    # load & map
    df_s = pd.read_csv(synth_path)
    df_s["satisfaction"] = df_s["satisfaction"].map({
        "satisfied": 1,
        "neutral or dissatisfied": 0
    })
    X_s, y_s = df_s.drop("satisfaction", axis=1), df_s["satisfaction"]
    # augment only train set
    X_train_aug = pd.concat([X_train, X_s], ignore_index=True)
    y_train_aug = pd.concat([y_train, y_s], ignore_index=True)
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

    print(f"\n### Training on ORIGINAL + {name}\n")
    pipe = build_pipeline()
    pipe.fit(X_train_aug, y_train_aug)
    y_pred = pipe.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["dissatisfied", "satisfied"]))
# %% [markdown]
# ### 4. Compare “no mix” vs. “mixed” for each synthetic set

# %%
# from sklearn.utils import shuffle
#
# def eval_train(X_tr, y_tr, X_te, y_te):
#     pipe = build_pipeline()
#     pipe.fit(X_tr, y_tr)
#     y_pred = pipe.predict(X_te)
#     return accuracy_score(y_te, y_pred)
#
# for synth_path in SYNTH_PATHS:
#     name = synth_path.rsplit("/", 1)[-1]
#     # load & encode
#     df_s = pd.read_csv(synth_path)
#     df_s["satisfaction"] = df_s["satisfaction"].map({"satisfied":1, "neutral or dissatisfied":0})
#     X_s, y_s = df_s.drop("satisfaction", axis=1), df_s["satisfaction"]
#
#     # augment original train
#     X_aug = pd.concat([X_train, X_s], ignore_index=True)
#     y_aug = pd.concat([y_train, y_s], ignore_index=True)
#
#     # 1) NO SHUFFLE
#     acc_nomix = eval_train(X_aug, y_aug, X_test, y_test)
#
#     # 2) SHUFFLE (“mix up”)
#     X_mix, y_mix = shuffle(X_aug, y_aug, random_state=42)
#     acc_mix   = eval_train(X_mix, y_mix, X_test, y_test)
#
#     print(f"\n**{name}**")
#     print(f" • No mix   Accuracy: {acc_nomix:.4f}")
#     print(f" • Mixed    Accuracy: {acc_mix:.4f}")
