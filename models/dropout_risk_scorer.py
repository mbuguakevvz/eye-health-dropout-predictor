# models/dropout_risk_scorer.py
# Dropout Risk Scoring Model
# Trains a Random Forest classifier on enriched Kenya patient data
# Outputs risk scores, feature importance, and saves model for dashboard use

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH        = "db/eye_health.db"
TABLE          = "kenya_patients_enriched"
MODEL_PATH     = "models/dropout_rf_model.pkl"
SCORES_TABLE   = "kenya_risk_scores"
REPORT_PATH    = "models/model_report.txt"

# ── FEATURES ─────────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "county",
    "gender",
    "diagnosis",
    "referral_level",
    "screener_type",
    "age_group",
    "distance_band",
    "dropout_timing",
]

NUMERIC_FEATURES = [
    "age",
    "is_rural",
    "distance_to_facility_km",
    "sms_reminder_sent",
    "transport_support",
    "vulnerability_score",
    "screening_month",
    "screening_quarter",
]

TARGET = "dropped_out"


# ── LOAD DATA ────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    log.info(f"Loading enriched data from {DB_PATH} → {TABLE}")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    log.info(f"  Loaded {len(df):,} records")
    return df


# ── ENCODE FEATURES ──────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    log.info("Encoding categorical features...")

    encoders = {}
    df_model = df.copy()

    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le
        log.info(f"  Encoded: {col} ({len(le.classes_)} classes)")

    return df_model, encoders


# ── TRAIN MODEL ──────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    log.info("Preparing features and target...")

    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[all_features]
    y = df[TARGET]

    log.info(f"  Features: {len(all_features)}")
    log.info(f"  Target distribution — Dropped: {y.sum():,} | Completed: {(y==0).sum():,}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train Random Forest
    log.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    log.info("  Training complete.")

    # Evaluate
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)
    cv_scores   = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    log.info(f"  Accuracy : {accuracy:.4f}")
    log.info(f"  ROC-AUC  : {roc_auc:.4f}")
    log.info(f"  CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, X_test, y_test, y_pred, y_prob, all_features, X_train, y_train


# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    ).round(2)

    return importance_df


# ── SCORE ALL PATIENTS ───────────────────────────────────────────────────────
def score_all_patients(df: pd.DataFrame, df_encoded: pd.DataFrame, model, feature_names: list):
    log.info("Scoring all patients...")

    all_features = feature_names
    X_all   = df_encoded[all_features]
    probs   = model.predict_proba(X_all)[:, 1]
    preds   = model.predict(X_all)

    scores_df = df[["patient_id", "county", "region", "age", "gender",
                     "diagnosis", "referral_level", "is_rural",
                     "distance_to_facility_km", "sms_reminder_sent",
                     "transport_support", "dropped_out",
                     "risk_tier", "vulnerability_score"]].copy()

    scores_df["predicted_dropout_prob"] = probs.round(4)
    scores_df["predicted_dropout"]      = preds
    scores_df["model_risk_tier"] = pd.cut(
        scores_df["predicted_dropout_prob"],
        bins=[0, 0.30, 0.50, 0.70, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    ).astype(str)
    scores_df["scored_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    return scores_df


# ── SAVE MODEL ───────────────────────────────────────────────────────────────
def save_model(model, encoders: dict):
    os.makedirs("models", exist_ok=True)
    payload = {"model": model, "encoders": encoders}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"  Model saved → {MODEL_PATH}")


# ── SAVE SCORES TO SQLITE ────────────────────────────────────────────────────
def save_scores(scores_df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    scores_df.to_sql(SCORES_TABLE, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_score_county ON {SCORES_TABLE}(county)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_score_risk   ON {SCORES_TABLE}(model_risk_tier)")
    conn.commit()
    conn.close()
    log.info(f"  Scores saved → {DB_PATH} (table: {SCORES_TABLE})")


# ── SAVE REPORT ──────────────────────────────────────────────────────────────
def save_report(model, X_test, y_test, y_pred, y_prob, importance_df, scores_df):
    os.makedirs("models", exist_ok=True)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("="*55 + "\n")
        f.write("  DROPOUT RISK MODEL REPORT\n")
        f.write(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write("="*55 + "\n\n")

        f.write(f"Model        : Random Forest (200 trees)\n")
        f.write(f"Accuracy     : {accuracy:.4f}\n")
        f.write(f"ROC-AUC      : {roc_auc:.4f}\n\n")

        f.write("── Classification Report ──\n")
        f.write(classification_report(y_test, y_pred,
                target_names=["Completed", "Dropped Out"]))

        f.write("\n── Confusion Matrix ──\n")
        cm = confusion_matrix(y_test, y_pred)
        f.write(f"  True Negatives  (Correctly predicted completed) : {cm[0][0]:,}\n")
        f.write(f"  False Positives (Flagged but completed)         : {cm[0][1]:,}\n")
        f.write(f"  False Negatives (Missed dropouts)               : {cm[1][0]:,}\n")
        f.write(f"  True Positives  (Correctly predicted dropout)   : {cm[1][1]:,}\n")

        f.write("\n── Top 10 Feature Importances ──\n")
        f.write(importance_df.head(10).to_string(index=False))

        f.write("\n\n── Risk Score Distribution ──\n")
        dist = scores_df["model_risk_tier"].value_counts()
        f.write(dist.to_string())

    log.info(f"  Report saved → {REPORT_PATH}")


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
def print_summary(importance_df, scores_df, y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)

    print("\n" + "="*55)
    print("  DROPOUT RISK MODEL — SUMMARY")
    print("="*55)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    print("\n── Top 10 Most Important Features ──")
    print(importance_df.head(10).to_string(index=False))

    print("\n── Predicted Risk Distribution ──")
    print(scores_df["model_risk_tier"].value_counts().to_string())

    print("\n── High/Critical Risk by County ──")
    high_risk = scores_df[scores_df["model_risk_tier"].isin(["High Risk", "Critical Risk"])]
    county_risk = high_risk.groupby("county").size().sort_values(ascending=False)
    print(county_risk.to_string())
    print("="*55)


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting dropout risk scoring model...")

    df                  = load_data()
    df_encoded, encoders = encode_features(df)

    model, X_test, y_test, y_pred, y_prob, feature_names, X_train, y_train = train_model(df_encoded)

    importance_df = get_feature_importance(model, feature_names)
    scores_df     = score_all_patients(df, df_encoded, model, feature_names)

    save_model(model, encoders)
    save_scores(scores_df)
    save_report(model, X_test, y_test, y_pred, y_prob, importance_df, scores_df)
    print_summary(importance_df, scores_df, y_test, y_pred, y_prob)

    log.info("Scoring model completed successfully.")