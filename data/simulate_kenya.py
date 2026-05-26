# data/simulate_kenya.py
# Kenya Community Eye Health - Synthetic Patient Data Simulator
# Modeled on the Vision Impact Project (CBM + Peek Vision, 7 counties)

import pandas as pd
import numpy as np
import sqlite3
import os
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
NUM_PATIENTS = 50000
OUTPUT_CSV   = "data/kenya_patients.csv"
OUTPUT_DB    = "db/eye_health.db"

# ── KENYA GEOGRAPHY ─────────────────────────────────────────────────────────
COUNTIES = {
    "Bomet":    {"region": "Rift Valley", "rural_pct": 0.82, "facility_distance_avg": 34},
    "Vihiga":   {"region": "Western",     "rural_pct": 0.78, "facility_distance_avg": 28},
    "Kisii":    {"region": "Nyanza",      "rural_pct": 0.75, "facility_distance_avg": 22},
    "Nyeri":    {"region": "Central",     "rural_pct": 0.60, "facility_distance_avg": 18},
    "Nakuru":   {"region": "Rift Valley", "rural_pct": 0.55, "facility_distance_avg": 15},
    "Mombasa":  {"region": "Coast",       "rural_pct": 0.30, "facility_distance_avg": 8},
    "Nairobi":  {"region": "Nairobi",     "rural_pct": 0.10, "facility_distance_avg": 5},
}

# ── DIAGNOSIS CATEGORIES ────────────────────────────────────────────────────
DIAGNOSES = [
    "Refractive Error",
    "Cataract",
    "Presbyopia",
    "Glaucoma Suspect",
    "Allergic Conjunctivitis",
    "Diabetic Retinopathy",
]

DIAGNOSIS_WEIGHTS = [0.35, 0.25, 0.20, 0.08, 0.08, 0.04]

# ── REFERRAL LEVELS ─────────────────────────────────────────────────────────
REFERRAL_LEVELS = ["Primary", "Secondary", "Specialist"]

# ── SCREENER TYPES ──────────────────────────────────────────────────────────
SCREENER_TYPES = ["Community Health Volunteer", "Nurse", "Clinical Officer", "Optometrist"]
SCREENER_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

# ── DROPOUT LOGIC ───────────────────────────────────────────────────────────
def calculate_dropout_probability(row):
    """
    Realistic dropout probability based on known barriers
    in community eye health programs in Sub-Saharan Africa.
    """
    prob = 0.30  # base dropout rate

    # Distance penalty
    if row["distance_to_facility_km"] > 30:
        prob += 0.25
    elif row["distance_to_facility_km"] > 15:
        prob += 0.12

    # Rural penalty
    if row["is_rural"] == 1:
        prob += 0.10

    # Age factor - elderly less likely to travel
    if row["age"] >= 60:
        prob += 0.10
    elif row["age"] <= 18:
        prob += 0.05

    # Gender gap in some regions
    if row["gender"] == "Female":
        prob += 0.05

    # SMS reminder reduces dropout
    if row["sms_reminder_sent"] == 1:
        prob -= 0.12

    # Transport support reduces dropout
    if row["transport_support"] == 1:
        prob -= 0.15

    # Screener quality
    if row["screener_type"] in ["Optometrist", "Clinical Officer"]:
        prob -= 0.08

    # Specialist referrals have higher dropout (longer journey)
    if row["referral_level"] == "Specialist":
        prob += 0.15
    elif row["referral_level"] == "Primary":
        prob -= 0.05

    # Diagnosis urgency - cataract patients more motivated
    if row["diagnosis"] == "Cataract":
        prob -= 0.10
    elif row["diagnosis"] == "Allergic Conjunctivitis":
        prob += 0.10

    return float(np.clip(prob, 0.05, 0.95))


# ── SIMULATE ────────────────────────────────────────────────────────────────
def simulate_kenya_patients(n=NUM_PATIENTS):
    records = []

    for i in range(n):
        county_name = random.choice(list(COUNTIES.keys()))
        county      = COUNTIES[county_name]

        age         = int(np.random.choice(
                          range(5, 85),
                          p=np.array([1 if 5<=x<=85 else 0 for x in range(5,85)]) / 80
                      ))
        gender      = random.choice(["Male", "Female"])
        is_rural    = 1 if random.random() < county["rural_pct"] else 0

        distance    = max(1, int(np.random.normal(
                          county["facility_distance_avg"],
                          county["facility_distance_avg"] * 0.4
                      )))

        diagnosis       = random.choices(DIAGNOSES, weights=DIAGNOSIS_WEIGHTS)[0]
        referral_level  = random.choices(
                              REFERRAL_LEVELS,
                              weights=[0.50, 0.35, 0.15]
                          )[0]
        screener_type   = random.choices(SCREENER_TYPES, weights=SCREENER_WEIGHTS)[0]

        sms_reminder    = 1 if random.random() < 0.55 else 0
        transport_support = 1 if random.random() < 0.20 else 0

        screening_date  = fake.date_between(start_date="-2y", end_date="today")
        days_to_dropout = int(np.random.exponential(scale=14)) + 1

        row = {
            "patient_id":             f"KE-{str(i+1).zfill(6)}",
            "county":                 county_name,
            "region":                 county["region"],
            "age":                    age,
            "gender":                 gender,
            "is_rural":               is_rural,
            "distance_to_facility_km": distance,
            "diagnosis":              diagnosis,
            "referral_level":         referral_level,
            "screener_type":          screener_type,
            "sms_reminder_sent":      sms_reminder,
            "transport_support":      transport_support,
            "screening_date":         screening_date,
            "days_to_referral_dropout": days_to_dropout,
        }

        dropout_prob      = calculate_dropout_probability(row)
        row["dropout_probability"] = round(dropout_prob, 4)
        row["dropped_out"]         = 1 if random.random() < dropout_prob else 0

        records.append(row)

    return pd.DataFrame(records)


# ── SAVE ─────────────────────────────────────────────────────────────────────
def save_to_csv(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ CSV saved → {OUTPUT_CSV}  ({len(df):,} records)")


def save_to_sqlite(df):
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    df.to_sql("kenya_patients", conn, if_exists="replace", index=False)

    # Index for fast querying
    conn.execute("CREATE INDEX IF NOT EXISTS idx_county ON kenya_patients(county)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dropout ON kenya_patients(dropped_out)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis ON kenya_patients(diagnosis)")
    conn.commit()
    conn.close()
    print(f"✓ SQLite saved → {OUTPUT_DB}  (table: kenya_patients)")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Simulating Kenya patient data...")
    df = simulate_kenya_patients()
    save_to_csv(df)
    save_to_sqlite(df)

    print("\n── Sample Records ──")
    print(df.head(3).to_string())

    print("\n── Dropout Rate by County ──")
    summary = df.groupby("county")["dropped_out"].mean().round(3).sort_values(ascending=False)
    print(summary.to_string())

    print("\n── Dropout Rate by Diagnosis ──")
    diag = df.groupby("diagnosis")["dropped_out"].mean().round(3).sort_values(ascending=False)
    print(diag.to_string())