# data/simulate_global.py
# Global Community Eye Health - Synthetic Patient Data Simulator
# Expands Kenya model to 9 additional countries across 4 regions
# Each country has realistic health system parameters

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from faker import Faker
from datetime import datetime
import random

fake = Faker()
np.random.seed(99)
random.seed(99)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
PATIENTS_PER_COUNTRY = 10000
OUTPUT_CSV           = "data/global_patients.csv"
DB_PATH              = "db/eye_health.db"
TABLE_NAME           = "global_patients"

# ── COUNTRY PROFILES ─────────────────────────────────────────────────────────
# Parameters based on WHO/NGO published data per country
COUNTRIES = {
    # ── SUB-SAHARAN AFRICA ──
    "Uganda": {
        "region":               "Sub-Saharan Africa",
        "income_level":         "Low",
        "uhc_index":            37,
        "rural_pct":            0.76,
        "facility_distance_avg": 31,
        "sms_penetration":      0.48,
        "transport_support_pct": 0.15,
        "base_dropout":         0.42,
        "ngo_presence":         "Sightsavers, CBM",
        "currency":             "UGX",
    },
    "Ethiopia": {
        "region":               "Sub-Saharan Africa",
        "income_level":         "Low",
        "uhc_index":            34,
        "rural_pct":            0.79,
        "facility_distance_avg": 38,
        "sms_penetration":      0.40,
        "transport_support_pct": 0.10,
        "base_dropout":         0.48,
        "ngo_presence":         "CBM, Light for the World",
        "currency":             "ETB",
    },
    "Tanzania": {
        "region":               "Sub-Saharan Africa",
        "income_level":         "Low",
        "uhc_index":            39,
        "rural_pct":            0.67,
        "facility_distance_avg": 27,
        "sms_penetration":      0.52,
        "transport_support_pct": 0.18,
        "base_dropout":         0.38,
        "ngo_presence":         "Fred Hollows, Sightsavers",
        "currency":             "TZS",
    },
    # ── SOUTH ASIA ──
    "India": {
        "region":               "South Asia",
        "income_level":         "Lower-Middle",
        "uhc_index":            56,
        "rural_pct":            0.65,
        "facility_distance_avg": 18,
        "sms_penetration":      0.72,
        "transport_support_pct": 0.28,
        "base_dropout":         0.28,
        "ngo_presence":         "Aravind, Operation Eyesight, LAICO",
        "currency":             "INR",
    },
    "Bangladesh": {
        "region":               "South Asia",
        "income_level":         "Lower-Middle",
        "uhc_index":            48,
        "rural_pct":            0.61,
        "facility_distance_avg": 22,
        "sms_penetration":      0.65,
        "transport_support_pct": 0.20,
        "base_dropout":         0.33,
        "ngo_presence":         "Orbis, Brien Holden",
        "currency":             "BDT",
    },
    # ── SOUTHEAST ASIA ──
    "Myanmar": {
        "region":               "Southeast Asia",
        "income_level":         "Lower-Middle",
        "uhc_index":            42,
        "rural_pct":            0.70,
        "facility_distance_avg": 29,
        "sms_penetration":      0.55,
        "transport_support_pct": 0.14,
        "base_dropout":         0.40,
        "ngo_presence":         "Fred Hollows, Orbis",
        "currency":             "MMK",
    },
    "Cambodia": {
        "region":               "Southeast Asia",
        "income_level":         "Lower-Middle",
        "uhc_index":            44,
        "rural_pct":            0.63,
        "facility_distance_avg": 24,
        "sms_penetration":      0.60,
        "transport_support_pct": 0.16,
        "base_dropout":         0.36,
        "ngo_presence":         "Fred Hollows, Sightsavers",
        "currency":             "KHR",
    },
    # ── LATIN AMERICA ──
    "Peru": {
        "region":               "Latin America",
        "income_level":         "Upper-Middle",
        "uhc_index":            65,
        "rural_pct":            0.22,
        "facility_distance_avg": 14,
        "sms_penetration":      0.78,
        "transport_support_pct": 0.30,
        "base_dropout":         0.22,
        "ngo_presence":         "Brien Holden, Seeing is Believing",
        "currency":             "PEN",
    },
    "Bolivia": {
        "region":               "Latin America",
        "income_level":         "Lower-Middle",
        "uhc_index":            52,
        "rural_pct":            0.31,
        "facility_distance_avg": 20,
        "sms_penetration":      0.68,
        "transport_support_pct": 0.22,
        "base_dropout":         0.30,
        "ngo_presence":         "Brien Holden, CBM",
        "currency":             "BOB",
    },
    # ── PACIFIC ──
    "Papua New Guinea": {
        "region":               "Pacific",
        "income_level":         "Lower-Middle",
        "uhc_index":            32,
        "rural_pct":            0.87,
        "facility_distance_avg": 52,
        "sms_penetration":      0.35,
        "transport_support_pct": 0.08,
        "base_dropout":         0.58,
        "ngo_presence":         "Fred Hollows",
        "currency":             "PGK",
    },
}

# ── DIAGNOSES ────────────────────────────────────────────────────────────────
DIAGNOSES        = ["Refractive Error", "Cataract", "Presbyopia",
                    "Glaucoma Suspect", "Allergic Conjunctivitis", "Diabetic Retinopathy"]
DIAGNOSIS_WEIGHTS = [0.35, 0.25, 0.20, 0.08, 0.08, 0.04]

REFERRAL_LEVELS   = ["Primary", "Secondary", "Specialist"]
SCREENER_TYPES    = ["Community Health Volunteer", "Nurse", "Clinical Officer", "Optometrist"]
SCREENER_WEIGHTS  = [0.50, 0.25, 0.15, 0.10]


# ── DROPOUT LOGIC ─────────────────────────────────────────────────────────────
def calculate_dropout_probability(row, country_profile: dict) -> float:
    prob = country_profile["base_dropout"]

    # Distance penalty
    if row["distance_to_facility_km"] > 40:
        prob += 0.28
    elif row["distance_to_facility_km"] > 25:
        prob += 0.15
    elif row["distance_to_facility_km"] > 10:
        prob += 0.06

    # Rural penalty
    if row["is_rural"] == 1:
        prob += 0.10

    # UHC index — stronger health systems retain patients better
    uhc_penalty = (100 - country_profile["uhc_index"]) / 100 * 0.15
    prob += uhc_penalty

    # Age
    if row["age"] >= 60:
        prob += 0.10
    elif row["age"] <= 18:
        prob += 0.05

    # Gender
    if row["gender"] == "Female":
        prob += 0.05

    # Interventions
    if row["sms_reminder_sent"] == 1:
        prob -= 0.12
    if row["transport_support"] == 1:
        prob -= 0.15

    # Screener quality
    if row["screener_type"] in ["Optometrist", "Clinical Officer"]:
        prob -= 0.08

    # Referral level
    if row["referral_level"] == "Specialist":
        prob += 0.15
    elif row["referral_level"] == "Primary":
        prob -= 0.05

    # Diagnosis urgency
    if row["diagnosis"] == "Cataract":
        prob -= 0.10
    elif row["diagnosis"] == "Allergic Conjunctivitis":
        prob += 0.10

    return float(np.clip(prob, 0.05, 0.95))


# ── SIMULATE ──────────────────────────────────────────────────────────────────
def simulate_global_patients() -> pd.DataFrame:
    all_records = []
    patient_counter = 1

    for country_name, profile in COUNTRIES.items():
        log.info(f"  Simulating {PATIENTS_PER_COUNTRY:,} patients for {country_name}...")
        country_records = []

        for _ in range(PATIENTS_PER_COUNTRY):
            is_rural  = 1 if random.random() < profile["rural_pct"] else 0
            distance  = max(1, int(np.random.normal(
                            profile["facility_distance_avg"],
                            profile["facility_distance_avg"] * 0.4
                        )))
            age       = int(np.random.choice(range(5, 85)))
            gender    = random.choice(["Male", "Female"])

            diagnosis      = random.choices(DIAGNOSES, weights=DIAGNOSIS_WEIGHTS)[0]
            referral_level = random.choices(REFERRAL_LEVELS, weights=[0.50, 0.35, 0.15])[0]
            screener_type  = random.choices(SCREENER_TYPES, weights=SCREENER_WEIGHTS)[0]

            sms_reminder      = 1 if random.random() < profile["sms_penetration"] else 0
            transport_support = 1 if random.random() < profile["transport_support_pct"] else 0

            screening_date    = fake.date_between(start_date="-2y", end_date="today")
            days_to_dropout   = int(np.random.exponential(scale=14)) + 1

            row = {
                "patient_id":               f"GL-{str(patient_counter).zfill(7)}",
                "country":                  country_name,
                "region":                   profile["region"],
                "income_level":             profile["income_level"],
                "uhc_index":                profile["uhc_index"],
                "ngo_presence":             profile["ngo_presence"],
                "age":                      age,
                "gender":                   gender,
                "is_rural":                 is_rural,
                "distance_to_facility_km":  distance,
                "diagnosis":                diagnosis,
                "referral_level":           referral_level,
                "screener_type":            screener_type,
                "sms_reminder_sent":        sms_reminder,
                "transport_support":        transport_support,
                "screening_date":           screening_date,
                "days_to_referral_dropout": days_to_dropout,
            }

            dropout_prob          = calculate_dropout_probability(row, profile)
            row["dropout_probability"] = round(dropout_prob, 4)
            row["dropped_out"]         = 1 if random.random() < dropout_prob else 0

            country_records.append(row)
            patient_counter += 1

        country_df  = pd.DataFrame(country_records)
        dropout_rate = country_df["dropped_out"].mean()
        log.info(f"    {country_name} dropout rate: {dropout_rate:.1%}")
        all_records.extend(country_records)

    return pd.DataFrame(all_records)


# ── SAVE ──────────────────────────────────────────────────────────────────────
def save_to_csv(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"CSV saved → {OUTPUT_CSV}  ({len(df):,} records)")


def save_to_sqlite(df: pd.DataFrame):
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gl_country   ON {TABLE_NAME}(country)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gl_region    ON {TABLE_NAME}(region)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gl_dropout   ON {TABLE_NAME}(dropped_out)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gl_diagnosis ON {TABLE_NAME}(diagnosis)")
    conn.commit()
    conn.close()
    log.info(f"SQLite saved → {DB_PATH}  (table: {TABLE_NAME})")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  GLOBAL SIMULATION SUMMARY")
    print("="*60)

    print("\n── Dropout Rate by Country ──")
    country_summary = df.groupby(["country", "region"]).agg(
        total_patients=("dropped_out", "count"),
        dropout_rate=("dropped_out", "mean"),
    ).round(3).sort_values("dropout_rate", ascending=False)
    country_summary["dropout_rate"] = (country_summary["dropout_rate"] * 100).round(1)
    print(country_summary.to_string())

    print("\n── Dropout Rate by Region ──")
    region_summary = df.groupby("region")["dropped_out"].mean().round(3) * 100
    print(region_summary.sort_values(ascending=False).to_string())

    print("\n── Dropout Rate by Income Level ──")
    income_summary = df.groupby("income_level")["dropped_out"].mean().round(3) * 100
    print(income_summary.sort_values(ascending=False).to_string())

    print("\n── Total Records ──")
    print(f"  {len(df):,} patients across {df['country'].nunique()} countries")
    print("="*60)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting global patient simulation...")
    df = simulate_global_patients()
    save_to_csv(df)
    save_to_sqlite(df)
    print_summary(df)
    log.info("Global simulation completed successfully.")