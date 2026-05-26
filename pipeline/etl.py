# pipeline/etl.py
# ETL Pipeline - Extract, Transform, Load
# Reads raw Kenya patient CSV → cleans → engineers features → loads to SQLite

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from datetime import datetime

# ── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
RAW_CSV    = "data/kenya_patients.csv"
DB_PATH    = "db/eye_health.db"
TABLE_RAW  = "kenya_patients"
TABLE_CLEAN = "kenya_patients_enriched"

# ── EXTRACT ──────────────────────────────────────────────────────────────────
def extract(path: str) -> pd.DataFrame:
    log.info(f"Extracting data from {path}")
    df = pd.read_csv(path)
    log.info(f"  Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


# ── VALIDATE ─────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Validating data...")

    initial = len(df)

    # Drop duplicates
    df = df.drop_duplicates(subset=["patient_id"])
    log.info(f"  Duplicates removed: {initial - len(df)}")

    # Drop nulls in critical columns
    critical = ["patient_id", "county", "age", "gender", "diagnosis", "dropped_out"]
    df = df.dropna(subset=critical)
    log.info(f"  Null rows removed: {initial - len(df)}")

    # Age range sanity check
    df = df[(df["age"] >= 1) & (df["age"] <= 100)]

    # Distance sanity check
    df = df[df["distance_to_facility_km"] >= 0]

    log.info(f"  Records after validation: {len(df):,}")
    return df


# ── TRANSFORM ────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming and engineering features...")

    # ── Age Groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 35, 50, 65, 100],
        labels=["Child (0-17)", "Young Adult (18-35)", "Adult (36-50)", "Senior (51-65)", "Elderly (65+)"]
    )

    # ── Distance Bands
    df["distance_band"] = pd.cut(
        df["distance_to_facility_km"],
        bins=[0, 10, 20, 35, 999],
        labels=["Near (<10km)", "Moderate (10-20km)", "Far (20-35km)", "Very Far (>35km)"]
    )

    # ── Risk Tier based on dropout_probability
    df["risk_tier"] = pd.cut(
        df["dropout_probability"],
        bins=[0, 0.30, 0.50, 0.70, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    )

    # ── Vulnerability Score (composite index 0-10)
    df["vulnerability_score"] = (
        (df["is_rural"] * 2.0) +
        (df["distance_to_facility_km"].clip(0, 50) / 50 * 2.5) +
        ((df["age"] >= 60).astype(int) * 1.5) +
        ((df["gender"] == "Female").astype(int) * 1.0) +
        ((df["sms_reminder_sent"] == 0).astype(int) * 1.5) +
        ((df["transport_support"] == 0).astype(int) * 1.5)
    ).round(2)

    # ── Intervention Flag (patients who need proactive outreach)
    df["needs_intervention"] = (
        (df["risk_tier"].isin(["High Risk", "Critical Risk"])) &
        (df["sms_reminder_sent"] == 0)
    ).astype(int)

    # ── Screening Month and Year
    df["screening_date"] = pd.to_datetime(df["screening_date"])
    df["screening_month"] = df["screening_date"].dt.month
    df["screening_year"]  = df["screening_date"].dt.year
    df["screening_quarter"] = df["screening_date"].dt.quarter

    # ── Days to dropout bucket
    df["dropout_timing"] = pd.cut(
        df["days_to_referral_dropout"],
        bins=[0, 7, 14, 30, 999],
        labels=["Week 1", "Week 2", "Month 1", "Beyond 1 Month"]
    )

    # ── Referral completion (inverse of dropout)
    df["completed_referral"] = 1 - df["dropped_out"]

    # ── Pipeline metadata
    df["etl_processed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df["data_source"]      = "Kenya VIP Simulation v1.0"
    df["country"]          = "Kenya"

    log.info(f"  New columns added: age_group, distance_band, risk_tier, vulnerability_score,")
    log.info(f"                     needs_intervention, screening_month/year/quarter,")
    log.info(f"                     dropout_timing, completed_referral")

    return df


# ── LOAD ─────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame, db_path: str, table: str):
    log.info(f"Loading enriched data to SQLite → table: {table}")

    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Convert categoricals to string for SQLite compatibility
    cat_cols = df.select_dtypes(include="category").columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    df.to_sql(table, conn, if_exists="replace", index=False)

    # Indexes for analytics queries
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_enr_county    ON {table}(county)",
        f"CREATE INDEX IF NOT EXISTS idx_enr_risk      ON {table}(risk_tier)",
        f"CREATE INDEX IF NOT EXISTS idx_enr_dropout   ON {table}(dropped_out)",
        f"CREATE INDEX IF NOT EXISTS idx_enr_diagnosis ON {table}(diagnosis)",
        f"CREATE INDEX IF NOT EXISTS idx_enr_year      ON {table}(screening_year)",
    ]
    for idx in indexes:
        conn.execute(idx)

    conn.commit()

    # Quick summary query
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    log.info(f"  Records loaded: {count:,}")

    conn.close()


# ── SUMMARY REPORT ────────────────────────────────────────────────────────────
def summary_report(db_path: str, table: str):
    conn = sqlite3.connect(db_path)

    print("\n" + "="*55)
    print("  ETL PIPELINE SUMMARY REPORT")
    print("="*55)

    queries = {
        "Dropout Rate by County": """
            SELECT county,
                   COUNT(*) as total_patients,
                   SUM(dropped_out) as dropouts,
                   ROUND(AVG(dropped_out)*100, 1) as dropout_pct
            FROM {t}
            GROUP BY county
            ORDER BY dropout_pct DESC
        """,
        "Risk Tier Distribution": """
            SELECT risk_tier,
                   COUNT(*) as patients,
                   ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM {t}), 1) as pct
            FROM {t}
            GROUP BY risk_tier
            ORDER BY patients DESC
        """,
        "Patients Needing Intervention": """
            SELECT county,
                   SUM(needs_intervention) as needs_outreach,
                   COUNT(*) as total
            FROM {t}
            GROUP BY county
            ORDER BY needs_outreach DESC
        """
    }

    for title, query in queries.items():
        print(f"\n── {title} ──")
        result = pd.read_sql_query(query.format(t=table), conn)
        print(result.to_string(index=False))

    conn.close()
    print("\n" + "="*55)


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting ETL pipeline...")

    df = extract(RAW_CSV)
    df = validate(df)
    df = transform(df)
    load(df, DB_PATH, TABLE_CLEAN)
    summary_report(DB_PATH, TABLE_CLEAN)

    log.info("ETL pipeline completed successfully.")