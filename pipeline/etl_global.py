# pipeline/etl_global.py
# Global ETL Pipeline - Extract, Transform, Load
# Reads raw global patient CSV → cleans → engineers features → loads to SQLite
# Extends Kenya ETL with country-level health system context

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
RAW_CSV      = "data/global_patients.csv"
DB_PATH      = "db/eye_health.db"
TABLE_RAW    = "global_patients"
TABLE_CLEAN  = "global_patients_enriched"

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

    df = df.drop_duplicates(subset=["patient_id"])
    critical = ["patient_id", "country", "region", "age", "gender",
                "diagnosis", "dropped_out"]
    df = df.dropna(subset=critical)
    df = df[(df["age"] >= 1) & (df["age"] <= 100)]
    df = df[df["distance_to_facility_km"] >= 0]

    log.info(f"  Records after validation: {len(df):,} (removed {initial - len(df)})")
    return df


# ── TRANSFORM ────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming and engineering features...")

    # ── Age Groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 35, 50, 65, 100],
        labels=["Child (0-17)", "Young Adult (18-35)",
                "Adult (36-50)", "Senior (51-65)", "Elderly (65+)"]
    )

    # ── Distance Bands
    df["distance_band"] = pd.cut(
        df["distance_to_facility_km"],
        bins=[0, 10, 20, 35, 999],
        labels=["Near (<10km)", "Moderate (10-20km)",
                "Far (20-35km)", "Very Far (>35km)"]
    )

    # ── Risk Tier
    df["risk_tier"] = pd.cut(
        df["dropout_probability"],
        bins=[0, 0.30, 0.50, 0.70, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    )

    # ── UHC Band (health system strength)
    df["uhc_band"] = pd.cut(
        df["uhc_index"],
        bins=[0, 40, 55, 70, 100],
        labels=["Fragile (<40)", "Developing (40-55)",
                "Moderate (55-70)", "Strong (>70)"]
    )

    # ── Vulnerability Score (global version includes UHC penalty)
    df["vulnerability_score"] = (
        (df["is_rural"] * 2.0) +
        (df["distance_to_facility_km"].clip(0, 60) / 60 * 2.5) +
        ((100 - df["uhc_index"]) / 100 * 2.0) +
        ((df["age"] >= 60).astype(int) * 1.5) +
        ((df["gender"] == "Female").astype(int) * 1.0) +
        ((df["sms_reminder_sent"] == 0).astype(int) * 1.5) +
        ((df["transport_support"] == 0).astype(int) * 1.5)
    ).round(2)

    # ── Intervention Flag
    df["needs_intervention"] = (
        (df["risk_tier"].isin(["High Risk", "Critical Risk"])) &
        (df["sms_reminder_sent"] == 0)
    ).astype(int)

    # ── Health System Penalty Score
    df["health_system_penalty"] = (
        (100 - df["uhc_index"]) / 100
    ).round(4)

    # ── Screening Date Features
    df["screening_date"]    = pd.to_datetime(df["screening_date"])
    df["screening_month"]   = df["screening_date"].dt.month
    df["screening_year"]    = df["screening_date"].dt.year
    df["screening_quarter"] = df["screening_date"].dt.quarter

    # ── Dropout Timing
    df["dropout_timing"] = pd.cut(
        df["days_to_referral_dropout"],
        bins=[0, 7, 14, 30, 999],
        labels=["Week 1", "Week 2", "Month 1", "Beyond 1 Month"]
    )

    # ── Referral Completion
    df["completed_referral"] = 1 - df["dropped_out"]

    # ── Global Comparison Flag
    df["is_high_burden_country"] = df["dropout_probability"].apply(
        lambda x: 1 if x >= 0.60 else 0
    )

    # ── Pipeline Metadata
    df["etl_processed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df["data_source"]      = "Global Eye Health Simulation v1.0"

    log.info(f"  Features engineered: {len(df.columns)} total columns")
    return df


# ── LOAD ─────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame, db_path: str, table: str):
    log.info(f"Loading enriched data → table: {table}")
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(db_path)

    cat_cols = df.select_dtypes(include="category").columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    df.to_sql(table, conn, if_exists="replace", index=False)

    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_country  ON {table}(country)",
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_region   ON {table}(region)",
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_risk     ON {table}(risk_tier)",
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_dropout  ON {table}(dropped_out)",
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_income   ON {table}(income_level)",
        f"CREATE INDEX IF NOT EXISTS idx_gl_enr_uhc      ON {table}(uhc_band)",
    ]
    for idx in indexes:
        conn.execute(idx)

    conn.commit()
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    log.info(f"  Records loaded: {cursor.fetchone()[0]:,}")
    conn.close()


# ── SUMMARY REPORT ────────────────────────────────────────────────────────────
def summary_report(db_path: str, table: str):
    conn = sqlite3.connect(db_path)

    print("\n" + "="*65)
    print("  GLOBAL ETL PIPELINE SUMMARY REPORT")
    print("="*65)

    queries = {
        "Dropout Rate by Country (with UHC Index)": """
            SELECT country, region, uhc_index,
                   COUNT(*) as total_patients,
                   SUM(dropped_out) as dropouts,
                   ROUND(AVG(dropped_out)*100, 1) as dropout_pct,
                   SUM(needs_intervention) as needs_outreach
            FROM {t}
            GROUP BY country
            ORDER BY dropout_pct DESC
        """,
        "Risk Tier Distribution by Region": """
            SELECT region, risk_tier,
                   COUNT(*) as patients,
                   ROUND(COUNT(*)*100.0/(
                       SELECT COUNT(*) FROM {t} t2
                       WHERE t2.region = {t}.region
                   ), 1) as pct_of_region
            FROM {t}
            GROUP BY region, risk_tier
            ORDER BY region, patients DESC
        """,
        "Total Intervention Need by Region": """
            SELECT region,
                   COUNT(*) as total_screened,
                   SUM(needs_intervention) as needs_outreach,
                   ROUND(SUM(needs_intervention)*100.0/COUNT(*), 1) as outreach_pct,
                   ROUND(AVG(vulnerability_score), 2) as avg_vulnerability
            FROM {t}
            GROUP BY region
            ORDER BY outreach_pct DESC
        """,
        "UHC Band vs Dropout Rate": """
            SELECT uhc_band,
                   COUNT(*) as patients,
                   ROUND(AVG(dropped_out)*100, 1) as dropout_pct,
                   ROUND(AVG(distance_to_facility_km), 1) as avg_distance_km
            FROM {t}
            GROUP BY uhc_band
            ORDER BY dropout_pct DESC
        """
    }

    for title, query in queries.items():
        print(f"\n── {title} ──")
        try:
            result = pd.read_sql_query(query.format(t=table), conn)
            print(result.to_string(index=False))
        except Exception as e:
            log.warning(f"Query failed: {e}")

    conn.close()
    print("\n" + "="*65)


# ── COMBINED VIEW ─────────────────────────────────────────────────────────────
def create_combined_view(db_path: str):
    log.info("Creating combined Kenya + Global view...")
    conn = sqlite3.connect(db_path)

    conn.execute("DROP VIEW IF EXISTS all_patients_combined")
    conn.execute("""
        CREATE VIEW all_patients_combined AS
        SELECT
            patient_id, country, region,
            age, gender, is_rural,
            distance_to_facility_km,
            diagnosis, referral_level, screener_type,
            sms_reminder_sent, transport_support,
            dropout_probability, dropped_out,
            risk_tier, vulnerability_score,
            needs_intervention, screening_year,
            'Kenya Programme' as programme_type
        FROM kenya_patients_enriched

        UNION ALL

        SELECT
            patient_id, country, region,
            age, gender, is_rural,
            distance_to_facility_km,
            diagnosis, referral_level, screener_type,
            sms_reminder_sent, transport_support,
            dropout_probability, dropped_out,
            risk_tier, vulnerability_score,
            needs_intervention, screening_year,
            'Global Programme' as programme_type
        FROM global_patients_enriched
    """)

    conn.commit()
    cursor = conn.execute("SELECT COUNT(*) FROM all_patients_combined")
    log.info(f"  Combined view created: {cursor.fetchone()[0]:,} total records")
    conn.close()


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting global ETL pipeline...")

    df = extract(RAW_CSV)
    df = validate(df)
    df = transform(df)
    load(df, DB_PATH, TABLE_CLEAN)
    summary_report(DB_PATH, TABLE_CLEAN)
    create_combined_view(DB_PATH)

    log.info("Global ETL pipeline completed successfully.")