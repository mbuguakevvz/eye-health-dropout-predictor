# 👁️ Eye Health Dropout Risk Predictor

> A data engineering pipeline that predicts referral dropout risk in 
> NGO-powered community eye health screening programs — built for Kenya 
> and scaled globally across 10 countries.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌍 The Problem

Over **1 billion people** live with unaddressed vision impairment globally.
The barrier is not medical technology — it is the gap between:

> *"You were screened and told to go to a clinic"*
> and
> *"You actually arrived and received treatment."*

NGOs like **Peek Vision**, **CBM**, **Fred Hollows Foundation**, and 
**Sightsavers** run community eye screening programs across Africa, Asia, 
and Latin America — but have no early warning system for which patients 
will drop out before completing their referral journey.

This project builds that early warning system.

---

## 🎯 What This Project Does

| Layer | What Was Built |
|---|---|
| **Data Simulation** | Synthetic patient records for Kenya (50K) + 10 countries (100K) |
| **ETL Pipeline** | Extract → Validate → Transform → Load to SQLite |
| **Feature Engineering** | Risk tiers, vulnerability scores, distance bands, age groups |
| **ML Model** | Random Forest dropout classifier · ROC-AUC 0.71 |
| **Risk Scoring** | Every patient scored and ranked by dropout probability |
| **Dashboard** | Interactive HTML visualization across all regions |

---

## 📊 Key Findings

### Global Dropout Rates by Country
| Country | Region | Dropout Rate | UHC Index |
|---|---|---|---|
| Papua New Guinea | Pacific | 90.5% | 32 |
| Ethiopia | Sub-Saharan Africa | 78.4% | 34 |
| Uganda | Sub-Saharan Africa | 67.6% | 37 |
| Myanmar | Southeast Asia | 62.4% | 42 |
| Tanzania | Sub-Saharan Africa | 59.7% | 39 |
| Cambodia | Southeast Asia | 52.6% | 44 |
| Bangladesh | South Asia | 46.6% | 48 |
| Bolivia | Latin America | 39.4% | 52 |
| India | South Asia | 37.2% | 56 |
| Peru | Latin America | 23.4% | 65 |

### Kenya County Breakdown (Vision Impact Project)
| County | Dropout Rate | Needs Outreach |
|---|---|---|
| Bomet | 49.8% | 2,039 |
| Vihiga | 46.4% | 1,782 |
| Kisii | 40.9% | 1,418 |
| Nyeri | 36.7% | 1,052 |
| Nakuru | 34.8% | 773 |
| Mombasa | 25.9% | 229 |
| Nairobi | 24.4% | 174 |

### Top Dropout Predictors (Feature Importance)
1. **Vulnerability Score** — 27.08%
2. **Distance to Facility** — 15.19%
3. **Referral Level** — 8.85%
4. **Age** — 8.76%
5. **Diagnosis Type** — 6.40%

### The Policy Insight
| Health System (UHC) | Dropout Rate |
|---|---|
| Fragile (<40) | 74.0% |
| Developing (40-55) | 50.2% |
| Moderate (55-70) | 30.3% |

> Every 10 points of UHC index improvement = ~20% fewer dropouts.

---

## 🗂️ Project Structure---

## ⚙️ How to Run

### 1. Clone and setup
```bash
git clone https://github.com/mbuguakevvz/eye-health-dropout-predictor.git
cd eye-health-dropout-predictor
python -m venv venv
venv\Scripts\Activate        # Windows
pip install -r requirements.txt
```

### 2. Generate data
```bash
python data/simulate_kenya.py
python data/simulate_global.py
```

### 3. Run ETL pipelines
```bash
python pipeline/etl.py
python pipeline/etl_global.py
```

### 4. Train and score model
```bash
python models/dropout_risk_scorer.py
```

### 5. Open dashboard
```bash
# Open dashboard/index.html in your browser
```

---

## 🗄️ Database Tables

| Table | Records | Description |
|---|---|---|
| `kenya_patients` | 50,000 | Raw Kenya simulation |
| `kenya_patients_enriched` | 50,000 | Kenya ETL output with features |
| `global_patients` | 100,000 | Raw global simulation |
| `global_patients_enriched` | 100,000 | Global ETL output with features |
| `kenya_risk_scores` | 50,000 | Model risk scores for Kenya |
| `all_patients_combined` | 150,000 | Combined view — Kenya + Global |

---

## 🏥 Real-World Inspiration

This project is modeled on documented NGO programs:

- **Peek Vision + CBM** — Vision Impact Project, Kenya (7 counties, 5M+ screened)
- **Fred Hollows Foundation** — West Pokot County, Kenya
- **Sightsavers** — Kenya pilot programme
- **Aravind Eye Care / LAICO** — India community programs
- **Brien Holden Vision Institute** — Latin America programs

---

## 🛠️ Tech Stack

- **Python 3.10** — simulation, ETL, modeling
- **pandas / numpy** — data transformation
- **scikit-learn** — Random Forest classifier
- **SQLite** — lightweight analytical database
- **Faker** — realistic synthetic data generation
- **Chart.js** — dashboard visualizations
- **HTML / CSS / JS** — dashboard frontend

---

## 👤 Author

**Kevin Mbugua** · [@mbuguakevvz](https://github.com/mbuguakevvz)

Data Engineer · Nairobi, Kenya

---

## 📄 License

MIT License — free to use, adapt, and build on.

---

*Built as a data engineering portfolio project. Synthetic data only —
no real patient records used. Inspired by the humanitarian work of
global eye health NGOs working to end preventable blindness.*