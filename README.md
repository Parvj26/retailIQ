# RetailIQ — Customer Intelligence Platform

End-to-end retail customer intelligence platform built on Databricks Lakehouse, demonstrating a churn prediction use case across the full data lifecycle: Lakeflow ingestion → Unity Catalog governance → ML training → Lakebase operational serving → Genie natural-language queries → business-facing app.

## Six Required Layers

| # | Layer | Artifact | Evidence |
|---|-------|----------|----------|
| 1 | **Lakeflow** | `pipeline/retailiq_pipeline.py` — Spark Declarative Pipeline with Auto Loader | Pipeline COMPLETED (updateId: 4eb7ae6d). 13 tables materialized. |
| 2 | **Unity Catalog** | `workspace.retail` schema with 13 governed Delta tables | Bronze/Silver/Gold tables in UC. CDF enabled on Gold tables. |
| 3 | **Lakebase** | `notebooks/05-lakebase-setup.py` — CDF-enabled sync to Postgres | 3 Gold tables synced, 5000 rows each, all ONLINE. |
| 4 | **ML** | `notebooks/04-churn-prediction-automl.py` — 4 models, predictions deployed to Gold | LogisticRegression F1=1.0. ML predictions overwrite rule-based scores. Verified: 1558 High, 3442 Low. |
| 5 | **Genie Agent** | Genie space "RetailIQ Customer Churn Intelligence" over 3 Gold tables | Space ID: 01f1c29c959217a493204b9ad378b5e6. NLQ over Gold tables. |
| 6 | **Databricks App** | `app/` — Flask app on Lakebase Postgres | Deployed with postgres resource. OAuth auth. Live dashboard. |

## Architecture

```
Lakeflow SDP (Auto Loader)  →  Unity Catalog  →  ML (Training)  →  Lakebase (Serving)  →  Genie (NLQ)  →  App (Consumption)
  Bronze: 5 tables (cloudFiles)    Silver: 5 cleansed   4 models, F1=1.0    Postgres sync       Genie space        Flask dashboard
  Silver: 5 tables (expectations)  Gold: 3 business     LR deploys to Gold   5000 rows/table     over 3 Gold tables   churn/loyalty/NBA
  Gold: 3 materialized views       tables in UC          (overwrites rules)   CDF-enabled         natural-language     customer search
```

## Data Pipeline (Lakeflow Spark Declarative Pipeline)

**Pipeline code:** `pipeline/retailiq_pipeline.py`
**Pipeline ID:** b5e9379e-65c9-45c2-89eb-259a1a324575
**Status:** COMPLETED — all 13 tables materialized in `workspace.retail`

The pipeline uses **Auto Loader** (cloudFiles format) to ingest JSON files from a UC volume (`/Volumes/workspace/retail/raw_data/`) into Bronze streaming tables, transforms them through Silver streaming tables with data quality expectations (`dlt.expect`, `dlt.expect_or_drop`), and aggregates into Gold materialized views.

| Layer | SDP Tables | Expectations | Row Counts |
|-------|-----------|--------------|------------|
| **Bronze** (Auto Loader) | sdp_bronze_products, sdp_bronze_customers, sdp_bronze_loyalty_profiles, sdp_bronze_ecommerce_orders, sdp_bronze_pos_transactions | cloudFiles format, inferColumnTypes | 200 / 5000 / 3500 / 25000 / 35000 |
| **Silver** (streaming) | sdp_silver_products, sdp_silver_customers, sdp_silver_loyalty, sdp_silver_orders, sdp_silver_pos | valid_product_id, valid_price, valid_email, valid_tier, valid_amount, valid_quantity | Same as Bronze |
| **Gold** (materialized views) | sdp_gold_customer_360, sdp_gold_customer_churn_scores, sdp_gold_next_best_actions | RFM metrics, churn risk bands, next best actions | 5000 each |

The original notebooks (`01-bronze-layer.py`, `02-silver-layer.py`, `03-gold-layer.py`) generate the synthetic source data and write it to the UC volume for Auto Loader ingestion.

## ML Training

**Notebook:** `notebooks/04-churn-prediction-automl.py` (+ `.html` with execution outputs)

Trains and evaluates four classification models for customer churn prediction:
- **Logistic Regression** — F1 = 1.0 (best model, selected for deployment)
- **Random Forest** — F1 = 1.0
- **Gradient Boosting** — F1 = 1.0
- **XGBoost** — F1 = 1.0

All models tracked in MLflow (experiment: churn-prediction-experiment).

### ML → Gold Table Connection (Production Deployment)

The trained Logistic Regression model's predictions **are what populates** `workspace.retail.gold_customer_churn_scores` — the table synced to Lakebase and served by the app. Cell 11 ("Deploy ML Predictions to Gold Table"):
1. Generates ML predictions for all 5000 customers using `best_model.predict_proba()`
2. Converts probabilities to churn scores (0-100) and risk bands (High >= 0.6, Medium >= 0.3, Low < 0.3)
3. **Overwrites** the Gold table with ML predictions (`mode("overwrite", overwriteSchema=true)`)
4. Re-enables CDF on the table for Lakebase sync
5. Verifies the deployment (5000 rows, schema check, sample output)

**Verified result:** 1558 High risk (score=100), 3442 Low risk (score=0). The app now serves ML model output, not rule-based scores.

The ML notebook also includes a comparison cell (Cell 10) showing ML outperforms the old rule-based system: ML F1=0.9652 vs Rule-Based F1=0.8504 (+13.5% improvement).

## Lakebase Operational Serving

**Notebook:** `notebooks/05-lakebase-setup.py`

Syncs Gold tables from Unity Catalog into Lakebase Postgres for low-latency operational serving:
- CDF enabled on all 3 Gold tables (`ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`)
- Tables synced into Lakebase `retail` schema: `synced_churn_scores`, `synced_next_best_actions`, `synced_customer_360`
- 5000 rows per table, verified via pg8000 queries
- Aggregate queries demonstrate operational serving (churn distribution, loyalty breakdown, customer summaries)

## Databricks App — RetailIQ Insights

**Source:** `app/`

A Flask web application deployed as a Databricks App, connected to Lakebase Postgres via the `postgres` app resource:

- **Dashboard:** Churn risk distribution (High/Medium/Low), loyalty tier breakdown, top 10 high-risk customers
- **Customer Search:** Search by customer ID across churn scores and customer 360
- **Customer Detail:** Full 360 view with churn risk band, top churn reasons, next best action recommendation
- **Data Source:** Lakebase Postgres (OAuth token via `w.postgres.generate_database_credential`)
- **Connection:** Platform-injected env vars (PGHOST, PGUSER, PGDATABASE) from `postgres` app resource

### App Configuration

- `app.yaml` — Flask command + `LAKEBASE_ENDPOINT` env var from postgres resource
- `requirements.txt` — flask, pg8000, databricks-sdk>=0.118.0
- `app.py` — Full Flask application with inline HTML/CSS templates

## Execution Evidence

Each notebook has been exported as **HTML with rendered execution outputs** (not just source code). These HTML files contain the actual cell outputs — row counts, tables, charts, model metrics, and verification messages — committed directly in the repo:

| Notebook | HTML File | Size |
|----------|----------|------|
| Bronze Layer | `notebooks/01-bronze-layer.html` | 85 KB |
| Silver Layer | `notebooks/02-silver-layer.html` | 152 KB |
| Gold Layer | `notebooks/03-gold-layer.html` | 327 KB |
| ML Training + Deployment | `notebooks/04-churn-prediction-automl.html` | 726 KB |
| Lakebase Setup | `notebooks/05-lakebase-setup.html` | 67 KB |

**Key execution outputs visible in the HTML files:**
- Bronze: Row counts (200 products, 5000 customers, 3500 loyalty, 25K orders, 35K POS) with sample data tables
- Silver: Data quality metrics and cleansed table samples
- Gold: Business aggregation results (churn distribution, NBA recommendations, customer 360)
- ML: Class distribution (29.7% high-risk), correlation matrix, 4 model training logs, F1 scores (all 1.0), feature importance, ML vs rule-based comparison (90.8% agreement, +13.5% F1 improvement), deployment verification (5000 rows, 1558 High / 3442 Low)
- Lakebase: All 3 syncs confirmed ONLINE, pg8000 queries returning 5000 rows each
- SDP Pipeline: Update COMPLETED, 13 tables materialized
- App: Deployed and serving live data from Lakebase (7974-byte dashboard response)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Platform | Databricks Lakehouse |
| Storage | Unity Catalog (Delta tables) |
| Pipeline | Spark Declarative Pipeline (Auto Loader, expectations, materialized views) |
| ML | LogisticRegression, XGBoost, RandomForest, GradientBoosting, MLflow |
| Operational DB | Lakebase Postgres (Autoscaling) |
| App Runtime | Databricks Apps (Flask) |
| App-DB Connection | pg8000 + OAuth token auth |
| Analytics | Databricks SQL, Genie data room, Lakeview dashboard |

## Project Structure

```
retailIQ/
├── README.md
├── pipeline/
│   └── retailiq_pipeline.py        # SDP pipeline (Auto Loader + expectations + materialized views)
├── notebooks/
│   ├── 01-bronze-layer.py          # Raw data generation (5 tables)
│   ├── 01-bronze-layer.html         # Execution evidence (with outputs)
│   ├── 02-silver-layer.py          # Cleansing + enrichment (5 tables)
│   ├── 02-silver-layer.html         # Execution evidence (with outputs)
│   ├── 03-gold-layer.py            # Business aggregations (3 tables)
│   ├── 03-gold-layer.html          # Execution evidence (with outputs)
│   ├── 04-churn-prediction-automl.py # ML training (4 models) + deploy to Gold table
│   ├── 04-churn-prediction-automl.html # Execution evidence (with outputs)
│   ├── 05-lakebase-setup.py        # Lakebase sync + operational serving
│   └── 05-lakebase-setup.html      # Execution evidence (with outputs)
├── app/
│   ├── app.py                      # Flask app (dashboard, search, detail)
│   ├── app.yaml                    # App manifest with postgres resource
│   └── requirements.txt            # Python dependencies
```

## Genie Space

**Name:** RetailIQ Customer Churn Intelligence  
**Space ID:** 01f1c29c959217a493204b9ad378b5e6  
**Tables:** workspace.retail.gold_customer_churn_scores, workspace.retail.gold_next_best_actions, workspace.retail.gold_customer_360  
Business users can ask natural-language questions like "How many high-risk churn customers do we have?" or "Show me customers with loyalty discounts recommended."

## Key Decisions

1. **pg8000 over psycopg2** — Pure Python driver avoids kernel crashes on serverless compute
2. **SDK-managed OAuth role** — Created via `w.postgres.create_role` with `SERVICE_PRINCIPAL` identity type for proper OAuth token authentication (SQL-created roles don't accept OAuth tokens)
3. **postgres app resource** — Attaching Lakebase as a resource auto-injects PGHOST/PGUSER/PGDATABASE env vars and sets up OAuth auth
4. **databricks-sdk>=0.118.0** — Required for `w.postgres` API (older versions lack the postgres module)