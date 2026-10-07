# RetailIQ — Customer Intelligence Platform

End-to-end retail customer intelligence demo built on Databricks Lakehouse, demonstrating operationalizing a churn prediction use case across the full data lifecycle: ingestion → medallion pipeline → ML training → Lakebase operational serving → business-facing app.

## Architecture

```
Bronze (Raw)     →  Silver (Cleansed)  →  Gold (Aggregated)   →  ML (Training)     →  Lakebase (Serving)  →  App (Consumption)
  5 source tables     5 enriched tables    3 business tables      3 models           Postgres sync         Flask dashboard
  synthetic data      validated+typed      churn_scores,           F1=1.0             retail schema         churn/loyalty/NBA
                                          next_best_actions,                          5000 rows each         customer search
                                          customer_360
```

## Data Pipeline (Medallion Architecture)

| Layer | Notebook | Tables | Description |
|-------|----------|--------|-------------|
| **Bronze** | `notebooks/01-bronze-layer.py` | bronze_products, bronze_customers, bronze_loyalty_profiles, bronze_ecommerce_orders, bronze_pos_transactions | Synthetic retail data generation: 200 products, 5000 customers, 3500 loyalty members, 25K ecommerce orders, 35K POS transactions |
| **Silver** | `notebooks/02-silver-layer.py` | silver_products, silver_customers, silver_loyalty, silver_orders, silver_pos | Data cleansing, type casting, deduplication, standardization across all Bronze sources |
| **Gold** | `notebooks/03-gold-layer.py` | gold_customer_churn_scores, gold_next_best_actions, gold_customer_360 | Business-ready aggregations: churn risk scoring (High/Medium/Low), next best action recommendations, unified customer 360 view |

## ML Training

**Notebook:** `notebooks/04-churn-prediction-automl.py`

Trains and evaluates three classification models for customer churn prediction:
- **XGBoost** — F1 = 1.0
- **Logistic Regression** — F1 = 1.0 (with imputation for NaN handling)
- **Random Forest** — F1 = 1.0

All models achieve perfect F1 due to a deterministic label derived from customer behavior patterns. Full training pipeline includes feature engineering, train/test split, model training, evaluation, and MLflow tracking.

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

All notebooks were executed in Databricks with visible outputs:
- Bronze: 5 tables created with row counts and sample data displayed
- Silver: 5 tables with data quality metrics
- Gold: 3 tables with business aggregations
- ML: All 3 models trained with F1 scores, confusion matrices, and MLflow tracking
- Lakebase: All syncs confirmed ONLINE, operational queries verified
- App: Deployed and serving live data from Lakebase (7974-byte dashboard response)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Platform | Databricks Lakehouse |
| Storage | Unity Catalog (Delta tables) |
| Pipeline | PySpark SQL (Medallion architecture) |
| ML | XGBoost, scikit-learn, MLflow |
| Operational DB | Lakebase Postgres (Autoscaling) |
| App Runtime | Databricks Apps (Flask) |
| App-DB Connection | pg8000 + OAuth token auth |
| Analytics | Databricks SQL, Genie data room, Lakeview dashboard |

## Project Structure

```
retailIQ/
├── README.md
├── notebooks/
│   ├── 01-bronze-layer.py          # Raw data ingestion (5 tables)
│   ├── 02-silver-layer.py          # Cleansing + enrichment (5 tables)
│   ├── 03-gold-layer.py            # Business aggregations (3 tables)
│   ├── 04-churn-prediction-automl.py # ML training (3 models)
│   └── 05-lakebase-setup.py        # Lakebase sync + operational serving
├── app/
│   ├── app.py                      # Flask app (dashboard, search, detail)
│   ├── app.yaml                    # App manifest with postgres resource
│   └── requirements.txt            # Python dependencies
```

## Key Decisions

1. **pg8000 over psycopg2** — Pure Python driver avoids kernel crashes on serverless compute
2. **SDK-managed OAuth role** — Created via `w.postgres.create_role` with `SERVICE_PRINCIPAL` identity type for proper OAuth token authentication (SQL-created roles don't accept OAuth tokens)
3. **postgres app resource** — Attaching Lakebase as a resource auto-injects PGHOST/PGUSER/PGDATABASE env vars and sets up OAuth auth
4. **databricks-sdk>=0.118.0** — Required for `w.postgres` API (older versions lack the postgres module)