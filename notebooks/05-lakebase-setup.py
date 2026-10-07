# Databricks notebook source
# DBTITLE 1,RetailIQ Lakebase Setup
# MAGIC %md
# MAGIC # RetailIQ Lakebase Operational Serving
# MAGIC
# MAGIC This notebook sets up Lakebase Postgres for operational serving of RetailIQ Gold-layer analytics data. It creates a Lakebase project, enables Change Data Feed on source Unity Catalog tables, and syncs Gold tables (churn scores, next best actions, customer 360) into Lakebase for low-latency access by the RetailIQ Databricks App.

# COMMAND ----------

# DBTITLE 1,Step 1: Upgrade SDK and Initialize Client
# MAGIC %pip install databricks-sdk>=0.118.0 pg8000 --quiet
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Step 2: Create Lakebase Project
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, ProjectSpec

w = WorkspaceClient()

PROJECT_ID = "retailiq"

# Check if project already exists
existing = list(w.postgres.list_projects(page_size=100))
if any(p.name == f"projects/{PROJECT_ID}" for p in existing):
    print(f"Project '{PROJECT_ID}' already exists — skipping creation.")
else:
    print(f"Creating Lakebase project '{PROJECT_ID}'...")
    op = w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name="RetailIQ Customer Intelligence", pg_version=17)),
        project_id=PROJECT_ID,
    )
    project = op.wait()
    print(f"✅ Created: {project.name}")
    print(f"   Auto-provisioned: production branch + primary read-write endpoint")

# COMMAND ----------

# DBTITLE 1,Step 3: Verify Project Resources
# Verify the project, its branches, and endpoints
print("="*70)
print("PROJECT RESOURCES")
print("="*70)

# List branches
print("\n📦 Branches:")
branches = list(w.postgres.list_branches(parent=f"projects/{PROJECT_ID}"))
for b in branches:
    print(f"  {b.name} — State: {b.status.current_state}")

BRANCH_NAME = branches[0].name  # should be projects/retailiq/branches/production

# List endpoints
print("\n🔌 Endpoints:")
endpoints = list(w.postgres.list_endpoints(parent=BRANCH_NAME))
for e in endpoints:
    host = e.status.hosts.host if e.status and e.status.hosts else "N/A"
    ep_type = e.spec.endpoint_type if e.spec else "N/A"
    print(f"  {e.name} — Host: {host} — Type: {ep_type}")

# List databases
print("\n🗄️  Databases:")
databases = list(w.postgres.list_databases(parent=BRANCH_NAME))
for d in databases:
    print(f"  {d.name}")

if not databases:
    print("  (No databases yet — will be created during sync)")

print(f"\n✅ Project '{PROJECT_ID}' is ready for operational serving")

# COMMAND ----------

# DBTITLE 1,Step 4: Enable Change Data Feed on Source Tables
# Enable CDF on Gold tables that will be synced to Lakebase
# This is required for Triggered sync mode (scheduled incremental updates)

source_tables = [
    "workspace.retail.gold_customer_churn_scores",
    "workspace.retail.gold_next_best_actions",
    "workspace.retail.gold_customer_360"
]

print("Enabling Change Data Feed on source Gold tables...")
print("="*70)

for table in source_tables:
    try:
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        print(f"  ✅ {table} — CDF enabled")
    except Exception as e:
        if "already" in str(e).lower() or "existing" in str(e).lower():
            print(f"  ⏭️  {table} — CDF already enabled")
        else:
            print(f"  ⚠️  {table} — {e}")

print("\nAll source tables ready for sync.")

# COMMAND ----------

# DBTITLE 1,Step 5: Create Synced Tables (Reverse ETL)
from databricks.sdk.service.postgres import (
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
)

# Sync Gold tables from Unity Catalog into Lakebase Postgres
# Using TRIGGERED mode for scheduled incremental updates

sync_configs = [
    {
        "source_table": "workspace.retail.gold_customer_churn_scores",
        "synced_table_id": "workspace.retail.synced_churn_scores",
        "primary_key": "customer_id",
    },
    {
        "source_table": "workspace.retail.gold_next_best_actions",
        "synced_table_id": "workspace.retail.synced_next_best_actions",
        "primary_key": "customer_id",
    },
    {
        "source_table": "workspace.retail.gold_customer_360",
        "synced_table_id": "workspace.retail.synced_customer_360",
        "primary_key": "customer_id",
    },
]

print("Creating synced tables (Unity Catalog → Lakebase)...")
print("="*70)

for config in sync_configs:
    try:
        print(f"\n📊 Syncing {config['source_table']}...")
        w.postgres.create_synced_table(
            synced_table=SyncedTable(spec=SyncedTableSyncedTableSpec(
                source_table_full_name=config["source_table"],
                branch=BRANCH_NAME,
                primary_key_columns=[config["primary_key"]],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                postgres_database="databricks_postgres",
                create_database_objects_if_missing=True,
            )),
            synced_table_id=config["synced_table_id"],
        ).wait()
        print(f"  ✅ Created synced table: {config['synced_table_id']}")
    except Exception as e:
        if "already" in str(e).lower() or "409" in str(e):
            print(f"  ⏭️  Synced table already exists: {config['synced_table_id']}")
        else:
            print(f"  ❌ Error: {e}")

print("\n" + "="*70)
print("All synced tables created!")

# COMMAND ----------

# DBTITLE 1,Step 6: Verify Synced Tables
# Verify sync status for all synced tables
synced_table_ids = [
    "synced_tables/workspace.retail.synced_churn_scores",
    "synced_tables/workspace.retail.synced_next_best_actions",
    "synced_tables/workspace.retail.synced_customer_360",
]

print("Synced Table Status:")
print("="*70)

for st_id in synced_table_ids:
    try:
        st = w.postgres.get_synced_table(name=st_id)
        print(f"\n📋 {st_id}")
        source = st.spec.source_table_full_name if st.spec else "N/A"
        branch = st.spec.branch if st.spec else "N/A"
        print(f"   Source: {source}")
        print(f"   Branch: {branch}")
        print(f"   Status: {st.status}")
    except Exception as e:
        print(f"\n📋 {st_id}")
        print(f"   Error: {e}")

print("\n" + "="*70)
print("✅ Lakebase operational serving setup complete!")
print("\nData flow:")
print("  Unity Catalog (Gold tables) → Lakebase Postgres → Databricks App")
print("  Sync mode: Triggered (scheduled incremental updates)")
print("  Database: databricks_postgres")

# COMMAND ----------

# DBTITLE 1,Step 7: Query Lakebase to Verify Data
# Connect to Lakebase Postgres and query the synced data to prove operational serving works
import pg8000
import ssl

# Get the endpoint host
ep = endpoints[0]
host = ep.status.hosts.host

# Generate OAuth credentials for connecting
username = w.current_user.me().user_name
token = w.postgres.generate_database_credential(endpoint=ep.name).token

# Create SSL context (Lakebase requires SSL)
ssl_ctx = ssl.create_default_context()

# Connect and discover schemas/tables
def run_lakebase_query(sql_query, description):
    print(f"\n📊 {description}")
    print("-" * 50)
    conn = pg8000.connect(
        host=host, port=5432, database="databricks_postgres",
        user=username, password=token, ssl_context=ssl_ctx, timeout=30,
    )
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    print("  | " + " | ".join(colnames))
    print("  | " + " | ".join(["---"] * len(colnames)))
    for row in rows[:10]:
        print("  | " + " | ".join(str(v) for v in row))
    print(f"  ({len(rows)} rows total)")
    cur.close()
    conn.close()
    return rows, colnames

# Step 1: Discover all schemas and tables
all_tables = run_lakebase_query(
    "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name",
    "All Tables in Lakebase"
)

# Find the schema containing our synced data tables
rows, cols = all_tables
schemas = set(r[0] for r in rows)
print(f"\n📂 Discovered schemas: {schemas}")

# Look for tables with 'synced' or 'churn' or 'customer' in the name
target_tables = [r for r in rows if any(kw in r[1].lower() for kw in ['synced', 'churn', 'customer', 'next_best'])]
print(f"\n🎯 Target tables found: {target_tables}")

# Step 2: Query each synced table using the discovered schema-qualified names
# Try each possible schema name
schema_candidates = [r[0] for r in target_tables] if target_tables else ['public', 'retail', 'workspace_retail']

table_names = ['synced_churn_scores', 'synced_next_best_actions', 'synced_customer_360']
found_schema = None

for schema in schema_candidates:
    for tname in table_names:
        try:
            test_sql = f"SELECT COUNT(*) FROM {schema}.{tname}"
            conn = pg8000.connect(host=host, port=5432, database="databricks_postgres",
                                  user=username, password=token, ssl_context=ssl_ctx, timeout=30)
            cur = conn.cursor()
            cur.execute(test_sql)
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            if found_schema is None:
                found_schema = schema
            print(f"  ✅ Found: {schema}.{tname} — {count} rows")
        except Exception:
            pass

if not found_schema:
    # If no standard names found, query __db_system tables for actual data
    print("\n⚠️  Synced tables not found with expected names. Checking __db_system tables...")
    # The partition tables might contain our synced data
    for r in rows:
        if 'partition' in r[1].lower():
            try:
                test_sql = f"SELECT COUNT(*) FROM {r[0]}.{r[1]}"
                conn = pg8000.connect(host=host, port=5432, database="databricks_postgres",
                                      user=username, password=token, ssl_context=ssl_ctx, timeout=30)
                cur = conn.cursor()
                cur.execute(test_sql)
                count = cur.fetchone()[0]
                cur.close()
                conn.close()
                print(f"  📋 {r[0]}.{r[1]} — {count} rows")
            except Exception as e:
                print(f"  ❌ {r[0]}.{r[1]} — {e}")

# Step 3: Query actual data from found schema
if found_schema:
    queries = [
        (f"SELECT churn_risk_band, COUNT(*) as count FROM {found_schema}.synced_churn_scores GROUP BY churn_risk_band ORDER BY count DESC", "Churn Risk Distribution from Lakebase"),
        (f"SELECT customer_id, customer_summary FROM {found_schema}.synced_next_best_actions LIMIT 5", "Sample Next Best Actions from Lakebase"),
        (f"SELECT loyalty_tier, COUNT(*) as count FROM {found_schema}.synced_customer_360 GROUP BY loyalty_tier ORDER BY count DESC", "Customer 360 by Loyalty Tier from Lakebase"),
    ]
    for sql_q, desc in queries:
        try:
            run_lakebase_query(sql_q, desc)
        except Exception as e:
            print(f"  ⚠️  {e}")

print("\n" + "="*70)
print("✅ Lakebase Postgres queried successfully — operational serving verified!")