"""
RetailIQ Customer Intelligence App
A Databricks App that surfaces customer churn insights and next best actions
to business users by querying Lakebase Postgres (operational serving layer).
"""

import os
import ssl
import traceback
import pg8000
from flask import Flask, request, jsonify, render_template_string
from databricks.sdk import WorkspaceClient

app = Flask(__name__)

# Lakebase configuration — env vars are auto-injected by the platform
# when the postgres resource is attached to the app
LAKEBASE_ENDPOINT = os.environ.get("LAKEBASE_ENDPOINT", "projects/retailiq/branches/production/endpoints/primary")
RETAIL_SCHEMA = "retail"

# Initialize Databricks SDK client (pre-authenticated in app runtime)
w = WorkspaceClient()

# ---------------------------------------------------------------------------
# Lakebase connection helper
# ---------------------------------------------------------------------------

def get_lakebase_connection():
    """Connect to Lakebase Postgres using platform-injected env vars + OAuth token."""
    host = os.environ["PGHOST"]
    user = os.environ["PGUSER"]
    port = int(os.environ.get("PGPORT", 5432))
    db = os.environ["PGDATABASE"]

    token = w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT).token

    ssl_ctx = ssl.create_default_context()
    conn = pg8000.connect(
        host=host, port=port, database=db,
        user=user, password=token, ssl_context=ssl_ctx, timeout=30,
    )
    return conn


def query(sql):
    """Execute a read-only query against Lakebase Postgres."""
    conn = get_lakebase_connection()
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return cols, rows


# ---------------------------------------------------------------------------
# HTML templates (inline for simplicity)
# ---------------------------------------------------------------------------

BASE_CSS = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #333; }
  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
            padding: 20px 40px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 24px; }
  .header .subtitle { font-size: 13px; opacity: 0.85; }
  .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }
  .card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .card h2 { font-size: 18px; margin-bottom: 16px; color: #4a5568; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  .metric { text-align: center; padding: 20px; border-radius: 10px; }
  .metric .value { font-size: 36px; font-weight: bold; }
  .metric .label { font-size: 13px; color: #718096; margin-top: 4px; text-transform: uppercase; }
  .metric.high { background: #fed7d7; } .metric.high .value { color: #c53030; }
  .metric.medium { background: #feebc8; } .metric.medium .value { color: #c05621; }
  .metric.low { background: #c6f6d5; } .metric.low .value { color: #276749; }
  .metric.info { background: #bee3f8; } .metric.info .value { color: #2b6cb0; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 10px 12px; border-bottom: 2px solid #e2e8f0;
       font-size: 12px; text-transform: uppercase; color: #718096; }
  td { padding: 10px 12px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
  tr:hover { background: #f7fafc; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
  .badge-high { background: #fed7d7; color: #c53030; }
  .badge-medium { background: #feebc8; color: #c05621; }
  .badge-low { background: #c6f6d5; color: #276749; }
  .search-bar { display: flex; gap: 10px; margin-bottom: 20px; }
  .search-bar input { flex: 1; padding: 10px 14px; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 14px; }
  .search-bar button { padding: 10px 24px; background: #667eea; color: white; border: none;
                       border-radius: 8px; cursor: pointer; font-size: 14px; }
  .search-bar button:hover { background: #5a67d8; }
  .bar-chart { margin: 10px 0; }
  .bar-row { display: flex; align-items: center; margin-bottom: 8px; }
  .bar-label { width: 80px; font-size: 13px; font-weight: 500; }
  .bar-track { flex: 1; background: #e2e8f0; border-radius: 6px; height: 24px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; display: flex; align-items: center;
              padding-left: 8px; color: white; font-size: 12px; font-weight: bold; }
  .bar-fill.high { background: #e53e3e; }
  .bar-fill.medium { background: #dd6b20; }
  .bar-fill.low { background: #38a169; }
  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .detail-item { padding: 12px; background: #f7fafc; border-radius: 8px; }
  .detail-item .key { font-size: 12px; text-transform: uppercase; color: #718096; }
  .detail-item .val { font-size: 16px; font-weight: 500; margin-top: 4px; }
  .back-link { display: inline-block; margin-bottom: 16px; color: #667eea; text-decoration: none; font-size: 14px; }
  .footer { text-align: center; padding: 20px; color: #a0aec0; font-size: 12px; }
</style>
"""

DASHBOARD_TEMPLATE = BASE_CSS + """
<div class="header">
  <div><h1>RetailIQ Customer Intelligence</h1><div class="subtitle">Churn prediction &amp; next best actions — powered by Lakebase operational serving</div></div>
</div>
<div class="container">
  <div class="card">
    <h2>Key Metrics</h2>
    <div class="metrics-grid">
      <div class="metric info"><div class="value">{{ total_customers }}</div><div class="label">Total Customers</div></div>
      <div class="metric high"><div class="value">{{ high_risk }}</div><div class="label">High Churn Risk</div></div>
      <div class="metric medium"><div class="value">{{ medium_risk }}</div><div class="label">Medium Risk</div></div>
      <div class="metric low"><div class="value">{{ low_risk }}</div><div class="label">Low Risk</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Churn Risk Distribution</h2>
    <div class="bar-chart">
      {% for band, count, pct, cls in churn_bars %}
      <div class="bar-row">
        <div class="bar-label">{{ band }}</div>
        <div class="bar-track"><div class="bar-fill {{ cls }}" style="width: {{ pct }}%">{{ count }} ({{ "%.1f"|format(pct) }}%)</div></div>
      </div>
      {% endfor %}
    </div>
  </div>
  <div class="card">
    <h2>Customer Lookup</h2>
    <form class="search-bar" action="/search" method="get">
      <input type="text" name="q" placeholder="Enter customer ID (e.g. CUST000001)" value="{{ query or '' }}">
      <button type="submit">Search</button>
    </form>
    {% if results %}
    <table>
      <tr><th>Customer ID</th><th>Name</th><th>Churn Risk</th><th>Score</th><th>Top Reason</th></tr>
      {% for r in results %}
      <tr>
        <td><a href="/customer/{{ r.customer_id }}">{{ r.customer_id }}</a></td>
        <td>{{ r.full_name or '' }}</td>
        <td><span class="badge badge-{{ (r.churn_risk_band or 'low')|lower }}">{{ r.churn_risk_band or 'N/A' }}</span></td>
        <td>{{ r.churn_score }}</td>
        <td>{{ r.top_reason_1 or '' }}</td>
      </tr>
      {% endfor %}
    </table>
    {% elif query %}
    <p style="color: #718096;">No customers found matching "{{ query }}".</p>
    {% endif %}
  </div>
  <div class="card">
    <h2>Loyalty Tier Distribution</h2>
    <table>
      <tr><th>Loyalty Tier</th><th>Customers</th><th>Percentage</th></tr>
      {% for tier, count, pct in loyalty_tiers %}
      <tr><td>{{ tier or 'Non-member' }}</td><td>{{ count }}</td><td>{{ "%.1f"|format(pct) }}%</td></tr>
      {% endfor %}
    </table>
  </div>
  <div class="card">
    <h2>High-Risk Customers — Immediate Action Needed</h2>
    <table>
      <tr><th>Customer ID</th><th>Churn Score</th><th>Top Reason</th><th>Recommended Action</th></tr>
      {% for r in high_risk_customers %}
      <tr>
        <td><a href="/customer/{{ r.customer_id }}">{{ r.customer_id }}</a></td>
        <td>{{ r.churn_score }}</td>
        <td>{{ r.top_reason_1 or '' }}</td>
        <td>{{ r.top_reason_2 or '' }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</div>
<div class="footer">RetailIQ Customer Intelligence — Data journey: Lakeflow → Unity Catalog → Lakebase → App</div>
"""

CUSTOMER_TEMPLATE = BASE_CSS + """
<div class="header">
  <div><h1>Customer Intelligence: {{ customer_id }}</h1><div class="subtitle">Churn risk &amp; next best actions from Lakebase operational serving</div></div>
</div>
<div class="container">
  <a href="/" class="back-link">&larr; Back to Dashboard</a>
  {% if error %}
  <div class="card"><p style="color: #c53030;">{{ error }}</p></div>
  {% else %}
  <div class="card">
    <h2>Churn Risk Assessment</h2>
    <div class="metrics-grid">
      <div class="metric {{ risk_class }}"><div class="value">{{ churn_score }}</div><div class="label">Churn Score</div></div>
      <div class="metric {{ risk_class }}"><div class="value">{{ churn_risk_band }}</div><div class="label">Risk Band</div></div>
    </div>
    <div class="detail-grid" style="margin-top:16px;">
      <div class="detail-item"><div class="key">Top Reason 1</div><div class="val">{{ top_reason_1 or 'N/A' }}</div></div>
      <div class="detail-item"><div class="key">Top Reason 2</div><div class="val">{{ top_reason_2 or 'N/A' }}</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Customer 360 Summary</h2>
    <div class="detail-grid">
      <div class="detail-item"><div class="key">Name</div><div class="val">{{ full_name or 'N/A' }}</div></div>
      <div class="detail-item"><div class="key">Loyalty Tier</div><div class="val">{{ loyalty_tier or 'Non-member' }}</div></div>
      <div class="detail-item"><div class="key">Total Revenue</div><div class="val">${{ "%.2f"|format(total_revenue or 0) }}</div></div>
      <div class="detail-item"><div class="key">Total Orders</div><div class="val">{{ total_orders or 0 }}</div></div>
      <div class="detail-item"><div class="key">Days Since Last Purchase</div><div class="val">{{ days_since_last_purchase or 'N/A' }}</div></div>
      <div class="detail-item"><div class="key">Preferred Channel</div><div class="val">{{ preferred_channel or 'N/A' }}</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Next Best Action</h2>
    <div class="detail-item" style="padding: 16px; background: #f7fafc; border-radius: 8px;">
      <div class="key">AI-Generated Customer Summary</div>
      <div class="val" style="margin-top: 8px; line-height: 1.6;">{{ customer_summary or 'N/A' }}</div>
    </div>
  </div>
  {% endif %}
</div>
<div class="footer">RetailIQ Customer Intelligence — Data journey: Lakeflow → Unity Catalog → Lakebase → App</div>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Main dashboard with aggregate metrics and high-risk customers."""
    try:
        # Churn risk distribution
        cols, rows = query(
            f"""SELECT churn_risk_band, COUNT(*) as count
                FROM {RETAIL_SCHEMA}.synced_churn_scores
                GROUP BY churn_risk_band"""
        )
        risk_data = {r[0]: r[1] for r in rows}
        total = sum(risk_data.values())
        high = risk_data.get("High", 0)
        medium = risk_data.get("Medium", 0)
        low = risk_data.get("Low", 0)
        churn_bars = [
            ("High", high, (high / total * 100) if total else 0, "high"),
            ("Medium", medium, (medium / total * 100) if total else 0, "medium"),
            ("Low", low, (low / total * 100) if total else 0, "low"),
        ]

        # Loyalty tier distribution
        cols2, rows2 = query(
            f"""SELECT COALESCE(loyalty_tier, 'Non-member') as tier, COUNT(*) as cnt
                FROM {RETAIL_SCHEMA}.synced_customer_360
                GROUP BY loyalty_tier
                ORDER BY cnt DESC"""
        )
        loyalty_tiers = [(r[0], r[1], (r[1] / total * 100) if total else 0) for r in rows2]

        # High-risk customers (top 10)
        cols3, rows3 = query(
            f"""SELECT cs.customer_id, cs.churn_score, cs.top_reason_1, cs.top_reason_2
                FROM {RETAIL_SCHEMA}.synced_churn_scores cs
                WHERE cs.churn_risk_band = 'High'
                ORDER BY cs.churn_score DESC
                LIMIT 10"""
        )
        high_risk_customers = [
            dict(zip(["customer_id", "churn_score", "top_reason_1", "top_reason_2"], r))
            for r in rows3
        ]

        # Search results
        search_query = request.args.get("q", "").strip()
        results = []
        if search_query:
            cols4, rows4 = query(
                f"""SELECT cs.customer_id, c360.full_name, cs.churn_risk_band, cs.churn_score, cs.top_reason_1
                    FROM {RETAIL_SCHEMA}.synced_churn_scores cs
                    LEFT JOIN {RETAIL_SCHEMA}.synced_customer_360 c360 ON cs.customer_id = c360.customer_id
                    WHERE cs.customer_id ILIKE '%{search_query}%'
                    LIMIT 20"""
            )
            results = [dict(zip(["customer_id", "full_name", "churn_risk_band", "churn_score", "top_reason_1"], r)) for r in rows4]

        return render_template_string(
            DASHBOARD_TEMPLATE,
            total_customers=total, high_risk=high, medium_risk=medium, low_risk=low,
            churn_bars=churn_bars, loyalty_tiers=loyalty_tiers,
            high_risk_customers=high_risk_customers, query=search_query, results=results,
        )
    except Exception as e:
        print(f"[DASHBOARD ERROR] {traceback.format_exc()}", flush=True)
        return f"""
        <html><body style="font-family:Arial;padding:40px;">
        <h2>RetailIQ App — Connection Error</h2>
        <p>Unable to connect to Lakebase Postgres: {e}</p>
        <p>Check app logs for details.</p>
        </body></html>
        """


@app.route("/customer/<customer_id>")
def customer_detail(customer_id):
    """Customer detail page showing churn risk, 360 view, and next best actions."""
    try:
        # Churn scores
        cols, rows = query(
            f"""SELECT customer_id, churn_score, churn_risk_band, top_reason_1, top_reason_2
                FROM {RETAIL_SCHEMA}.synced_churn_scores
                WHERE customer_id = '{customer_id}'"""
        )
        if not rows:
            return render_template_string(CUSTOMER_TEMPLATE, customer_id=customer_id, error="Customer not found.")
        churn = dict(zip(cols, rows[0]))

        # Customer 360
        cols2, rows2 = query(
            f"""SELECT full_name, loyalty_tier, total_revenue, total_orders,
                      days_since_last_purchase, preferred_channel
                FROM {RETAIL_SCHEMA}.synced_customer_360
                WHERE customer_id = '{customer_id}'"""
        )
        c360 = dict(zip(cols2, rows2[0])) if rows2 else {}

        # Next best actions
        cols3, rows3 = query(
            f"""SELECT customer_summary
                FROM {RETAIL_SCHEMA}.synced_next_best_actions
                WHERE customer_id = '{customer_id}'"""
        )
        nba = dict(zip(cols3, rows3[0])) if rows3 else {}

        risk_class = (churn.get("churn_risk_band") or "low").lower()

        return render_template_string(
            CUSTOMER_TEMPLATE,
            customer_id=customer_id,
            churn_score=churn.get("churn_score", "N/A"),
            churn_risk_band=churn.get("churn_risk_band", "N/A"),
            risk_class=risk_class,
            top_reason_1=churn.get("top_reason_1"),
            top_reason_2=churn.get("top_reason_2"),
            full_name=c360.get("full_name"),
            loyalty_tier=c360.get("loyalty_tier"),
            total_revenue=c360.get("total_revenue", 0),
            total_orders=c360.get("total_orders", 0),
            days_since_last_purchase=c360.get("days_since_last_purchase"),
            preferred_channel=c360.get("preferred_channel"),
            customer_summary=nba.get("customer_summary"),
        )
    except Exception as e:
        return render_template_string(CUSTOMER_TEMPLATE, customer_id=customer_id, error=str(e))


@app.route("/health")
def health():
    """Health check endpoint."""
    try:
        conn = get_lakebase_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({"status": "healthy", "lakebase": "connected"})
    except Exception as e:
        return jsonify({"status": "degraded", "error": str(e)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)