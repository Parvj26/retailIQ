# Databricks notebook source
# DBTITLE 1,Gold Layer 1: Customer 360
# =============================================================================
# GOLD LAYER 1: CUSTOMER 360 VIEW
# =============================================================================
# Business Purpose:
#   Create a unified, trusted view of each customer by combining:
#   - Customer profile data (demographics, location)
#   - Loyalty program status (tier, points, membership)
#   - Transaction behavior (revenue, frequency, recency)
#   - Channel preferences (Ecommerce vs In-Store)
#   - Product preferences (favorite categories)
#
# This table powers:
#   - Customer segmentation and personalization
#   - Marketing campaign targeting
#   - Executive dashboards
#   - Downstream ML models (churn prediction)
# =============================================================================

from pyspark.sql.functions import (
    col, count, sum as _sum, avg, min as _min, max as _max, when, datediff, current_date, lit, to_date, coalesce, first, mode
)
from pyspark.sql.window import Window

# Load Silver layer tables
# These tables are already cleaned, validated, and deduplicated
customer_df = spark.table("workspace.retail.silver_customers")
loyalty_df = spark.table("workspace.retail.silver_loyalty_profiles")
trans_df = spark.table("workspace.retail.silver_transactions_unified")
products_df = spark.table("workspace.retail.silver_products")

# -----------------------------------------------------------------------------
# STEP 1: Calculate Customer Preferences (Channel & Category)
# -----------------------------------------------------------------------------
# Business Logic:
#   - Preferred Channel: The channel (Ecommerce/In-Store) where customer shops most
#   - Favorite Category: The product category customer purchases most frequently
#   - Use mode() to find the most common value for each customer

trans_with_category = trans_df.join(products_df, "product_id", "left")

channel_category_prefs = (
    trans_with_category
    .groupBy("customer_id")
    .agg(
        mode(col("channel")).alias("preferred_channel"),      # Most frequent shopping channel
        mode(col("category")).alias("favorite_category")      # Most purchased category
    )
)

# -----------------------------------------------------------------------------
# STEP 2: Calculate Transaction Metrics Per Customer
# -----------------------------------------------------------------------------
# Business Logic:
#   - Total Transactions: All transactions (completed + pending + cancelled)
#   - Completed Transactions: Only successful purchases
#   - Revenue Metrics: Total, by channel (Ecommerce vs In-Store)
#   - Average Order Value (AOV): Mean transaction amount
#   - Recency: Days since last purchase (key churn indicator)
#   - Refund/Return Count: Quality and satisfaction indicator

trans_metrics = (
    trans_df.groupBy("customer_id")
    .agg(
        # Transaction counts
        count("*").alias("total_transactions"),
        _sum(when(col("is_completed_transaction") == True, 1).otherwise(0)).alias("completed_transactions"),
        
        # Revenue metrics
        _sum(col("gross_amount")).alias("total_revenue"),
        _sum(when(col("channel") == "Ecommerce", col("gross_amount")).otherwise(0)).alias("ecommerce_revenue"),
        _sum(when(col("channel") == "In-Store", col("gross_amount")).otherwise(0)).alias("instore_revenue"),
        avg(col("gross_amount")).alias("avg_order_value"),
        
        # Recency metrics (critical for churn prediction)
        _min(to_date(col("transaction_timestamp"))).alias("first_purchase_date"),
        _max(to_date(col("transaction_timestamp"))).alias("last_purchase_date"),
        
        # Quality/satisfaction indicator
        _sum(when(col("is_refund_or_return") == True, 1).otherwise(0)).alias("refund_return_count")
    )
    .withColumn("days_since_last_purchase", datediff(current_date(), col("last_purchase_date")))
    .withColumn("customer_lifetime_value_proxy", col("total_revenue"))  # Simple CLV = total revenue
)

# -----------------------------------------------------------------------------
# STEP 3: Join All Dimensions to Create Customer 360
# -----------------------------------------------------------------------------
# Business Logic:
#   - Start with all customers from customer master
#   - LEFT JOIN loyalty (some customers may not be loyalty members)
#   - LEFT JOIN transaction metrics (some customers may have no transactions yet)
#   - LEFT JOIN preferences (derived from transactions)
#   - Use COALESCE to provide sensible defaults for NULL values

customer_360 = (
    customer_df.join(loyalty_df, "customer_id", "left")
    .join(trans_metrics, "customer_id", "left")
    .join(channel_category_prefs, "customer_id", "left")
    .select(
        # Customer identity
        "customer_id", "full_name", "email", "city", "state",
        
        # Loyalty program status
        "loyalty_tier", "points_balance", "is_loyalty_member",
        
        # Transaction metrics (with defaults for new customers)
        coalesce(col("total_transactions"), lit(0)).alias("total_transactions"),
        coalesce(col("completed_transactions"), lit(0)).alias("completed_transactions"),
        coalesce(col("total_revenue"), lit(0.0)).alias("total_revenue"),
        coalesce(col("ecommerce_revenue"), lit(0.0)).alias("ecommerce_revenue"),
        coalesce(col("instore_revenue"), lit(0.0)).alias("instore_revenue"),
        coalesce(col("avg_order_value"), lit(0.0)).alias("avg_order_value"),
        
        # Recency indicators
        "first_purchase_date", "last_purchase_date",
        coalesce(col("days_since_last_purchase"), lit(9999)).alias("days_since_last_purchase"),  # 9999 = never purchased
        
        # Preferences
        "favorite_category", "preferred_channel",
        
        # Quality indicators
        coalesce(col("refund_return_count"), lit(0)).alias("refund_return_count"),
        coalesce(col("customer_lifetime_value_proxy"), lit(0.0)).alias("customer_lifetime_value_proxy")
    )
)

# Save as Delta table in Gold layer
customer_360.write.format("delta").mode("overwrite").saveAsTable("workspace.retail.gold_customer_360")

print("✅ Gold Customer 360 table created successfully")

# -----------------------------------------------------------------------------
# VALIDATION & DATA QUALITY CHECKS
# -----------------------------------------------------------------------------

# Check 1: Row count
display(spark.sql("""
SELECT COUNT(*) AS row_count FROM workspace.retail.gold_customer_360
"""))

# Check 2: Table schema
display(spark.sql("""
DESCRIBE TABLE workspace.retail.gold_customer_360
"""))

# Check 3: Sample records
display(spark.sql("""
SELECT * FROM workspace.retail.gold_customer_360 LIMIT 10
"""))

# Check 4: NULL checks for critical fields
display(spark.sql("""
SELECT 
  COUNT(*) as total_customers,
  SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) as null_customer_ids,
  SUM(CASE WHEN total_revenue IS NULL THEN 1 ELSE 0 END) as null_revenues
FROM workspace.retail.gold_customer_360
"""))

# COMMAND ----------

# DBTITLE 1,Gold Layer 2: Customer Churn Features
# =============================================================================
# GOLD LAYER 2: CUSTOMER CHURN FEATURES
# =============================================================================
# Business Purpose:
#   Create model-ready features for churn prediction ML models.
#   Churn = customers who are likely to stop purchasing from RetailIQ.
#
# Key Predictive Signals:
#   - RECENCY: How long since last purchase? (RFM model)
#   - FREQUENCY: How often do they shop? Recent activity trends?
#   - MONETARY: How much do they spend? Revenue trajectory?
#   - LOYALTY: What's their loyalty tier and tenure?
#   - QUALITY: Do they have high refund/return rates?
#
# This table feeds:
#   - Churn prediction ML models
#   - Rule-based churn scoring (next cell)
#   - Proactive retention campaigns
# =============================================================================

from pyspark.sql.functions import col, when, datediff, current_date, coalesce, lit, to_date

# Load the Customer 360 table and raw transactions
customer_360_df = spark.table("workspace.retail.gold_customer_360")
trans_df = spark.table("workspace.retail.silver_transactions_unified")

# -----------------------------------------------------------------------------
# STEP 1: Calculate Time-Based Transaction Metrics
# -----------------------------------------------------------------------------
# Business Logic:
#   - Look at RECENT behavior (last 30/90 days) vs lifetime behavior
#   - Declining recent activity is a strong churn signal
#   - Example: Customer with 50 lifetime transactions but 0 in last 90 days = HIGH RISK

trans_recent = (
    trans_df
    .withColumn("transaction_date", to_date(col("transaction_timestamp")))
    .withColumn("days_ago", datediff(current_date(), col("transaction_date")))  # How many days ago was this transaction?
    .groupBy("customer_id")
    .agg(
        # Last 30 days activity (very recent)
        _sum(when(col("days_ago") <= 30, 1).otherwise(0)).alias("transactions_last_30d"),
        _sum(when(col("days_ago") <= 30, col("gross_amount")).otherwise(0)).alias("revenue_last_30d"),
        
        # Last 90 days activity (recent quarter)
        _sum(when(col("days_ago") <= 90, 1).otherwise(0)).alias("transactions_last_90d"),
        _sum(when(col("days_ago") <= 90, col("gross_amount")).otherwise(0)).alias("revenue_last_90d")
    )
)

# -----------------------------------------------------------------------------
# STEP 2: Get Customer Tenure
# -----------------------------------------------------------------------------
# Business Logic:
#   - Customer tenure = days since registration
#   - Newer customers churn at higher rates (haven't built loyalty yet)
#   - Longer tenure customers are more valuable to retain

customer_df = spark.table("workspace.retail.silver_customers")
customer_tenure = (
    customer_df
    .select("customer_id", "customer_tenure_days")
)

# -----------------------------------------------------------------------------
# STEP 3: Build Complete Feature Set
# -----------------------------------------------------------------------------
# Business Logic:
#   Combine all features that predict churn:
#   1. Loyalty indicators (tier, points)
#   2. Recency (days since last purchase)
#   3. Frequency (transaction counts in different time windows)
#   4. Monetary (revenue in different time windows, AOV)
#   5. Tenure (how long they've been a customer)
#   6. Quality (refund/return rate)
#   7. Preferences (channel, category)

churn_features = (
    customer_360_df
    .select(
        "customer_id",
        "loyalty_tier",              # Bronze/Silver/Gold/Platinum (or NULL)
        "points_balance",            # Accumulated loyalty points
        "days_since_last_purchase",  # CRITICAL: Recency metric
        "avg_order_value",           # Monetary value per transaction
        "preferred_channel",         # Ecommerce vs In-Store
        "favorite_category",         # Most purchased category
        "total_transactions",        # Lifetime transaction count
        "refund_return_count"        # Quality/satisfaction metric
    )
    .join(customer_tenure, "customer_id", "left")
    .join(trans_recent, "customer_id", "left")
    
    # Handle NULLs: New customers or inactive customers may have 0 recent activity
    .withColumn("transactions_last_30d", coalesce(col("transactions_last_30d"), lit(0)))
    .withColumn("transactions_last_90d", coalesce(col("transactions_last_90d"), lit(0)))
    .withColumn("revenue_last_30d", coalesce(col("revenue_last_30d"), lit(0.0)))
    .withColumn("revenue_last_90d", coalesce(col("revenue_last_90d"), lit(0.0)))
    
    # Calculate refund/return RATE (percentage of transactions)
    .withColumn(
        "refund_return_rate",
        when(col("total_transactions") > 0, col("refund_return_count") / col("total_transactions")).otherwise(0.0)
    )
    
    # -----------------------------------------------------------------------------
    # CHURN LABEL (Proxy for ML Training)
    # -----------------------------------------------------------------------------
    # Business Rule:
    #   Label customer as "churned" if EITHER:
    #   1. Days since last purchase > 90 days (inactive for a quarter)
    #   2. Zero transactions in last 90 days (same outcome, different path)
    #
    # In production ML:
    #   - This would be replaced with historical labels (e.g., "churned in next 30 days")
    #   - Model would learn which features predict churn BEFORE it happens
    # -----------------------------------------------------------------------------
    .withColumn(
        "churn_label_proxy",
        when((col("days_since_last_purchase") > 90) | (col("transactions_last_90d") == 0), 1).otherwise(0)
    )
    
    .select(
        "customer_id",
        "loyalty_tier",
        "points_balance",
        coalesce(col("customer_tenure_days"), lit(0)).alias("customer_tenure_days"),
        "days_since_last_purchase",
        "transactions_last_30d",
        "transactions_last_90d",
        "revenue_last_30d",
        "revenue_last_90d",
        "avg_order_value",
        "preferred_channel",
        "refund_return_rate",
        "favorite_category",
        "churn_label_proxy"
    )
)

# Save as Delta table
churn_features.write.format("delta").mode("overwrite").saveAsTable("workspace.retail.gold_customer_churn_features")

print("✅ Gold Customer Churn Features table created successfully")

# -----------------------------------------------------------------------------
# VALIDATION & CHURN DISTRIBUTION ANALYSIS
# -----------------------------------------------------------------------------

# Check 1: Row count
display(spark.sql("""
SELECT COUNT(*) AS row_count FROM workspace.retail.gold_customer_churn_features
"""))

# Check 2: Schema
display(spark.sql("""
DESCRIBE TABLE workspace.retail.gold_customer_churn_features
"""))

# Check 3: Sample features
display(spark.sql("""
SELECT * FROM workspace.retail.gold_customer_churn_features LIMIT 10
"""))

# Check 4: Churn distribution (how many customers are at risk?)
display(spark.sql("""
SELECT 
  churn_label_proxy,
  COUNT(*) as customer_count,
  AVG(days_since_last_purchase) as avg_days_since_last_purchase,
  AVG(transactions_last_90d) as avg_transactions_90d
FROM workspace.retail.gold_customer_churn_features
GROUP BY churn_label_proxy
"""))

# COMMAND ----------

# DBTITLE 1,Gold Layer 3: Customer Churn Scores
# =============================================================================
# GOLD LAYER 3: CUSTOMER CHURN SCORES
# =============================================================================
# Business Purpose:
#   Convert churn features into actionable SCORES and RISK BANDS.
#   Uses a rule-based scoring system for demo purposes.
#
# Scoring Logic (100-point scale):
#   - RECENCY: Up to 40 points (most important - when did they last shop?)
#   - FREQUENCY: Up to 30 points (are they shopping less often?)
#   - REFUND RATE: Up to 20 points (are they dissatisfied?)
#   - LOYALTY TIER: Up to 10 points (lower tiers = higher risk)
#
# Risk Bands:
#   - HIGH (60-100): Immediate intervention needed
#   - MEDIUM (30-59): Proactive re-engagement
#   - LOW (0-29): Maintain current engagement
#
# This table enables:
#   - Prioritized retention campaigns
#   - Sales team outreach lists
#   - Executive churn dashboards
#   - Next-best-action recommendations
# =============================================================================

from pyspark.sql.functions import col, when, expr, coalesce, lit, concat_ws, array

churn_features_df = spark.table("workspace.retail.gold_customer_churn_features")

# -----------------------------------------------------------------------------
# SCORING SYSTEM: Calculate Individual Risk Components
# -----------------------------------------------------------------------------

churn_scores = (
    churn_features_df
    
    # -------------------------------------------------------------------------
    # RECENCY SCORE (0-40 points) - MOST CRITICAL
    # -------------------------------------------------------------------------
    # Business Logic:
    #   The longer since last purchase, the higher the churn risk.
    #   - 180+ days: Maximum risk (40 points)
    #   - 90-180 days: High risk (25 points) - inactive for 1-2 quarters
    #   - 60-90 days: Elevated risk (15 points)
    #   - 30-60 days: Some risk (5 points)
    #   - <30 days: Active, no recency risk (0 points)
    .withColumn(
        "recency_score",
        when(col("days_since_last_purchase") > 180, 40)
        .when(col("days_since_last_purchase") > 90, 25)
        .when(col("days_since_last_purchase") > 60, 15)
        .when(col("days_since_last_purchase") > 30, 5)
        .otherwise(0)
    )
    
    # -------------------------------------------------------------------------
    # FREQUENCY SCORE (0-30 points)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Recent transaction frequency predicts future behavior.
    #   - 0 transactions in last 90 days: Maximum risk (30 points)
    #   - 1 transaction: Very low engagement (20 points)
    #   - 2-3 transactions: Below normal (10 points)
    #   - 4+ transactions: Normal/healthy (0 points)
    .withColumn(
        "frequency_score",
        when(col("transactions_last_90d") == 0, 30)
        .when(col("transactions_last_90d") <= 1, 20)
        .when(col("transactions_last_90d") <= 3, 10)
        .otherwise(0)
    )
    
    # -------------------------------------------------------------------------
    # REFUND/RETURN SCORE (0-20 points)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   High refund rates indicate:
    #   - Product dissatisfaction
    #   - Quality concerns
    #   - Mismatch between expectations and reality
    #   These customers are more likely to churn.
    #   - >30% refund rate: Major concern (20 points)
    #   - 15-30% refund rate: Moderate concern (10 points)
    #   - <15%: Normal (0 points)
    .withColumn(
        "refund_score",
        when(col("refund_return_rate") > 0.3, 20)
        .when(col("refund_return_rate") > 0.15, 10)
        .otherwise(0)
    )
    
    # -------------------------------------------------------------------------
    # LOYALTY TIER SCORE (0-10 points)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Lower-tier customers have:
    #   - Less invested in the program
    #   - Lower switching costs
    #   - Less emotional attachment
    #   Non-members and Bronze are highest risk.
    .withColumn(
        "loyalty_score",
        when(col("loyalty_tier") == "Bronze", 10)    # Lowest tier members
        .when(col("loyalty_tier") == "Silver", 5)     # Mid-tier members
        .otherwise(0)                                   # Gold/Platinum or non-members
    )
    
    # -------------------------------------------------------------------------
    # TOTAL CHURN SCORE (0-100)
    # -------------------------------------------------------------------------
    .withColumn(
        "churn_score",
        (col("recency_score") + col("frequency_score") + col("refund_score") + col("loyalty_score"))
    )
    
    # -------------------------------------------------------------------------
    # RISK BAND CLASSIFICATION
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Translate numeric score into business action category:
    #   - HIGH (60+): Needs immediate intervention (retention specialist)
    #   - MEDIUM (30-59): Needs proactive outreach (automated campaign)
    #   - LOW (0-29): Maintain current engagement (nurture campaign)
    .withColumn(
        "churn_risk_band",
        when(col("churn_score") >= 60, "High")
        .when(col("churn_score") >= 30, "Medium")
        .otherwise("Low")
    )
    
    # -------------------------------------------------------------------------
    # TOP REASON #1: Primary Driver of Churn Risk
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Identify the MAIN reason for churn risk (highest score component)
    #   This helps personalize the retention message.
    .withColumn(
        "top_reason_1",
        when(col("recency_score") >= 25, "Long time since last purchase")
        .when(col("frequency_score") >= 20, "Low purchase frequency")
        .when(col("refund_score") >= 10, "High refund/return rate")
        .otherwise("Low engagement signals")
    )
    
    # -------------------------------------------------------------------------
    # TOP REASON #2: Secondary Driver
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Identify secondary factors to provide more context for action.
    .withColumn(
        "top_reason_2",
        when((col("frequency_score") >= 10) & (col("recency_score") < 25), "Declining transaction frequency")
        .when((col("refund_score") >= 10) & (col("recency_score") < 25), "Quality or satisfaction concerns")
        .when(col("loyalty_tier") == "Bronze", "Low loyalty tier")
        .otherwise("Needs re-engagement")
    )
    
    # -------------------------------------------------------------------------
    # RECOMMENDED ACTION
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Map risk band to specific business action:
    #   - High: Human touch, personalized offer, retention specialist
    #   - Medium: Automated email campaign, targeted offers
    #   - Low: Continue standard nurture, loyalty rewards
    .withColumn(
        "recommended_action",
        when(col("churn_risk_band") == "High", "Immediate intervention: personalized retention offer")
        .when(col("churn_risk_band") == "Medium", "Proactive outreach: re-engagement campaign")
        .otherwise("Monitor and maintain: loyalty rewards")
    )
    
    .select(
        "customer_id",
        "churn_score",
        "churn_risk_band",
        "top_reason_1",
        "top_reason_2",
        "recommended_action"
    )
)

# Save as Delta table
churn_scores.write.format("delta").mode("overwrite").saveAsTable("workspace.retail.gold_customer_churn_scores")

print("✅ Gold Customer Churn Scores table created successfully")

# -----------------------------------------------------------------------------
# VALIDATION & RISK DISTRIBUTION ANALYSIS
# -----------------------------------------------------------------------------

# Check 1: Row count
display(spark.sql("""
SELECT COUNT(*) AS row_count FROM workspace.retail.gold_customer_churn_scores
"""))

# Check 2: Schema
display(spark.sql("""
DESCRIBE TABLE workspace.retail.gold_customer_churn_scores
"""))

# Check 3: Highest risk customers (prioritize these for retention)
display(spark.sql("""
SELECT * FROM workspace.retail.gold_customer_churn_scores ORDER BY churn_score DESC LIMIT 10
"""))

# Check 4: Risk band distribution (how many customers in each category?)
display(spark.sql("""
SELECT 
  churn_risk_band,
  COUNT(*) as customer_count,
  AVG(churn_score) as avg_churn_score,
  MIN(churn_score) as min_score,
  MAX(churn_score) as max_score
FROM workspace.retail.gold_customer_churn_scores
GROUP BY churn_risk_band
ORDER BY avg_churn_score DESC
"""))

# COMMAND ----------

# DBTITLE 1,Gold Layer 4: Next Best Actions
# =============================================================================
# GOLD LAYER 4: NEXT BEST ACTIONS
# =============================================================================
# Business Purpose:
#   Generate personalized, actionable recommendations for each customer.
#   This table bridges analytics → execution.
#
# What This Enables:
#   - Marketing campaign personalization
#   - Customer service agent guidance
#   - GenAI chatbot recommendations
#   - Automated email/SMS content
#   - Sales team talking points
#
# Personalization Elements:
#   - Customer summary (context for agents/systems)
#   - Next best action (what to do)
#   - Personalized offer (incentive based on value/risk)
#   - Recommended message (ready-to-send copy)
#
# Demo Use Cases:
#   1. Retention specialist dashboard: "These 50 high-risk customers need calls"
#   2. Email campaign: Load Medium-risk customers, send re-engagement emails
#   3. GenAI agent: Customer calls → agent sees summary + recommended offer
# =============================================================================

from pyspark.sql.functions import col, when, concat, lit, coalesce, round as spark_round

# Load Customer 360 and Churn Scores
customer_360_df = spark.table("workspace.retail.gold_customer_360")
churn_scores_df = spark.table("workspace.retail.gold_customer_churn_scores")

# -----------------------------------------------------------------------------
# BUILD PERSONALIZED RECOMMENDATIONS
# -----------------------------------------------------------------------------

next_best_actions = (
    customer_360_df
    .join(churn_scores_df, "customer_id", "inner")
    
    # -------------------------------------------------------------------------
    # CUSTOMER SUMMARY (Context for Agents/Systems)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Create a human-readable summary that gives context at a glance.
    #   Useful for:
    #   - Customer service agents answering calls
    #   - Marketing managers reviewing campaigns
    #   - GenAI systems generating responses
    .withColumn(
        "customer_summary",
        concat(
            coalesce(col("loyalty_tier"), lit("Standard")),           # Loyalty status
            lit(" tier customer with $"),
            spark_round(coalesce(col("total_revenue"), lit(0)), 2),   # Lifetime value
            lit(" lifetime value. Prefers "),
            coalesce(col("preferred_channel"), lit("mixed channels")), # Shopping preference
            lit(" shopping in "),
            coalesce(col("favorite_category"), lit("various categories")), # Product preference
            lit(". Last purchase: "),
            coalesce(col("days_since_last_purchase"), lit(0)),        # Recency
            lit(" days ago.")
        )
    )
    
    # -------------------------------------------------------------------------
    # NEXT BEST ACTION (What to Do)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Different risk levels require different actions:
    #
    #   HIGH RISK:
    #     - Assign to retention specialist (human touch)
    #     - Create urgency ("win-back campaign")
    #     - Offer exclusive deal in their favorite category
    #
    #   MEDIUM RISK:
    #     - Automated email campaign (scalable)
    #     - Feature products they're interested in
    #     - Offer loyalty points incentive (lower cost than discount)
    #
    #   LOW RISK:
    #     - Maintain engagement (don't over-market)
    #     - Celebrate milestones (positive reinforcement)
    #     - Early access to new products (VIP treatment)
    .withColumn(
        "next_best_action",
        when(col("churn_risk_band") == "High",
            concat(
                lit("URGENT: Win-back campaign. Customer showing high churn risk. Recommend immediate personalized outreach with exclusive offer in "),
                coalesce(col("favorite_category"), lit("preferred category")),
                lit(". Assign to retention specialist.")
            )
        )
        .when(col("churn_risk_band") == "Medium",
            concat(
                lit("Re-engagement needed. Send targeted email campaign featuring "),
                coalesce(col("favorite_category"), lit("best-selling items")),
                lit(" with loyalty points incentive via "),
                coalesce(col("preferred_channel"), lit("preferred channel")),
                lit(".")
            )
        )
        .otherwise(
            concat(
                lit("Maintain engagement. Continue personalized recommendations in "),
                coalesce(col("favorite_category"), lit("favorite category")),
                lit(". Celebrate loyalty milestones and offer early access to new arrivals.")
            )
        )
    )
    
    # -------------------------------------------------------------------------
    # PERSONALIZED OFFER (Incentive)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Match offer value to customer value and risk level:
    #
    #   HIGH RISK:
    #     - Gold/Platinum: 30% off + 2x points (high value, high investment)
    #     - Others: 20% off + free shipping (still generous)
    #
    #   MEDIUM RISK:
    #     - 15% off + bonus points (moderate incentive)
    #
    #   LOW RISK:
    #     - Early access + points (non-discount reward)
    #
    #   All offers personalized to favorite category when available.
    .withColumn(
        "personalized_offer",
        when(col("churn_risk_band") == "High",
            when(col("loyalty_tier").isin(["Gold", "Platinum"]),
                concat(lit("30% off + 2x points on next purchase in "), coalesce(col("favorite_category"), lit("any category")))
            )
            .otherwise(
                concat(lit("20% off + free shipping in "), coalesce(col("favorite_category"), lit("any category")))
            )
        )
        .when(col("churn_risk_band") == "Medium",
            concat(lit("15% off + bonus loyalty points in "), coalesce(col("favorite_category"), lit("selected categories")))
        )
        .otherwise(
            concat(lit("Early access to new arrivals in "), coalesce(col("favorite_category"), lit("your favorite category")), lit(" + 500 bonus points"))
        )
    )
    
    # -------------------------------------------------------------------------
    # RECOMMENDED MESSAGE (Ready-to-Send Copy)
    # -------------------------------------------------------------------------
    # Business Logic:
    #   Generate message copy that can be:
    #   - Sent directly via email/SMS
    #   - Used as template by GenAI systems
    #   - Customized further by marketing teams
    #
    #   Tone varies by risk level:
    #   - High: "We miss you" (emotional, urgent)
    #   - Medium: "New arrivals" (product-focused, exciting)
    #   - Low: "Thank you" (appreciation, positive reinforcement)
    .withColumn(
        "recommended_message",
        when(col("churn_risk_band") == "High",
            concat(
                lit("Hi "),
                col("full_name"),
                lit(", we miss you! As a valued "),
                coalesce(col("loyalty_tier"), lit("customer")),
                lit(" member, here's an exclusive offer just for you.")
            )
        )
        .when(col("churn_risk_band") == "Medium",
            concat(
                lit("Hello "),
                col("full_name"),
                lit(", we have new arrivals in "),
                coalesce(col("favorite_category"), lit("categories you love")),
                lit("! Check them out with your special discount.")
            )
        )
        .otherwise(
            concat(
                lit("Hi "),
                col("full_name"),
                lit(", thank you for being a loyal customer! Enjoy exclusive early access and bonus rewards.")
            )
        )
    )
    
    .select(
        "customer_id",
        "full_name",
        "loyalty_tier",
        "total_revenue",
        "favorite_category",
        "preferred_channel",
        "churn_score",
        "churn_risk_band",
        "customer_summary",
        "next_best_action",
        "personalized_offer",
        "recommended_message"
    )
)

# Save as Delta table
next_best_actions.write.format("delta").mode("overwrite").saveAsTable("workspace.retail.gold_next_best_actions")

print("✅ Gold Next Best Actions table created successfully")

# -----------------------------------------------------------------------------
# VALIDATION & CAMPAIGN PLANNING
# -----------------------------------------------------------------------------

# Check 1: Row count
display(spark.sql("""
SELECT COUNT(*) AS row_count FROM workspace.retail.gold_next_best_actions
"""))

# Check 2: Schema
display(spark.sql("""
DESCRIBE TABLE workspace.retail.gold_next_best_actions
"""))

# Check 3: Sample HIGH RISK customers (immediate action needed)
display(spark.sql("""
SELECT 
  customer_id,
  full_name,
  churn_risk_band,
  customer_summary,
  personalized_offer,
  recommended_message
FROM workspace.retail.gold_next_best_actions 
WHERE churn_risk_band = 'High'
LIMIT 5
"""))

# Check 4: Campaign volume by risk band and loyalty tier
# (Helps plan retention budget and campaign sizing)
display(spark.sql("""
SELECT 
  churn_risk_band,
  loyalty_tier,
  COUNT(*) as customer_count,
  AVG(total_revenue) as avg_revenue
FROM workspace.retail.gold_next_best_actions
GROUP BY churn_risk_band, loyalty_tier
ORDER BY churn_risk_band, loyalty_tier
"""))

# COMMAND ----------

# DBTITLE 1,Gold Layer 5: Executive KPIs
# =============================================================================
# GOLD LAYER 5: EXECUTIVE KPIs
# =============================================================================
# Business Purpose:
#   Create dashboard-ready business metrics for executive reporting.
#   Pre-aggregated by date and channel for fast dashboard rendering.
#
# Key Metrics (Daily + Channel Level):
#   - Total Revenue: Business top line
#   - Total Transactions: Volume indicator
#   - Unique Customers: Reach/engagement
#   - Average Order Value (AOV): Quality of transactions
#   - Refund/Return Rate: Quality/satisfaction indicator
#
# This table powers:
#   - Executive dashboards (CEO, CFO, CMO)
#   - Board presentations
#   - Daily business reviews
#   - Channel performance analysis (Ecommerce vs In-Store)
#   - Trend analysis and forecasting
#
# Design Principle:
#   Pre-aggregate at the grain executives care about (date + channel).
#   This makes dashboards FAST - no need to aggregate billions of transactions
#   at query time.
# =============================================================================

from pyspark.sql.functions import col, count, countDistinct, to_date, coalesce, lit, when

# Load unified transactions
trans_df = spark.table("workspace.retail.silver_transactions_unified")

# -----------------------------------------------------------------------------
# AGGREGATE DAILY METRICS BY CHANNEL
# -----------------------------------------------------------------------------
# Business Logic:
#   Group by:
#   - metric_date: Daily granularity (can roll up to week/month in BI tool)
#   - channel: Ecommerce vs In-Store (critical business segment)
#
#   This grain provides:
#   - Trend analysis over time
#   - Channel comparison and optimization
#   - Fast dashboard queries (pre-aggregated)

executive_kpis = (
    trans_df
    .withColumn("metric_date", to_date(col("transaction_timestamp")))  # Extract date from timestamp
    .groupBy("metric_date", "channel")
    .agg(
        # =====================================================================
        # REVENUE METRICS
        # =====================================================================
        # Total Revenue: The #1 metric executives care about
        # Measures: Business growth, market share, campaign effectiveness
        _sum(col("gross_amount")).alias("total_revenue"),
        
        # =====================================================================
        # VOLUME METRICS
        # =====================================================================
        # Total Transactions: Volume indicator
        # Measures: Customer activity, operational load, funnel conversion
        count("*").alias("total_transactions"),
        
        # Unique Customers: How many distinct customers shopped?
        # Measures: Customer reach, engagement breadth, retention
        countDistinct("customer_id").alias("unique_customers"),
        
        # =====================================================================
        # QUALITY METRICS
        # =====================================================================
        # Average Order Value (AOV): Revenue per transaction
        # Measures: Upsell effectiveness, basket size, customer value
        # Formula: Total Revenue / Total Transactions
        avg(col("gross_amount")).alias("avg_order_value"),
        
        # Refund/Return Count: How many transactions were refunded or returned?
        # Measures: Product quality, customer satisfaction, operational efficiency
        _sum(when(col("is_refund_or_return") == True, 1).otherwise(0)).alias("refund_return_count")
    )
    
    # Calculate Refund/Return Rate
    # Business Logic:
    #   Percentage of transactions that end in refund/return
    #   - High rate (>5%): Quality issues, sizing problems, misleading descriptions
    #   - Low rate (<2%): Good product-market fit, accurate descriptions
    #   - Benchmark varies by category (apparel higher than electronics)
    .withColumn(
        "refund_return_rate",
        when(col("total_transactions") > 0, col("refund_return_count") / col("total_transactions")).otherwise(0.0)
    )
    
    .select(
        "metric_date",
        "channel",
        "total_revenue",
        "total_transactions",
        "unique_customers",
        "avg_order_value",
        "refund_return_rate"
    )
    .orderBy("metric_date", "channel")  # Sort for easy reading
)

# Save as Delta table
executive_kpis.write.format("delta").mode("overwrite").saveAsTable("workspace.retail.gold_executive_kpis")

print("✅ Gold Executive KPIs table created successfully")

# -----------------------------------------------------------------------------
# VALIDATION & BUSINESS INSIGHTS
# -----------------------------------------------------------------------------

# Check 1: Row count (should be ~730-1460 rows for 1-2 years of data, 2 channels)
display(spark.sql("""
SELECT COUNT(*) AS row_count FROM workspace.retail.gold_executive_kpis
"""))

# Check 2: Schema
display(spark.sql("""
DESCRIBE TABLE workspace.retail.gold_executive_kpis
"""))

# Check 3: Most recent performance (what happened yesterday?)
display(spark.sql("""
SELECT * FROM workspace.retail.gold_executive_kpis ORDER BY metric_date DESC, channel LIMIT 10
"""))

# Check 4: Channel Performance Summary
# Business Insight:
#   - Which channel drives more revenue?
#   - Which has better AOV?
#   - Which has quality issues (high refund rate)?
display(spark.sql("""
SELECT 
  channel,
  COUNT(DISTINCT metric_date) as date_count,
  SUM(total_revenue) as total_revenue,
  SUM(total_transactions) as total_transactions,
  AVG(avg_order_value) as avg_order_value,
  AVG(refund_return_rate) as avg_refund_rate
FROM workspace.retail.gold_executive_kpis
GROUP BY channel
ORDER BY total_revenue DESC
"""))

# Check 5: Overall Business Health
# Business Insight:
#   - Total business size
#   - Data coverage (how many days?)
#   - Customer engagement (total visits)
display(spark.sql("""
SELECT 
  SUM(total_revenue) as total_business_revenue,
  SUM(total_transactions) as total_business_transactions,
  COUNT(DISTINCT metric_date) as days_of_data,
  SUM(unique_customers) as total_unique_customer_visits
FROM workspace.retail.gold_executive_kpis
"""))

# COMMAND ----------

# DBTITLE 1,Gold Layer Summary
# =============================================================================
# GOLD LAYER SUMMARY
# =============================================================================
# Overview of all Gold tables created in the RetailIQ demo.
# This cell provides a complete picture of what's been built.
# =============================================================================

print("="*80)
print("RETAILIQ GOLD LAYER - COMPLETE")
print("="*80)
print("\n✅ All 5 Gold Delta tables created successfully in workspace.retail schema\n")

# =============================================================================
# TABLE 1: CUSTOMER 360
# =============================================================================
print("1. workspace.retail.gold_customer_360")
print("   Purpose: Trusted single view of each customer")
print("   Key fields: customer_id, full_name, loyalty_tier, total_revenue, behavioral metrics")
print("   Use cases: Segmentation, personalization, ML features, executive dashboards\n")

# =============================================================================
# TABLE 2: CUSTOMER CHURN FEATURES
# =============================================================================
print("2. workspace.retail.gold_customer_churn_features")
print("   Purpose: Model-ready features for churn prediction")
print("   Key fields: customer_id, recency/frequency metrics, churn_label_proxy")
print("   Use cases: ML model training, churn prediction, customer scoring\n")

# =============================================================================
# TABLE 3: CUSTOMER CHURN SCORES
# =============================================================================
print("3. workspace.retail.gold_customer_churn_scores")
print("   Purpose: Churn risk assessment with rule-based scoring")
print("   Key fields: customer_id, churn_score, churn_risk_band, top reasons")
print("   Use cases: Retention campaigns, sales prioritization, executive alerts\n")

# =============================================================================
# TABLE 4: NEXT BEST ACTIONS
# =============================================================================
print("4. workspace.retail.gold_next_best_actions")
print("   Purpose: Personalized marketing activation & GenAI prompts")
print("   Key fields: customer_id, next_best_action, personalized_offer, messages")
print("   Use cases: Campaign automation, GenAI agents, customer service, email/SMS\n")

# =============================================================================
# TABLE 5: EXECUTIVE KPIs
# =============================================================================
print("5. workspace.retail.gold_executive_kpis")
print("   Purpose: Executive dashboard metrics by date and channel")
print("   Key fields: metric_date, channel, revenue, transactions, customers")
print("   Use cases: Executive dashboards, board presentations, trend analysis\n")

print("="*80)
print("DEMO STORY FLOW")
print("="*80)
print("➡️  Bronze: Raw ingestion from Ecommerce + POS + Loyalty systems")
print("   - Data is messy, duplicated, inconsistent")
print("   - Different schemas, data quality issues\n")

print("➡️  Silver: Cleaned, unified, deduplicated customer & transaction data")
print("   - Single source of truth for customers and transactions")
print("   - Data quality rules applied")
print("   - Ready for analytics\n")

print("➡️  Gold: Business-ready analytics, churn ML, GenAI actions, executive metrics")
print("   - Customer 360 view")
print("   - Churn prediction and scoring")
print("   - Personalized next-best-actions")
print("   - Executive KPIs\n")

print("🎯 Ready for: Dashboards, ML models, GenAI agents, executive reporting\n")

# =============================================================================
# GOLD LAYER STATISTICS
# =============================================================================
print("="*80)
print("GOLD LAYER STATISTICS")
print("="*80)

# Get row counts for all Gold tables
gold_tables = [
    "workspace.retail.gold_customer_360",
    "workspace.retail.gold_customer_churn_features",
    "workspace.retail.gold_customer_churn_scores",
    "workspace.retail.gold_next_best_actions",
    "workspace.retail.gold_executive_kpis"
]

for table in gold_tables:
    count_result = spark.sql(f"SELECT COUNT(*) as cnt FROM {table}").collect()[0]['cnt']
    print(f"{table}: {count_result:,} rows")

print("="*80)

# =============================================================================
# NEXT STEPS
# =============================================================================
print("\n🚀 NEXT STEPS:")
print("\n1. CREATE DASHBOARDS:")
print("   - Executive KPI dashboard (use gold_executive_kpis)")
print("   - Churn risk dashboard (use gold_customer_churn_scores)")
print("   - Customer segmentation dashboard (use gold_customer_360)\n")

print("2. BUILD ML MODELS:")
print("   - Train churn prediction model (use gold_customer_churn_features)")
print("   - Customer lifetime value model (use gold_customer_360)")
print("   - Product recommendation model\n")

print("3. ACTIVATE GENAI:")
print("   - Connect GenAI agent to gold_next_best_actions")
print("   - Generate personalized customer messages")
print("   - Power customer service chatbots\n")

print("4. OPERATIONALIZE:")
print("   - Schedule daily Gold table refreshes")
print("   - Set up data quality monitoring")
print("   - Create retention campaign workflows")
print("   - Build executive email alerts\n")

print("="*80)