import dlt
from pyspark.sql import functions as F
from pyspark.sql import Window

VOLUME_BASE = "/Volumes/workspace/retail/raw_data"

# =============================================================================
# BRONZE LAYER - Auto Loader ingestion from UC Volume
# =============================================================================

@dlt.table(
    name="sdp_bronze_products",
    comment="Raw product catalog ingested via Auto Loader from UC volume"
)
def bronze_products():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_BASE}/bronze_products/")
    )


@dlt.table(
    name="sdp_bronze_customers",
    comment="Raw customer records ingested via Auto Loader from UC volume"
)
def bronze_customers():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_BASE}/bronze_customers/")
    )


@dlt.table(
    name="sdp_bronze_loyalty_profiles",
    comment="Raw loyalty program data ingested via Auto Loader from UC volume"
)
def bronze_loyalty_profiles():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_BASE}/bronze_loyalty_profiles/")
    )


@dlt.table(
    name="sdp_bronze_ecommerce_orders",
    comment="Raw ecommerce orders ingested via Auto Loader from UC volume"
)
def bronze_ecommerce_orders():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_BASE}/bronze_ecommerce_orders/")
    )


@dlt.table(
    name="sdp_bronze_pos_transactions",
    comment="Raw POS transactions ingested via Auto Loader from UC volume"
)
def bronze_pos_transactions():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{VOLUME_BASE}/bronze_pos_transactions/")
    )


# =============================================================================
# SILVER LAYER - Cleansed, typed, validated with expectations
# =============================================================================

@dlt.table(
    name="sdp_silver_products",
    comment="Cleansed product catalog with standardized types"
)
@dlt.expect("valid_product_id", "product_id IS NOT NULL")
@dlt.expect("valid_price", "price >= 0")
def silver_products():
    return (
        spark.readStream.table("sdp_bronze_products")
        .select(
            F.col("product_id"),
            F.trim(F.col("product_name")).alias("product_name"),
            F.initcap(F.col("category")).alias("category"),
            F.trim(F.col("brand")).alias("brand"),
            F.col("price").cast("double"),
        )
    )


@dlt.table(
    name="sdp_silver_customers",
    comment="Cleansed customer records with validated contact info"
)
@dlt.expect_or_drop("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect("valid_email", "email IS NOT NULL AND email LIKE '%@%'")
def silver_customers():
    return (
        spark.readStream.table("sdp_bronze_customers")
        .select(
            F.col("customer_id"),
            F.trim(F.col("first_name")).alias("first_name"),
            F.trim(F.col("last_name")).alias("last_name"),
            F.lower(F.trim(F.col("email"))).alias("email"),
            F.col("phone"),
            F.col("registration_date"),
            F.trim(F.col("city")).alias("city"),
            F.trim(F.col("state")).alias("state"),
            F.trim(F.col("country")).alias("country"),
        )
    )


@dlt.table(
    name="sdp_silver_loyalty",
    comment="Cleansed loyalty profiles with validated tiers"
)
@dlt.expect("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect("valid_tier", "loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum')")
def silver_loyalty():
    return (
        spark.readStream.table("sdp_bronze_loyalty_profiles")
        .select(
            F.col("customer_id"),
            F.col("loyalty_tier"),
            F.col("points_balance").cast("long"),
            F.col("member_since"),
            F.col("status"),
        )
    )


@dlt.table(
    name="sdp_silver_orders",
    comment="Cleansed ecommerce orders with validated amounts"
)
@dlt.expect_or_drop("valid_order_id", "order_id IS NOT NULL")
@dlt.expect("valid_amount", "order_amount >= 0")
@dlt.expect("valid_quantity", "quantity > 0")
def silver_orders():
    return (
        spark.readStream.table("sdp_bronze_ecommerce_orders")
        .select(
            F.col("order_id"),
            F.col("customer_id"),
            F.col("product_id"),
            F.col("order_date"),
            F.col("quantity").cast("long"),
            F.col("order_amount").cast("double"),
            F.col("order_status"),
            F.col("channel"),
            F.col("payment_method"),
        )
    )


@dlt.table(
    name="sdp_silver_pos",
    comment="Cleansed POS transactions with validated amounts"
)
@dlt.expect_or_drop("valid_transaction_id", "transaction_id IS NOT NULL")
@dlt.expect("valid_amount", "transaction_amount >= 0")
@dlt.expect("valid_quantity", "quantity > 0")
def silver_pos():
    return (
        spark.readStream.table("sdp_bronze_pos_transactions")
        .select(
            F.col("transaction_id"),
            F.col("customer_id"),
            F.col("product_id"),
            F.col("transaction_date"),
            F.col("quantity").cast("long"),
            F.col("transaction_amount").cast("double"),
            F.col("transaction_status"),
            F.col("channel"),
            F.col("store_id"),
            F.col("store_name"),
            F.col("store_city"),
            F.col("store_state"),
            F.col("payment_method"),
        )
    )


# =============================================================================
# GOLD LAYER - Business aggregations (Materialized Views, batch reads)
# These canonical gold_* tables feed ML training, Lakebase sync, Genie, and the app
# =============================================================================

@dlt.table(
    name="gold_customer_360",
    comment="Unified customer 360 view: RFM metrics, loyalty, omnichannel revenue"
)
def gold_customer_360():
    customers = spark.read.table("sdp_silver_customers")
    loyalty = spark.read.table("sdp_silver_loyalty")
    orders = spark.read.table("sdp_silver_orders")
    pos = spark.read.table("sdp_silver_pos")
    products = spark.read.table("sdp_silver_products")

    orders_agg = (
        orders.filter(F.col("order_status") == "Completed")
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("ecom_orders"),
            F.sum("order_amount").alias("ecom_revenue"),
        )
    )

    pos_agg = (
        pos.filter(F.col("transaction_status") == "Completed")
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("pos_orders"),
            F.sum("transaction_amount").alias("pos_revenue"),
        )
    )

    refunds = (
        orders.filter(F.col("order_status").isin("Refunded", "Returned"))
        .groupBy("customer_id")
        .agg(F.count("*").alias("refund_return_count"))
    )

    purchase_dates = (
        orders.filter(F.col("order_status") == "Completed")
        .groupBy("customer_id")
        .agg(
            F.min("order_date").alias("first_purchase_date"),
            F.max("order_date").alias("last_purchase_date"),
        )
    )

    cat_orders = (
        orders.filter(F.col("order_status") == "Completed")
        .join(products, "product_id")
        .groupBy("customer_id", "category")
        .agg(F.count("*").alias("cat_count"))
    )
    fav_cat = (
        cat_orders.withColumn("rn", F.row_number().over(
            Window.partitionBy("customer_id").orderBy(F.desc("cat_count"))
        ))
        .filter("rn = 1")
        .select("customer_id", F.col("category").alias("favorite_category"))
    )

    channel_orders = (
        orders.filter(F.col("order_status") == "Completed")
        .groupBy("customer_id", "channel")
        .agg(F.count("*").alias("ch_count"))
    )
    pref_channel = (
        channel_orders.withColumn("rn", F.row_number().over(
            Window.partitionBy("customer_id").orderBy(F.desc("ch_count"))
        ))
        .filter("rn = 1")
        .select("customer_id", F.col("channel").alias("preferred_channel"))
    )

    combined = (
        customers
        .join(loyalty, "customer_id", "left")
        .join(orders_agg, "customer_id", "left")
        .join(pos_agg, "customer_id", "left")
        .join(refunds, "customer_id", "left")
        .join(purchase_dates, "customer_id", "left")
        .join(fav_cat, "customer_id", "left")
        .join(pref_channel, "customer_id", "left")
    )

    total_txn = F.coalesce(F.col("ecom_orders"), F.lit(0)) + F.coalesce(F.col("pos_orders"), F.lit(0))
    total_rev = F.coalesce(F.col("ecom_revenue"), F.lit(0)) + F.coalesce(F.col("pos_revenue"), F.lit(0))

    return (
        combined.select(
            F.col("customer_id"),
            F.concat_ws(" ", F.col("first_name"), F.col("last_name")).alias("full_name"),
            F.col("email"),
            F.col("phone"),
            F.col("city"),
            F.col("state"),
            F.col("loyalty_tier"),
            F.coalesce(F.col("points_balance"), F.lit(0)).alias("points_balance"),
            F.col("loyalty_tier").isNotNull().alias("is_loyalty_member"),
            total_txn.alias("total_transactions"),
            F.coalesce(F.col("ecom_orders"), F.lit(0)).alias("completed_transactions"),
            total_rev.alias("total_revenue"),
            F.coalesce(F.col("ecom_revenue"), F.lit(0)).alias("ecommerce_revenue"),
            F.coalesce(F.col("pos_revenue"), F.lit(0)).alias("instore_revenue"),
            F.when(total_txn > 0, total_rev / total_txn).otherwise(0.0).alias("avg_order_value"),
            F.col("first_purchase_date"),
            F.col("last_purchase_date"),
            F.coalesce(F.datediff(F.current_date(), F.col("last_purchase_date")), F.lit(9999)).alias("days_since_last_purchase"),
            F.col("favorite_category"),
            F.col("preferred_channel"),
            F.coalesce(F.col("refund_return_count"), F.lit(0)).alias("refund_return_count"),
            total_rev.alias("customer_lifetime_value_proxy"),
        )
    )


@dlt.table(
    name="gold_customer_churn_features",
    comment="ML-ready churn prediction features derived from customer 360 and silver layer. Label has 12%% noise to prevent trivial separation."
)
def gold_customer_churn_features():
    cust360 = spark.read.table("gold_customer_360")
    customers = spark.read.table("sdp_silver_customers")
    orders = spark.read.table("sdp_silver_orders")
    pos = spark.read.table("sdp_silver_pos")

    # Customer tenure from registration date
    tenure = customers.select(
        F.col("customer_id"),
        F.coalesce(F.datediff(F.current_date(), F.col("registration_date")), F.lit(0)).cast("int").alias("customer_tenure_days"),
    )

    # Last 30/90 day aggregations from completed orders
    orders_30d = (
        orders.filter(
            (F.col("order_status") == "Completed") &
            (F.col("order_date") >= F.date_sub(F.current_date(), 30))
        )
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("ord_30d_cnt"),
            F.sum("order_amount").alias("ord_30d_rev"),
        )
    )

    orders_90d = (
        orders.filter(
            (F.col("order_status") == "Completed") &
            (F.col("order_date") >= F.date_sub(F.current_date(), 90))
        )
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("ord_90d_cnt"),
            F.sum("order_amount").alias("ord_90d_rev"),
        )
    )

    # Last 30/90 day aggregations from completed POS transactions
    pos_30d = (
        pos.filter(
            (F.col("transaction_status") == "Completed") &
            (F.col("transaction_date") >= F.date_sub(F.current_date(), 30))
        )
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("pos_30d_cnt"),
            F.sum("transaction_amount").alias("pos_30d_rev"),
        )
    )

    pos_90d = (
        pos.filter(
            (F.col("transaction_status") == "Completed") &
            (F.col("transaction_date") >= F.date_sub(F.current_date(), 90))
        )
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("pos_90d_cnt"),
            F.sum("transaction_amount").alias("pos_90d_rev"),
        )
    )

    combined = (
        cust360
        .join(tenure, "customer_id", "left")
        .join(orders_30d, "customer_id", "left")
        .join(orders_90d, "customer_id", "left")
        .join(pos_30d, "customer_id", "left")
        .join(pos_90d, "customer_id", "left")
    )

    txn_30d = F.coalesce(F.col("ord_30d_cnt"), F.lit(0)) + F.coalesce(F.col("pos_30d_cnt"), F.lit(0))
    txn_90d = F.coalesce(F.col("ord_90d_cnt"), F.lit(0)) + F.coalesce(F.col("pos_90d_cnt"), F.lit(0))
    rev_30d = F.coalesce(F.col("ord_30d_rev"), F.lit(0)) + F.coalesce(F.col("pos_30d_rev"), F.lit(0))
    rev_90d = F.coalesce(F.col("ord_90d_rev"), F.lit(0)) + F.coalesce(F.col("pos_90d_rev"), F.lit(0))

    total_txn = F.coalesce(F.col("total_transactions"), F.lit(0))
    refund_cnt = F.coalesce(F.col("refund_return_count"), F.lit(0))
    refund_rate = F.when(total_txn > 0, refund_cnt / total_txn).otherwise(0.0)

    # Base churn label: deterministic rule (days_since_last_purchase > 90)
    base_label = F.when(F.col("days_since_last_purchase") > 90, F.lit(1)).otherwise(F.lit(0))

    # Add 12%% noise: flip label with probability 0.12 to prevent trivial separation
    noisy_label = F.when(F.rand() < 0.12, F.lit(1) - base_label).otherwise(base_label)

    return (
        combined.select(
            F.col("customer_id"),
            F.col("loyalty_tier"),
            F.col("points_balance").cast("long"),
            F.col("customer_tenure_days"),
            F.col("days_since_last_purchase"),
            txn_30d.alias("transactions_last_30d"),
            txn_90d.alias("transactions_last_90d"),
            rev_30d.alias("revenue_last_30d"),
            rev_90d.alias("revenue_last_90d"),
            F.col("avg_order_value"),
            F.col("preferred_channel"),
            refund_rate.alias("refund_return_rate"),
            F.col("favorite_category"),
            noisy_label.cast("int").alias("churn_label_proxy"),
        )
    )


@dlt.table(
    name="gold_customer_churn_scores",
    comment="Customer churn risk scores based on RFM analysis. ML model predictions overwrite this after training."
)
def gold_customer_churn_scores():
    cust360 = spark.read.table("gold_customer_360")

    return (
        cust360.select(
            F.col("customer_id"),
            F.least(
                F.lit(100),
                F.when(F.col("days_since_last_purchase") > 180, 40).otherwise(0)
                + F.when(F.col("days_since_last_purchase") > 90, 20).otherwise(0)
                + F.when(F.col("total_transactions") < 5, 15).otherwise(0)
                + F.when(F.col("total_revenue") < 500, 15).otherwise(0)
                + F.when(F.col("refund_return_count") > 3, 10).otherwise(0),
            ).cast("int").alias("churn_score"),
            F.when(
                (F.col("days_since_last_purchase") > 180) | (F.col("total_revenue") < 100),
                F.lit("High"),
            ).when(
                (F.col("days_since_last_purchase") > 90) | (F.col("total_revenue") < 500),
                F.lit("Medium"),
            ).otherwise(F.lit("Low")).alias("churn_risk_band"),
            F.when(F.col("days_since_last_purchase") > 180, F.lit("Extended purchase gap (180+ days)")).otherwise(F.lit("No recent engagement")).alias("top_reason_1"),
            F.when(F.col("total_transactions") < 5, F.lit("Low purchase frequency")).when(F.col("total_revenue") < 500, F.lit("Low customer lifetime value")).when(F.col("refund_return_count") > 3, F.lit("High return rate")).otherwise(F.lit("Limited engagement")).alias("top_reason_2"),
            F.when(
                (F.col("days_since_last_purchase") > 180) | (F.col("total_revenue") < 100),
                F.lit("Send re-engagement campaign with 15% discount"),
            ).when(
                (F.col("days_since_last_purchase") > 90) | (F.col("total_revenue") < 500),
                F.lit("Send personalized product recommendations"),
            ).otherwise(F.lit("Continue loyalty rewards program")).alias("recommended_action"),
        )
    )


@dlt.table(
    name="gold_next_best_actions",
    comment="Personalized next best actions based on churn risk and customer profile"
)
def gold_next_best_actions():
    cust360 = spark.read.table("gold_customer_360")
    churn = spark.read.table("gold_customer_churn_scores")

    return (
        cust360.join(churn, "customer_id")
        .select(
            F.col("customer_id"),
            F.col("full_name"),
            F.col("loyalty_tier"),
            F.col("total_revenue"),
            F.col("favorite_category"),
            F.col("preferred_channel"),
            F.col("churn_score"),
            F.col("churn_risk_band"),
            F.concat_ws(" ", F.col("full_name"), F.lit("is a"), F.col("loyalty_tier"), F.lit("member with $"), F.round(F.col("total_revenue"), 2), F.lit("in lifetime revenue.")).alias("customer_summary"),
            F.when(F.col("churn_risk_band") == "High", F.lit("Immediate re-engagement: Send 15% discount via preferred channel"))
            .when(F.col("churn_risk_band") == "Medium", F.lit("Nurture: Send category recommendations and loyalty perk"))
            .otherwise(F.lit("Maintain: Continue rewards and early access to new products")).alias("next_best_action"),
            F.when(F.col("churn_risk_band") == "High", F.concat_ws("", F.lit("15% off your next "), F.col("favorite_category"), F.lit(" purchase")))
            .when(F.col("churn_risk_band") == "Medium", F.lit("Free shipping on next order + 2x loyalty points"))
            .otherwise(F.lit("VIP early access to new arrivals")).alias("personalized_offer"),
            F.when(F.col("churn_risk_band") == "High", F.concat_ws(" ", F.lit("Hi"), F.col("full_name"), F.lit("- we miss you! Here's 15% off to welcome you back.")))
            .when(F.col("churn_risk_band") == "Medium", F.concat_ws(" ", F.lit("Hi"), F.col("full_name"), F.lit("- check out our latest"), F.col("favorite_category"), F.lit("selections.")))
            .otherwise(F.concat_ws(" ", F.lit("Hi"), F.col("full_name"), F.lit("- as a valued"), F.col("loyalty_tier"), F.lit("member, enjoy early access."))).alias("recommended_message"),
        )
    )
