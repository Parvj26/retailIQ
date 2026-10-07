# Databricks notebook source
# DBTITLE 1,Cell 1
# =============================================================================
# SILVER LAYER SETUP - RetailIQ Customer Intelligence Platform
# =============================================================================
# Purpose: Create cleaned, standardized, deduplicated Silver layer tables
# Source: Bronze layer tables in workspace.retail
# Target: Silver layer tables in workspace.retail
# 
# BUSINESS CONTEXT:
# The Silver layer serves as the "single source of truth" for analytics by:
# - Removing duplicates and data quality issues from raw Bronze data
# - Standardizing formats, naming conventions, and data types
# - Enriching data with calculated fields and business flags
# - Conforming dimensions for consistent cross-channel reporting
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

# Set catalog and schema context
# Using workspace catalog for development; production would use a governed catalog
spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA retail")

print("✅ Silver Layer setup complete")
print(f"📍 Working in: {spark.sql('SELECT current_catalog(), current_schema()').collect()[0]}")
print(f"📅 Processing date: {datetime.now().strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Cell 2
# =============================================================================
# SILVER TABLE 1: silver_customers
# =============================================================================
# Business Purpose: Clean customer master data
# Transformations: Deduplicate, standardize email/phone, add tenure and validity flags
# 
# BUSINESS RATIONALE:
# Customer data may have duplicates from multiple registration sources (web, mobile, in-store).
# We keep the most recent registration to ensure latest contact info while maintaining
# historical customer_id for transaction linking. Email standardization enables accurate
# customer matching and communication campaigns.
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=" * 80)
print("CREATING SILVER TABLE 1: silver_customers")
print("=" * 80)

# Read Bronze customers
bronze_customers = spark.table("workspace.retail.bronze_customers")

# DEDUPLICATION STRATEGY:
# Window function ranks records by registration_date (most recent first)
# Business rule: Latest registration contains most up-to-date contact information
window_spec = Window.partitionBy("customer_id").orderBy(F.desc("registration_date"))

silver_customers = bronze_customers \
    .withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num") \
    .select(
        F.col("customer_id"),
        
        # BUSINESS FIELD: full_name
        # Concatenate first and last name for easier display in reports and dashboards
        F.concat_ws(" ", F.col("first_name"), F.col("last_name")).alias("full_name"),
        
        # STANDARDIZATION: email to lowercase
        # Ensures case-insensitive matching for customer identification
        # Prevents duplicate accounts (e.g., John@email.com vs john@email.com)
        F.lower(F.trim(F.col("email"))).alias("email"),
        
        # Keep phone as-is (already formatted in Bronze)
        F.col("phone"),
        F.col("registration_date"),
        
        # Geographic fields for regional analysis and targeted campaigns
        F.col("city"),
        F.col("state"),
        F.col("country"),
        
        # BUSINESS METRIC: customer_tenure_days
        # Used for customer lifetime value calculations and segmentation
        # Helps identify new vs. established customers for retention strategies
        F.datediff(F.current_date(), F.col("registration_date")).alias("customer_tenure_days"),
        
        # DATA QUALITY FLAG: is_customer_record_valid
        # Flags records with missing critical fields for data quality monitoring
        # Downstream analytics can filter on this flag for accurate reporting
        F.when(
            (F.col("customer_id").isNotNull()) &
            (F.col("email").isNotNull()) &
            (F.col("registration_date").isNotNull()),
            True
        ).otherwise(False).alias("is_customer_record_valid")
    )

# Save as Delta table with overwrite mode
# Overwrite ensures idempotent pipeline execution for daily refreshes
silver_customers.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.silver_customers")

# Validation and data quality checks
count = silver_customers.count()
valid_count = silver_customers.filter("is_customer_record_valid = true").count()

print(f"\n✅ Created silver_customers table")
print(f"📊 Total rows: {count:,}")
print(f"✓  Valid records: {valid_count:,} ({(valid_count/count)*100:.1f}%)")
print(f"⚠️  Invalid records: {count - valid_count:,}")
print("\n📋 Schema:")
silver_customers.printSchema()
print("\n📊 Sample records:")
display(silver_customers.orderBy(F.desc("customer_tenure_days")).limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 3
# =============================================================================
# SILVER TABLE 2: silver_loyalty_profiles
# =============================================================================
# Business Purpose: Clean loyalty profile data
# Transformations: Deduplicate, standardize tier/status, validate points
# 
# BUSINESS RATIONALE:
# Loyalty program data drives personalized marketing, retention campaigns, and
# customer segmentation. Point balance validation prevents negative balances from
# data errors. Tier standardization ensures consistent treatment of members across
# all channels (web, mobile, in-store POS systems).
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=" * 80)
print("CREATING SILVER TABLE 2: silver_loyalty_profiles")
print("=" * 80)

# Read Bronze loyalty profiles
bronze_loyalty = spark.table("workspace.retail.bronze_loyalty_profiles")

# DEDUPLICATION STRATEGY:
# Keep most recent member_since date for each customer
# Business rule: Latest enrollment date reflects most accurate tier status
window_spec = Window.partitionBy("customer_id").orderBy(F.desc("member_since"))

silver_loyalty = bronze_loyalty \
    .withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num") \
    .select(
        F.col("customer_id"),
        
        # STANDARDIZATION: loyalty_tier
        # Consistent capitalization (Bronze, Silver, Gold, Platinum)
        # Enables accurate tier-based reporting and benefits application
        F.initcap(F.trim(F.col("loyalty_tier"))).alias("loyalty_tier"),
        
        # DATA QUALITY: points_balance validation
        # Business rule: Points cannot be negative due to accounting rules
        # Negative values indicate data quality issues - reset to 0 and flag for review
        F.when(F.col("points_balance") < 0, 0).otherwise(F.col("points_balance")).alias("points_balance"),
        
        # member_since tracks loyalty program tenure
        # Used for calculating member lifetime value and retention metrics
        F.col("member_since"),
        
        # STANDARDIZATION: status field
        # Values: Active, Inactive, Suspended
        # Determines eligibility for points earning and redemption
        F.initcap(F.trim(F.col("status"))).alias("status"),
        
        # BUSINESS FLAG: is_loyalty_member
        # Simplifies downstream joins and filtering for member-only analytics
        F.lit(True).alias("is_loyalty_member")
    )

# Save as Delta table with overwrite mode
silver_loyalty.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.silver_loyalty_profiles")

# Validation and business metrics
count = silver_loyalty.count()
print(f"\n✅ Created silver_loyalty_profiles table")
print(f"📊 Total rows: {count:,}")
print(f"\n📊 Loyalty tier distribution:")
display(silver_loyalty.groupBy("loyalty_tier").count().orderBy("loyalty_tier"))
print(f"\n📊 Points balance statistics:")
display(silver_loyalty.select(
    F.min("points_balance").alias("min_points"),
    F.avg("points_balance").alias("avg_points"),
    F.max("points_balance").alias("max_points")
))
print("\n📋 Schema:")
silver_loyalty.printSchema()
print("\n📊 Sample high-value members:")
display(silver_loyalty.orderBy(F.desc("points_balance")).limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 4
# =============================================================================
# SILVER TABLE 3: silver_products
# =============================================================================
# Business Purpose: Clean product reference data
# Transformations: Deduplicate, standardize category/brand, validate price
# 
# BUSINESS RATIONALE:
# Product master data is the foundation for inventory management, pricing analytics,
# and merchandising decisions. Price validation ensures only valid products appear
# in analytics (removes test data and discontinued items with $0 price). Category
# and brand standardization enables accurate product hierarchy reporting.
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=" * 80)
print("CREATING SILVER TABLE 3: silver_products")
print("=" * 80)

# Read Bronze products
bronze_products = spark.table("workspace.retail.bronze_products")

# DEDUPLICATION STRATEGY:
# Keep first occurrence of each product_id
# Business rule: Product catalog should have one record per product_id
window_spec = Window.partitionBy("product_id").orderBy("product_id")

silver_products = bronze_products \
    .withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num") \
    .select(
        F.col("product_id"),
        
        # Clean product name (remove leading/trailing spaces)
        F.trim(F.col("product_name")).alias("product_name"),
        
        # STANDARDIZATION: category
        # Consistent capitalization (Electronics, Clothing, Home & Garden)
        # Enables accurate product hierarchy and category performance reporting
        F.initcap(F.trim(F.col("category"))).alias("category"),
        
        # STANDARDIZATION: brand
        # Consistent capitalization for brand analytics and vendor management
        F.initcap(F.trim(F.col("brand"))).alias("brand"),
        
        # DATA QUALITY: price validation
        # Business rule: Prices must be positive (> 0)
        # Null prices indicate test data, discontinued items, or data quality issues
        # These products are excluded from sales analytics and reporting
        F.when(F.col("price") <= 0, None).otherwise(F.col("price")).alias("price")
    ) \
    .filter(F.col("price").isNotNull())  # Remove products with invalid prices

# Save as Delta table with overwrite mode
silver_products.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.silver_products")

# Validation and business metrics
count = silver_products.count()
print(f"\n✅ Created silver_products table")
print(f"📊 Total rows: {count:,}")
print(f"\n📊 Products by category:")
display(silver_products.groupBy("category").count().orderBy(F.desc("count")))
print(f"\n📊 Price statistics:")
display(silver_products.select(
    F.min("price").alias("min_price"),
    F.avg("price").alias("avg_price"),
    F.max("price").alias("max_price")
))
print("\n📋 Schema:")
silver_products.printSchema()
print("\n📊 Sample products:")
display(silver_products.orderBy("product_id").limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 5
# =============================================================================
# SILVER TABLE 4: silver_transactions_unified
# =============================================================================
# Business Purpose: Unified transaction table across ecommerce and POS channels
# Transformations: Standardize columns, add flags, filter invalid records
# 
# BUSINESS RATIONALE:
# Combines transactions from two source systems (Postgres ecommerce and POS) into
# a single conformed fact table. This enables:
# - Unified customer journey analytics across all channels
# - Consistent revenue reporting regardless of purchase channel
# - Cross-channel customer behavior analysis (omnichannel insights)
# - Single source of truth for transaction-level metrics and KPIs
# =============================================================================

from pyspark.sql import functions as F

print("=" * 80)
print("CREATING SILVER TABLE 4: silver_transactions_unified")
print("=" * 80)

# =============================================================================
# PART 1: Transform Ecommerce Orders to Unified Schema
# =============================================================================

# Read Bronze ecommerce orders (source: Postgres database)
bronze_ecommerce = spark.table("workspace.retail.bronze_ecommerce_orders")

# Transform ecommerce orders to unified schema
# Mapping: order_id -> transaction_id, order_date -> transaction_timestamp, etc.
ecommerce_unified = bronze_ecommerce.select(
    # Rename order_id to transaction_id for consistency across channels
    F.col("order_id").alias("transaction_id"),
    
    # customer_id links to customer master data
    # May be null for guest checkouts
    F.col("customer_id"),
    
    # product_id links to product master data
    F.col("product_id"),
    
    # Rename order_date to transaction_timestamp for consistency
    # Allows uniform time-based analytics across channels
    F.col("order_date").alias("transaction_timestamp"),
    
    # quantity purchased (line item level)
    F.col("quantity"),
    
    # Rename order_amount to gross_amount
    # gross_amount = quantity * unit_price before any discounts/taxes
    F.col("order_amount").alias("gross_amount"),
    
    # Rename order_status to transaction_status
    # Values: Completed, Pending, Cancelled, Refunded, Returned
    F.col("order_status").alias("transaction_status"),
    
    # channel identifies purchase origin (Ecommerce vs In-Store)
    F.col("channel"),
    
    # payment_method for payment analytics (Credit Card, PayPal, etc.)
    F.col("payment_method"),
    
    # Add null store fields for ecommerce (no physical store location)
    # This maintains schema consistency with POS data
    F.lit(None).cast("string").alias("store_id"),
    F.lit(None).cast("string").alias("store_name"),
    F.lit(None).cast("string").alias("store_city"),
    F.lit(None).cast("string").alias("store_state"),
    
    # Add source system for data lineage and troubleshooting
    F.lit("Postgres Ecommerce").alias("source_system")
)

# =============================================================================
# PART 2: Transform POS Transactions to Unified Schema
# =============================================================================

# Read Bronze POS transactions (source: In-store point-of-sale systems)
bronze_pos = spark.table("workspace.retail.bronze_pos_transactions")

# Transform POS transactions to unified schema
pos_unified = bronze_pos.select(
    # transaction_id already matches unified schema
    F.col("transaction_id"),
    
    # customer_id from loyalty card swipe or manual lookup
    # May be null for non-loyalty customers
    F.col("customer_id"),
    
    # product_id scanned at checkout
    F.col("product_id"),
    
    # Rename transaction_date to transaction_timestamp for consistency
    F.col("transaction_date").alias("transaction_timestamp"),
    
    # quantity purchased
    F.col("quantity"),
    
    # Rename transaction_amount to gross_amount
    F.col("transaction_amount").alias("gross_amount"),
    
    # transaction_status already matches unified schema
    F.col("transaction_status"),
    
    # channel = "In-Store" for all POS transactions
    F.col("channel"),
    
    # payment_method captured at POS
    F.col("payment_method"),
    
    # Include store location fields for regional analytics
    # Enables store performance tracking and geographic analysis
    F.col("store_id"),
    F.col("store_name"),
    F.col("store_city"),
    F.col("store_state"),
    
    # Add source system for data lineage
    F.lit("POS").alias("source_system")
)

# =============================================================================
# PART 3: Union Both Datasets and Add Business Flags
# =============================================================================

# Union ecommerce and POS data into single unified table
silver_transactions_unified = ecommerce_unified.union(pos_unified) \
    .filter(
        # DATA QUALITY FILTERS:
        # Remove invalid records that would skew analytics
        (F.col("product_id").isNotNull()) &  # Must have valid product
        (F.col("quantity") > 0) &             # Quantity must be positive
        (F.col("gross_amount") > 0)           # Amount must be positive
    ) \
    .withColumn(
        # BUSINESS FLAG: is_completed_transaction
        # Identifies revenue-generating transactions for financial reporting
        # Only "Completed" status contributes to revenue metrics
        "is_completed_transaction",
        F.when(F.col("transaction_status") == "Completed", True).otherwise(False)
    ) \
    .withColumn(
        # BUSINESS FLAG: is_refund_or_return
        # Tracks returns and refunds for customer satisfaction analysis
        # High return rates may indicate product quality or sizing issues
        "is_refund_or_return",
        F.when(F.col("transaction_status").isin("Refunded", "Returned", "Cancelled"), True).otherwise(False)
    ) \
    .withColumn(
        # BUSINESS FLAG: is_customer_identified
        # Differentiates known customers from guests
        # Guest transactions (null customer_id) limit personalization capabilities
        "is_customer_identified",
        F.when(F.col("customer_id").isNotNull(), True).otherwise(False)
    )

# Save as Delta table with overwrite mode
silver_transactions_unified.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.silver_transactions_unified")

# =============================================================================
# Validation and Business Metrics
# =============================================================================

count = silver_transactions_unified.count()
completed_count = silver_transactions_unified.filter("is_completed_transaction = true").count()
identified_count = silver_transactions_unified.filter("is_customer_identified = true").count()
ecommerce_count = silver_transactions_unified.filter("channel = 'Ecommerce'").count()
pos_count = silver_transactions_unified.filter("channel = 'In-Store'").count()

print(f"\n✅ Created silver_transactions_unified: {count:,} rows")
print(f"   📦 Ecommerce: {ecommerce_count:,} transactions")
print(f"   🏪 In-Store: {pos_count:,} transactions")
print(f"   ✓  Completed: {completed_count:,} ({(completed_count/count)*100:.1f}%)")
print(f"   👤 Identified customers: {identified_count:,} ({(identified_count/count)*100:.1f}%)")

print("\n📋 Schema:")
silver_transactions_unified.printSchema()

print("\n📊 Sample transactions by channel:")
display(silver_transactions_unified.orderBy(F.desc("transaction_timestamp")).limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 6
# =============================================================================
# SILVER TABLE 5: silver_customer_identity
# =============================================================================
# Business Purpose: Conformed customer identity with loyalty enrichment
# Transformations: Left join customers + loyalty, add defaults, segment customers
# 
# BUSINESS RATIONALE:
# Creates a "Customer 360" view by combining customer master data with loyalty
# program information. This conformed dimension enables:
# - Single customer view across all business functions
# - Customer segmentation for targeted marketing campaigns
# - Consistent customer metrics (tenure, loyalty tier, value tier)
# - Foundation for customer lifetime value (CLV) modeling
# 
# LEFT JOIN STRATEGY: Preserves all customers (members and non-members)
# - Loyalty members get actual tier, points, and status
# - Non-members get default values to maintain uniform schema
# =============================================================================

from pyspark.sql import functions as F

print("=" * 80)
print("CREATING SILVER TABLE 5: silver_customer_identity")
print("=" * 80)

# Read silver customers and loyalty profiles
silver_customers = spark.table("workspace.retail.silver_customers")
silver_loyalty = spark.table("workspace.retail.silver_loyalty_profiles")

# =============================================================================
# PART 1: Join Customers with Loyalty Profiles
# =============================================================================

# LEFT JOIN: Keep all customers, enrich with loyalty data where available
# Business rule: Every customer record is preserved, regardless of loyalty membership
silver_customer_identity = silver_customers.join(
    silver_loyalty,
    on="customer_id",
    how="left"  # LEFT JOIN preserves non-members
) \
.select(
    # =============================================================================
    # CUSTOMER MASTER FIELDS (from silver_customers)
    # =============================================================================
    F.col("customer_id"),  # Primary key
    F.col("full_name"),    # Display name
    F.col("email"),         # Primary contact method
    F.col("phone"),         # Secondary contact method
    F.col("registration_date"),  # When customer first registered
    
    # Geographic fields for regional segmentation
    F.col("city"),
    F.col("state"),
    F.col("country"),
    
    # METRIC: customer_tenure_days
    # Used for lifecycle stage analysis (new, growing, mature, at-risk)
    F.col("customer_tenure_days"),
    
    # =============================================================================
    # LOYALTY FIELDS WITH DEFAULTS (from silver_loyalty_profiles)
    # =============================================================================
    # For non-members (where join returns null), apply business defaults
    
    # loyalty_tier: Defaults to "Non-Member" for customers without loyalty profiles
    # Values: Bronze, Silver, Gold, Platinum, Non-Member
    F.coalesce(F.col("loyalty_tier"), F.lit("Non-Member")).alias("loyalty_tier"),
    
    # points_balance: Defaults to 0 for non-members
    # Enables uniform point-based calculations without null handling
    F.coalesce(F.col("points_balance"), F.lit(0)).alias("points_balance"),
    
    # member_since: Null for non-members (intentionally null to distinguish)
    F.col("member_since"),
    
    # loyalty_status: Defaults to "Non-Member" for non-members
    # Values for members: Active, Inactive, Suspended
    F.coalesce(F.col("status"), F.lit("Non-Member")).alias("loyalty_status"),
    
    # is_loyalty_member: Boolean flag for easy filtering
    # True = enrolled in loyalty program, False = not enrolled
    F.coalesce(F.col("is_loyalty_member"), F.lit(False)).alias("is_loyalty_member"),
    
    # Data quality flag from customer master
    F.col("is_customer_record_valid")
) \
.withColumn(
    # =============================================================================
    # BUSINESS SEGMENTATION: customer_segment
    # =============================================================================
    # Strategic customer segments based on loyalty tier and tenure
    # Used for prioritizing marketing spend and customer service resources
    "customer_segment",
    F.when(F.col("loyalty_tier") == "Platinum", "VIP")
     # VIP: Highest value customers, premium service and exclusive benefits
     
    .when(F.col("loyalty_tier") == "Gold", "High-Value")
     # High-Value: Strong customers, targeted for retention and upsell
     
    .when(F.col("loyalty_tier") == "Silver", "Growing")
     # Growing: Emerging value customers, nurture with engagement campaigns
     
    .when(F.col("loyalty_tier") == "Bronze", "Standard")
     # Standard: Entry-level loyalty members, opportunity for tier progression
     
    .when(
        (F.col("loyalty_tier") == "Non-Member") & (F.col("customer_tenure_days") > 365),
        "At-Risk Non-Member"
    )
     # At-Risk Non-Member: Long-time customer but never joined loyalty program
     # High priority for loyalty enrollment campaigns
     
    .when(
        (F.col("loyalty_tier") == "Non-Member") & (F.col("customer_tenure_days") <= 365),
        "New Non-Member"
    )
     # New Non-Member: Recently registered, target for welcome campaign
     # Opportunity to enroll before habits form
     
    .otherwise("Unknown")
)

# Save as Delta table with overwrite mode
silver_customer_identity.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.silver_customer_identity")

# =============================================================================
# Validation and Business Metrics
# =============================================================================

count = silver_customer_identity.count()
loyalty_members = silver_customer_identity.filter("is_loyalty_member = true").count()
non_members = silver_customer_identity.filter("is_loyalty_member = false").count()

print(f"\n✅ Created silver_customer_identity: {count:,} customers")
print(f"   👥 Loyalty members: {loyalty_members:,} ({(loyalty_members/count)*100:.1f}%)")
print(f"   👤 Non-members: {non_members:,} ({(non_members/count)*100:.1f}%)")

print("\n📊 Customer segmentation:")
display(silver_customer_identity.groupBy("customer_segment").count().orderBy(F.desc("count")))

print("\n📊 Loyalty tier distribution:")
display(silver_customer_identity.groupBy("loyalty_tier").count().orderBy(F.desc("count")))

print("\n📋 Schema:")
silver_customer_identity.printSchema()

print("\n📊 Sample VIP and High-Value customers:")
display(silver_customer_identity.filter("customer_segment IN ('VIP', 'High-Value')").orderBy(F.desc("points_balance")).limit(10))

print("\n" + "=" * 80)
print("✅ SILVER LAYER COMPLETE - ALL 5 TABLES CREATED")
print("=" * 80)