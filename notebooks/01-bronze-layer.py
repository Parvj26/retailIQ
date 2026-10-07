# Databricks notebook source
# DBTITLE 1,Setup: Create Retail Schema
# =============================================================================
# BRONZE LAYER SETUP - RetailIQ Customer Intelligence Platform
# =============================================================================
# Purpose: Create the retail schema to house all Bronze layer tables
# Catalog: workspace | Schema: retail
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import random

# Set catalog and create schema
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.retail COMMENT 'RetailIQ Bronze Layer - Raw synthetic retail data'")
spark.sql("USE CATALOG workspace")
spark.sql("USE SCHEMA retail")

print("✅ Schema 'workspace.retail' created successfully")
print(f"📍 Current location: {spark.sql('SELECT current_catalog(), current_schema()').collect()[0]}")

# COMMAND ----------

# DBTITLE 1,Bronze Table 1: Product Catalog
# =============================================================================
# BRONZE TABLE: Product Catalog
# =============================================================================
# Business Purpose: Master product reference data for retail operations
# Contains: SKU, product name, category, brand, price
# Usage: Enriches orders and transactions with product attributes
# =============================================================================

import random
from pyspark.sql import Row

# Define product categories and brands for realistic retail data
categories = [
    "Electronics", "Apparel", "Home & Garden", "Beauty & Personal Care", 
    "Sports & Outdoors", "Toys & Games", "Food & Beverage", "Health & Wellness"
]

brands = [
    "GlobalTech", "StyleCo", "HomeEssentials", "PureBeauty", "ActiveLife",
    "PlayZone", "FreshMarket", "WellnessPlus", "UrbanWear", "SmartHome"
]

# Generate 200 products with realistic pricing based on category
products_data = []
for i in range(1, 201):
    category = random.choice(categories)
    
    # Set price ranges based on category (realistic retail pricing)
    if category == "Electronics":
        price = round(random.uniform(49.99, 999.99), 2)
    elif category == "Apparel":
        price = round(random.uniform(19.99, 199.99), 2)
    elif category == "Home & Garden":
        price = round(random.uniform(24.99, 499.99), 2)
    else:
        price = round(random.uniform(9.99, 149.99), 2)
    
    products_data.append(
        Row(
            product_id=f"SKU{i:05d}",
            product_name=f"{random.choice(brands)} {category} Item {i}",
            category=category,
            brand=random.choice(brands),
            price=price
        )
    )

# Create DataFrame and save as Delta table
products_df = spark.createDataFrame(products_data)

products_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.bronze_products")

print(f"✅ Created bronze_products table with {products_df.count()} products")
print("\n📊 Sample products:")
display(products_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Bronze Table 2: Customers
# =============================================================================
# BRONZE TABLE: Customers
# =============================================================================
# Business Purpose: Master customer records from PostgreSQL ecommerce database
# Contains: Customer ID, name, email, phone, registration date, location
# Usage: Foundation for Customer 360 and loyalty program linking
# =============================================================================

from datetime import datetime, timedelta
import random

# Customer attributes for synthetic data generation
first_names = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"
]

cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Seattle", "Denver", "Boston", "Portland", "Miami", "Atlanta"
]

states = [
    "NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA",
    "TX", "FL", "WA", "CO", "MA", "OR", "FL", "GA"
]

# Generate 5,000 customers registered over the past 3 years
customers_data = []
base_date = datetime.now() - timedelta(days=1095)  # 3 years ago

for i in range(1, 5001):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    
    # Create realistic email addresses
    email_domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com"])
    email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@{email_domain}"
    
    # Generate registration date (older customers more likely to be loyal)
    days_since_registration = random.randint(0, 1095)
    registration_date = base_date + timedelta(days=days_since_registration)
    
    # Select random city and corresponding state
    city_idx = random.randint(0, len(cities) - 1)
    
    customers_data.append(
        Row(
            customer_id=f"CUST{i:06d}",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            registration_date=registration_date.date(),
            city=cities[city_idx],
            state=states[city_idx],
            country="USA"
        )
    )

# Create DataFrame and save as Delta table
customers_df = spark.createDataFrame(customers_data)

customers_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.bronze_customers")

print(f"✅ Created bronze_customers table with {customers_df.count()} customers")
print("\n📊 Sample customers:")
display(customers_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Bronze Table 3: Loyalty Profiles
# =============================================================================
# BRONZE TABLE: Loyalty Profiles
# =============================================================================
# Business Purpose: Customer loyalty program data from MySQL loyalty database
# Contains: Customer ID, loyalty tier, points balance, member since date
# Usage: Enriches Customer 360 with engagement tier and lifetime value signals
# =============================================================================

import random
from datetime import datetime, timedelta

# Loyalty tiers with realistic distribution
# 60% Bronze, 25% Silver, 12% Gold, 3% Platinum (mimics real loyalty programs)
loyalty_tiers = ["Bronze"] * 60 + ["Silver"] * 25 + ["Gold"] * 12 + ["Platinum"] * 3

# Generate loyalty profiles for 70% of customers (not all customers join loyalty program)
num_loyalty_customers = 3500
eligible_customer_ids = [f"CUST{i:06d}" for i in range(1, 5001)]
random.shuffle(eligible_customer_ids)
loyalty_customer_ids = eligible_customer_ids[:num_loyalty_customers]

loyalty_data = []
for customer_id in loyalty_customer_ids:
    # Extract customer number to determine registration timing
    cust_num = int(customer_id.replace("CUST", ""))
    
    # Loyalty members join 30-180 days after initial registration
    base_registration = datetime.now() - timedelta(days=1095)
    days_offset = (cust_num / 5000) * 1095  # Proportional to customer age
    loyalty_join_date = base_registration + timedelta(days=days_offset) + timedelta(days=random.randint(30, 180))
    
    # Assign loyalty tier (newer members more likely Bronze, older members higher tiers)
    if days_offset > 730:  # Customers older than 2 years
        tier_pool = ["Bronze"] * 40 + ["Silver"] * 30 + ["Gold"] * 20 + ["Platinum"] * 10
    elif days_offset > 365:  # 1-2 years
        tier_pool = ["Bronze"] * 50 + ["Silver"] * 30 + ["Gold"] * 15 + ["Platinum"] * 5
    else:  # Less than 1 year
        tier_pool = ["Bronze"] * 75 + ["Silver"] * 20 + ["Gold"] * 4 + ["Platinum"] * 1
    
    tier = random.choice(tier_pool)
    
    # Points balance correlates with tier
    if tier == "Bronze":
        points = random.randint(0, 4999)
    elif tier == "Silver":
        points = random.randint(5000, 14999)
    elif tier == "Gold":
        points = random.randint(15000, 49999)
    else:  # Platinum
        points = random.randint(50000, 150000)
    
    loyalty_data.append(
        Row(
            customer_id=customer_id,
            loyalty_tier=tier,
            points_balance=points,
            member_since=loyalty_join_date.date(),
            status="Active"  # Could add "Inactive" for churned members in real scenario
        )
    )

# Create DataFrame and save as Delta table
loyalty_df = spark.createDataFrame(loyalty_data)

loyalty_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.bronze_loyalty_profiles")

print(f"✅ Created bronze_loyalty_profiles table with {loyalty_df.count()} loyalty members")
print(f"📈 Loyalty penetration: {(loyalty_df.count() / 5000) * 100:.1f}% of customer base")
print("\n📊 Loyalty tier distribution:")
display(loyalty_df.groupBy("loyalty_tier").count().orderBy("loyalty_tier"))

# COMMAND ----------

# DBTITLE 1,Bronze Table 4: Ecommerce Orders
# =============================================================================
# BRONZE TABLE: Ecommerce Orders
# =============================================================================
# Business Purpose: Online order transactions from PostgreSQL ecommerce database
# Contains: Order ID, customer, product, quantity, amount, timestamp, channel
# Usage: Revenue attribution, purchase frequency analysis, churn prediction signals
# =============================================================================

import random
from datetime import datetime, timedelta

# Generate ecommerce orders over the past 2 years
# Distribution: More recent orders, varying frequency by customer segment

num_orders = 25000
ecommerce_orders_data = []

# Get product IDs and prices for order line items
product_ids = [f"SKU{i:05d}" for i in range(1, 201)]

# Generate orders with realistic patterns
for i in range(1, num_orders + 1):
    # Select customer (some customers order more frequently than others)
    # 20% of customers generate 80% of orders (Pareto principle)
    if random.random() < 0.8:
        customer_id = f"CUST{random.randint(1, 1000):06d}"  # Top 20% customers
    else:
        customer_id = f"CUST{random.randint(1001, 5000):06d}"  # Remaining 80% customers
    
    # Order date: weighted toward recent (more orders in recent months)
    days_ago = int(random.expovariate(1/180))  # Exponential distribution, avg 180 days
    if days_ago > 730:  # Cap at 2 years
        days_ago = 730
    order_date = datetime.now() - timedelta(days=days_ago)
    
    # Select product and quantity
    product_id = random.choice(product_ids)
    quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]  # Most orders 1 item
    
    # Calculate order amount (product price * quantity + variation)
    base_amount = random.uniform(29.99, 299.99) * quantity
    
    # Apply occasional discounts (20% of orders)
    if random.random() < 0.2:
        discount_pct = random.choice([0.10, 0.15, 0.20, 0.25])
        order_amount = base_amount * (1 - discount_pct)
    else:
        order_amount = base_amount
    
    # Order status (95% completed, 5% cancelled/returned)
    order_status = random.choices(
        ["Completed", "Cancelled", "Returned"],
        weights=[95, 3, 2]
    )[0]
    
    ecommerce_orders_data.append(
        Row(
            order_id=f"EC{i:08d}",
            customer_id=customer_id,
            product_id=product_id,
            order_date=order_date,
            quantity=quantity,
            order_amount=round(order_amount, 2),
            order_status=order_status,
            channel="Ecommerce",
            payment_method=random.choice(["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay"])
        )
    )

# Create DataFrame and save as Delta table
ecommerce_df = spark.createDataFrame(ecommerce_orders_data)

ecommerce_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.bronze_ecommerce_orders")

print(f"✅ Created bronze_ecommerce_orders table with {ecommerce_df.count()} orders")
print(f"💰 Total ecommerce revenue: ${ecommerce_df.filter('order_status = "Completed"').agg({'order_amount': 'sum'}).collect()[0][0]:,.2f}")
print("\n📊 Sample orders:")
display(ecommerce_df.orderBy(F.desc("order_date")).limit(5))

# COMMAND ----------

# DBTITLE 1,Bronze Table 5: POS Transactions
# =============================================================================
# BRONZE TABLE: POS Transactions
# =============================================================================
# Business Purpose: In-store point-of-sale transactions from retail locations
# Contains: Transaction ID, customer (when loyalty card used), product, amount, store
# Usage: Omnichannel behavior analysis, store performance, customer journey mapping
# =============================================================================

import random
from datetime import datetime, timedelta

# Store locations for POS transactions
store_locations = [
    {"store_id": "ST001", "store_name": "RetailIQ Flagship - Manhattan", "city": "New York", "state": "NY"},
    {"store_id": "ST002", "store_name": "RetailIQ Downtown - LA", "city": "Los Angeles", "state": "CA"},
    {"store_id": "ST003", "store_name": "RetailIQ Mall - Chicago", "city": "Chicago", "state": "IL"},
    {"store_id": "ST004", "store_name": "RetailIQ Plaza - Houston", "city": "Houston", "state": "TX"},
    {"store_id": "ST005", "store_name": "RetailIQ Center - Phoenix", "city": "Phoenix", "state": "AZ"},
    {"store_id": "ST006", "store_name": "RetailIQ Square - Boston", "city": "Boston", "state": "MA"},
    {"store_id": "ST007", "store_name": "RetailIQ Market - Seattle", "city": "Seattle", "state": "WA"},
    {"store_id": "ST008", "store_name": "RetailIQ Hub - Miami", "city": "Miami", "state": "FL"},
]

# Generate 35,000 POS transactions (higher volume than ecommerce)
num_pos_transactions = 35000
pos_transactions_data = []

# Get product IDs for transactions
product_ids = [f"SKU{i:05d}" for i in range(1, 201)]

for i in range(1, num_pos_transactions + 1):
    # Transaction date: last 2 years, with seasonal peaks
    days_ago = random.randint(0, 730)
    transaction_date = datetime.now() - timedelta(days=days_ago)
    
    # 60% of in-store transactions are from loyalty members (identified customers)
    # 40% are anonymous walk-ins
    if random.random() < 0.6:
        # Loyalty member transaction - link to customer
        customer_id = f"CUST{random.randint(1, 5000):06d}"
    else:
        # Anonymous transaction - no customer ID
        customer_id = None
    
    # Select store location
    store = random.choice(store_locations)
    
    # Select product and quantity (in-store tends to have smaller basket sizes)
    product_id = random.choice(product_ids)
    quantity = random.choices([1, 2, 3], weights=[70, 25, 5])[0]
    
    # Calculate transaction amount
    base_amount = random.uniform(19.99, 199.99) * quantity
    
    # In-store discounts less frequent (10% of transactions)
    if random.random() < 0.1:
        discount_pct = random.choice([0.10, 0.15, 0.20])
        transaction_amount = base_amount * (1 - discount_pct)
    else:
        transaction_amount = base_amount
    
    # Transaction status (98% completed, 2% refunded)
    transaction_status = random.choices(
        ["Completed", "Refunded"],
        weights=[98, 2]
    )[0]
    
    pos_transactions_data.append(
        Row(
            transaction_id=f"POS{i:08d}",
            customer_id=customer_id,
            product_id=product_id,
            transaction_date=transaction_date,
            quantity=quantity,
            transaction_amount=round(transaction_amount, 2),
            transaction_status=transaction_status,
            channel="In-Store",
            store_id=store["store_id"],
            store_name=store["store_name"],
            store_city=store["city"],
            store_state=store["state"],
            payment_method=random.choice(["Credit Card", "Debit Card", "Cash", "Mobile Pay"])
        )
    )

# Create DataFrame and save as Delta table
pos_df = spark.createDataFrame(pos_transactions_data)

pos_df.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable("workspace.retail.bronze_pos_transactions")

print(f"✅ Created bronze_pos_transactions table with {pos_df.count()} transactions")
print(f"💰 Total in-store revenue: ${pos_df.filter('transaction_status = "Completed"').agg({'transaction_amount': 'sum'}).collect()[0][0]:,.2f}")
print(f"🏪 Identified customers: {(pos_df.filter('customer_id IS NOT NULL').count() / pos_df.count()) * 100:.1f}%")
print("\n📊 Sample transactions:")
display(pos_df.orderBy(F.desc("transaction_date")).limit(5))