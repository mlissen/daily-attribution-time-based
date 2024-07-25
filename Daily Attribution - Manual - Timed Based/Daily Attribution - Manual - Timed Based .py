import sys
import os
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import to_date, lit, when, avg, sum, round, concat_ws, from_unixtime, unix_timestamp, row_number, expr, min, max, coalesce, explode, col, to_timestamp, from_unixtime
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.utils import AnalysisException
from datetime import datetime, timedelta
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, DecimalType

# Initialize Spark and Glue contexts
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session
job = Job(glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'START_DATE', 'END_DATE'])
job.init(args['JOB_NAME'], args)

# Get the GlueLogger
logger = glueContext.get_logger()

# Parse start and end dates
start_date = datetime.strptime(args['START_DATE'], '%Y-%m-%d')
end_date = datetime.strptime(args['END_DATE'], '%Y-%m-%d')

logger.info(f"LISS Job started. Processing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

def format_time_for_predicate_days(start_dt, end_dt):
    return f"(year * 10000 + month * 100 + day) >= {start_dt.year * 10000 + start_dt.month * 100 + start_dt.day} AND (year * 10000 + month * 100 + day) <= {end_dt.year * 10000 + end_dt.month * 100 + end_dt.day}"
    
def log_sample_data(df, description, num_samples=2):
    sample_data = df.limit(num_samples).collect()
    for i, row in enumerate(sample_data):
        logger.info(f"LISS {description} sample row {i+1}: {row}")

# Function to load data using predicates
def load_data_with_predicates(database, table_name, start_date, end_date):
    predicate = format_time_for_predicate_days(start_date, end_date)
    logger.info(f"LISS Loading data from {database}.{table_name} with predicate: {predicate}")
    df = glueContext.create_dynamic_frame.from_catalog(
        database=database,
        table_name=table_name,
        push_down_predicate=predicate
    ).toDF()
    row_count = df.count()
    logger.info(f"LISS Loaded {row_count} rows from {database}.{table_name}")
    logger.info(f"LISS Columns in {table_name}: {df.columns}")
    
    # Log sample data
    sample_data = df.limit(2).collect()
    for row in sample_data:
        logger.info(f"LISS {table_name} sample row data: {row}")
    
    return df

# Helper function to check if a DataFrame is empty
def check_empty_df(df, description):
    if df.rdd.isEmpty():
        logger.warn(f"LISS Warning: No data found for {description}. Exiting job.")
        job.commit()
        os._exit(0)  # Exit the job gracefully
    return False

# Load sales data
sales_start_date = start_date - timedelta(days=60)
sales_data = load_data_with_predicates("aggregatedsales", "aggregatedsales", sales_start_date, end_date)
logger.info(f"LISS Sales data loaded. Row count: {sales_data.count()}")
check_empty_df(sales_data, "sales data")

# Add sales_date column
sales_data = sales_data.withColumn(
    "sales_date",
    to_date(concat_ws("-", col("year"), col("month"), col("day")), "yyyy-MM-dd")
)

# Load ad traffic data
ad_traffic_start_date = start_date - timedelta(days=60)
placements_data = load_data_with_predicates("adtraffic", "winningadplacements", ad_traffic_start_date, end_date)
logger.info(f"LISS Ad traffic data loaded. Row count: {placements_data.count()}")
check_empty_df(placements_data, "ad traffic data")


#Adding date columns
sales_data = sales_data.withColumn("date", to_date(col("timestamp")))
placements_data = placements_data.withColumn("date", to_date(from_unixtime(col("begintime") / 1000)))

sales_row_count = sales_data.count()
placements_row_count = placements_data.count()

if sales_row_count > 0:
    earliest_sales_date = sales_data.select(min("date")).first()[0] if "date" in sales_data.columns else "N/A"
    latest_sales_date = sales_data.select(max("date")).first()[0] if "date" in sales_data.columns else "N/A"
    logger.info(f"LISS Earliest date in sales_data: {earliest_sales_date}")
    logger.info(f"LISS Latest date in sales_data: {latest_sales_date}")

if placements_row_count > 0:
    earliest_placements_date = placements_data.select(min("date")).first()[0] if "date" in placements_data.columns else "N/A"
    latest_placements_date = placements_data.select(max("date")).first()[0] if "date" in placements_data.columns else "N/A"
    logger.info(f"LISS Earliest date in placements_data: {earliest_placements_date}")
    logger.info(f"LISS Latest date in placements_data: {latest_placements_date}")
    
# Log earliest and latest dates
earliest_sales_date = sales_data.select(min("date")).first()[0]
earliest_ad_traffic_date = placements_data.select(min("date")).first()[0]
latest_sales_date = sales_data.select(max("date")).first()[0]
latest_ad_traffic_date = placements_data.select(max("date")).first()[0]

logger.info(f"LISS Earliest sales date: {earliest_sales_date}")
logger.info(f"LISS Earliest ad traffic date: {earliest_ad_traffic_date}")
logger.info(f"LISS Latest sales date: {latest_sales_date}")
logger.info(f"LISS Latest ad traffic date: {latest_ad_traffic_date}")

# Process winning bid columns
try:
    placements_data = placements_data.withColumn("winningbidamount",
        when(col("winningbidamount.double").isNotNull(), col("winningbidamount.double"))
        .when(col("winningbidamount.int").isNotNull(), col("winningbidamount.int").cast("double"))
        .otherwise(None)
    )
except AnalysisException:
    placements_data = placements_data.withColumn("winningbidamount", col("winningbidamount").cast("double"))

try:
    placements_data = placements_data.withColumn("winningbidprice",
        when(col("winningbidprice.double").isNotNull(), col("winningbidprice.double"))
        .when(col("winningbidprice.int").isNotNull(), col("winningbidprice.int").cast("double"))
        .otherwise(None)
    )
except AnalysisException:
    placements_data = placements_data.withColumn("winningbidprice", col("winningbidprice").cast("double"))

# Separate explode and cast operations
placements_data = placements_data.withColumn("placement_tracked_skus", explode("trackedskus"))
placements_data = placements_data.withColumn("placement_tracked_skus", col("placement_tracked_skus").cast("string"))

# Log all column names after processing
logger.info(f"LISS Columns in placements_data after processing: {placements_data.columns}")

# Cast columns to appropriate types
sales_data = sales_data.withColumn("sale_amount", when(col("sale_amount.double").isNotNull(), col("sale_amount.double")).when(col("sale_amount.int").isNotNull(), col("sale_amount.int").cast("double")).otherwise(None))

sales_data = sales_data.withColumnRenamed("year", "sales_year") \
                       .withColumnRenamed("month", "sales_month") \
                       .withColumnRenamed("day", "sales_day")

placements_data = placements_data.withColumnRenamed("year", "placement_year") \
                                 .withColumnRenamed("month", "placement_month") \
                                 .withColumnRenamed("day", "placement_day")


log_sample_data(placements_data, "Placements data")

log_sample_data(sales_data, "Sales data")

# Loading PostgreSQL data
def load_postgres_data(table_name):
    df = glueContext.create_dynamic_frame.from_options(connection_type="postgresql",
        connection_options={"useConnectionProperties": "true",
        "dbtable": table_name,
        "connectionName": "Postgresql connection2"}).toDF()
    logger.info(f"LISS Loaded {df.count()} rows from PostgreSQL table: {table_name}")
    logger.info(f"LISS Columns in {table_name}: {df.columns}")
    return df

categories_data_df = load_postgres_data("categories")
campaigns_data_df = load_postgres_data("campaigns")
organizations_data_df = load_postgres_data("organizations")
location_data_df = load_postgres_data("locations")
ad_group_data_df = load_postgres_data("ad_groups")
ads_data_df = load_postgres_data("ads")

def get_historical_baseline(current_date, sales_data, ad_traffic_data):
    logger.info(f"LISS Calculating historical baseline for date: {current_date}")
    historical_end_date = current_date - timedelta(days=1)
    historical_start_date = historical_end_date - timedelta(days=59)
    historical_sales = sales_data.filter((col("date") >= historical_start_date) & (col("date") <= historical_end_date))
    historical_ad_traffic = ad_traffic_data.filter((col("date") >= historical_start_date) & (col("date") <= historical_end_date))
    
    logger.info(f"LISS Historical sales count: {historical_sales.count()}")
    logger.info(f"LISS Historical ad traffic count: {historical_ad_traffic.count()}")
    
    sales_without_ad_traffic = historical_sales.join(
        historical_ad_traffic,
        (historical_sales.sku == historical_ad_traffic.placement_tracked_skus) &
        (historical_sales.date == historical_ad_traffic.date) &
        (historical_sales.location_id == historical_ad_traffic.locationid),
        "leftanti"
    )
    
    logger.info(f"LISS Sales without ad traffic count: {sales_without_ad_traffic.count()}")
    
    window_spec = Window.partitionBy("sku").orderBy(col("date").desc())
    avg_sales_no_ad_traffic = sales_without_ad_traffic.withColumn("row_number", row_number().over(window_spec)).filter(col("row_number") <= 3).groupBy("sku").agg(avg("units_sold").alias("avg_units_sold_no_ad_traffic"), avg("sale_amount").alias("avg_sales_no_ad_traffic"))
    
    logger.info(f"LISS Average sales without ad traffic count: {avg_sales_no_ad_traffic.count()}")
    
    return avg_sales_no_ad_traffic
    
# Main job loop
current_date = start_date
logger.info(f"LISS Starting main processing loop from {current_date} to {end_date}")
while current_date <= end_date:
    logger.info(f"LISS Processing data for date: {current_date}")
    
    baseline_sales = get_historical_baseline(current_date, sales_data, placements_data)
    if baseline_sales.rdd.isEmpty():
        logger.warn(f"LISS No baseline sales data for date: {current_date}")
        current_date += timedelta(days=1)
        continue

    current_sales = sales_data.filter(col("date") == current_date)
    current_ad_traffic = placements_data.filter(col("date") == current_date)
    
    logger.info(f"LISS Current sales count: {current_sales.count()}")
    logger.info(f"LISS Current ad traffic count: {current_ad_traffic.count()}")
    
    # Log column names for debugging
    logger.info(f"LISS Current sales columns: {current_sales.columns}")
    logger.info(f"LISS Current ad traffic columns: {current_ad_traffic.columns}")
    
    attributed_data = current_sales.join(current_ad_traffic, 
                                         (current_sales.sku == current_ad_traffic.placement_tracked_skus) & 
                                         (current_sales.location_id == current_ad_traffic.locationid), 
                                         "inner")
    attributed_data = attributed_data.withColumn("attribution_type", lit("time_based"))
    logger.info(f"LISS Attributed data count: {attributed_data.count()}")
    
    # Log column names for debugging
    logger.info(f"LISS Attributed data columns: {attributed_data.columns}")
    
    
    # Convert timestamp string to timestamp type
    attributed_data = attributed_data.withColumn("timestamp", to_timestamp("timestamp"))
    
    # Convert begintime and endtime from milliseconds to timestamp
    attributed_data = attributed_data.withColumn("begintime", from_unixtime(col("begintime") / 1000))
    attributed_data = attributed_data.withColumn("endtime", from_unixtime(col("endtime") / 1000))
    
    log_sample_data(attributed_data, "Attributed data:")
    
    # Now apply the filter
    attributed_data = attributed_data.filter(
        ((col("begintime") <= col("timestamp")) & 
         (col("endtime") >= col("timestamp"))) |
        (col("endtime") >= col("timestamp") - expr("INTERVAL 2 HOURS"))
    )

    logger.info(f"LISS Filtered attributed data count: {attributed_data.count()}")

    
    if attributed_data.rdd.isEmpty():
        logger.warn(f"LISS No attributed data for date: {current_date}")
        current_date += timedelta(days=1)
        continue
    
    window_spec = Window.partitionBy(attributed_data.order_id, attributed_data.sku, attributed_data.location_id).orderBy(attributed_data.endtime.desc())
    attributed_data = attributed_data.withColumn("row_number", row_number().over(window_spec)).filter(col("row_number") == 1)
    logger.info(f"LISS Final attributed data count: {attributed_data.count()}")
    
    log_sample_data(attributed_data, "Final attributed data:")


    # Join with categories data and select necessary columns
    categories_data_df = categories_data_df.withColumnRenamed("name", "category_name")
    attributed_data_w_names1 = attributed_data.join(
        categories_data_df,
        placements_data["aislecategoryid"] == categories_data_df["id"],
        "left"
    )
    
    # Select only necessary columns after join
    categories_join_selected = attributed_data_w_names1.select(
        attributed_data_w_names1["*"],
        col("category_name")
    )
    
    # Perform similar steps for campaigns data
    campaigns_data_df = campaigns_data_df.withColumnRenamed("name", "campaign_name")
    joined_with_campaign = categories_join_selected.join(
        campaigns_data_df,
        categories_join_selected["campaignid"] == campaigns_data_df["id"],
        "left"
    )
    
    # Log distinct campaign IDs from categories_join_selected
    distinct_campaign_ids = categories_join_selected.select("campaignid").distinct().collect()
    logger.info(f"LISS Distinct campaign IDs: {[row['campaignid'] for row in distinct_campaign_ids]}")
    
    # Log distinct IDs from campaigns_data_df
    distinct_ids_from_campaigns = campaigns_data_df.select("id").distinct().collect()
    logger.info(f"LISS Distinct IDs in campaigns_data_df: {[row['id'] for row in distinct_ids_from_campaigns]}")
    
    # Handle empty DataFrame after join
    if joined_with_campaign.rdd.isEmpty():
        logger.warn("LISS campaign Join operation resulted in an empty DataFrame.")
    else:
        logger.info("LISS campaign Join operation successful.")
    
    # Perform similar steps for organizations data
    organizations_data_df = organizations_data_df.withColumnRenamed("name", "organization_name")
    joined_with_organizations = joined_with_campaign.join(
        organizations_data_df,
        joined_with_campaign["retailerorgid"] == organizations_data_df["id"],
        "left"
    )
    
    # Join with locations data, focusing on 'name' column only
    location_data_df = location_data_df.withColumnRenamed("name", "location_name")
    locations_join = joined_with_organizations.join(
        location_data_df,
        joined_with_organizations["locationid"] == location_data_df["id"],
        "left"
    ).select(
        joined_with_organizations["*"],
        location_data_df["location_name"]
    )
    
    # Join with ad groups data, focusing on 'name' column only
    ad_group_data_df = ad_group_data_df.withColumnRenamed("name", "ad_group_name")
    
    try:
        ad_groups_join = locations_join.join(
            ad_group_data_df,
            locations_join["adgroupid"] == ad_group_data_df["id"],
            "left"
        ).select(
            locations_join["*"],
            ad_group_data_df["ad_group_name"]
        )
    
    except Exception as e:
        logger.error(f"LISS Error occurred during ad group join: {str(e)}")
        raise e
    
    # Join with ads data, focusing on 'name' column only
    ads_data_df = ads_data_df.withColumnRenamed("name", "ad_name")
    
    try:
        enriched_attributed_data = ad_groups_join.join(
            ads_data_df,
            ad_groups_join["winningadid"] == ads_data_df["id"],
            "left"
        ).select(
            ad_groups_join["*"],
            ads_data_df["ad_name"]
        )
    
    except Exception as e:
        logger.error(f"LISS Error occurred during ads join: {str(e)}")
        raise e
        
    log_sample_data(enriched_attributed_data, "Enriched atrributed data:")

    # Join with baseline sales data
    final_data = enriched_attributed_data.join(baseline_sales, "sku", "left") \
                                         .withColumn("incremental_units_sold", when(col("units_sold") - col("avg_units_sold_no_ad_traffic") > 0, col("units_sold") - col("avg_units_sold_no_ad_traffic")).otherwise(0)) \
                                         .withColumn("incremental_sales", when(col("sale_amount") - col("avg_sales_no_ad_traffic") > 0, col("sale_amount") - col("avg_sales_no_ad_traffic")).otherwise(0))

    log_sample_data(final_data, "Final data:")
    
    final_data = final_data.withColumn(
            "sales_date",
            to_date(concat_ws("-", col("sales_year"), col("sales_month"), col("sales_day")), "yyyy-MM-dd")
        )

    
    # Aggregate metrics for time-based attribution
    time_based_conversions = final_data.groupBy(
        col("organization_name").alias("retailer_name"),
        col("advertiserOrgId").alias("advertiser_org_id"),
        col("daily_budget"),
        col("status"),
        col("campaign_type"),
        col("campaign_name"),
        col("adProduct").alias("ad_product"),
        col("ad_group_name"),
        col("ad_name"),
        col("state"),
        col("city"),
        col("location_name"),
        col("category_name").alias("aisle_category_name"),
        col("sku"),
        col("product_name"),
        col("winningCurrencyCode").alias("winning_currency_code"),
        col("campaignId").alias("campaign_id"),
        col("adGroupId").alias("adgroup_id"),
        col("offerCode").alias("offer_code"),
        col("locationid").alias("location"),
        col("sales_year"),  # Add this line
        col("sales_month"), # Add this line
        col("sales_day")    # Add this line
    ).agg(
        sum("incremental_units_sold").alias("units_sold"),
        round(sum("incremental_sales"), 2).alias("sales")
    )

    time_based_conversions = time_based_conversions.withColumn(
        "units_sold", col("units_sold").cast("integer")
    ).withColumn(
        "sales", col("sales").cast("decimal(10,2)")
    )
    
    time_based_conversions = time_based_conversions.withColumn(
    "daily_budget", col("daily_budget").cast(DecimalType(10,2))
    )
    
    # Add the sales_date column after the aggregation
    time_based_conversions = time_based_conversions.withColumn("attribution_method", lit("time_based_incremental"))

    # Log sample data after aggregation
    log_sample_data(time_based_conversions, "Time-based conversions after aggregation")

    # Converting back to DynamicFrame for output
    final_dyf = DynamicFrame.fromDF(time_based_conversions, glueContext, "final_dyf")

    # Writing results to S3
    output_path = "s3://attribution-results-098573017733-us-west-2/daily_ad_conversions_aggregate/"
    glueContext.write_dynamic_frame.from_options(
        frame=final_dyf, 
        connection_type="s3", 
        connection_options={"path": output_path, "partitionKeys": ["sales_year", "sales_month", "sales_day"]}, 
        format="parquet"
    )

    logger.info(f"LISS Data for {current_date} processed and written to S3")

    # Move to the next date
    current_date += timedelta(days=1)

# End of while loop
logger.info("LISS Job completed successfully")
job.commit()