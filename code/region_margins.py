# Databricks notebook source
# MAGIC %md
# MAGIC # [Marcelino Mayorga Quesada](https://marcelinomayorga.com/)

# COMMAND ----------

# DBTITLE 1,Imports
# Imports

from pyspark.sql.functions import col, expr
from pyspark.sql.window import Window
from pyspark.sql.functions import lag



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 0 - Background:
# MAGIC
# MAGIC - Databricks deploys its products in multiple regions.
# MAGIC - Each region generates revenue and has dedicated cloud resource footprint.
# MAGIC - We measure the health of each region / product line by tracking the margins
# MAGIC - Margin = (Revenue - Cost)/(Cost)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1 - Load Region Product Margin Data
# MAGIC

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/data.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Rename columns
df = df.withColumnRenamed("Region", "region")
df = df.withColumnRenamed("Product Line", "product_line")
df = df.withColumnRenamed("Month", "month")
df = df.withColumnRenamed("Cost (M$)", "cost_m_usd")
df = df.withColumnRenamed("Revevue (M$)", "revenue_m_usd")

display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Table Column Description
# MAGIC

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Table Summary
# MAGIC
# MAGIC Observations:
# MAGIC
# MAGIC - 2 regions: us-west1 and us-east1
# MAGIC - Data scoped for Y2023
# MAGIC - No missing values
# MAGIC - Cost and Revenue columns are double representing Millions and USD

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2 - Add Margin (Performance)
# MAGIC
# MAGIC $$\text{Margin} = \frac{\text{Revenue} - \text{Cost}}{\text{Cost}}$$
# MAGIC

# COMMAND ----------

df = df.withColumn("margin", expr("(revenue_m_usd - cost_m_usd) / cost_m_usd"))
display(df.sample(False,0.5))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 - Dashboards

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1 - Global month over month margin trend (margin_mom_change)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Observations:
# MAGIC
# MAGIC - Early Year Peaked performance with 0.25 margin MoM.
# MAGIC - Flat Negative margin MoM from February through June with a slight uptick in July and trending upwards and Positive in August.
# MAGIC
# MAGIC Actions:
# MAGIC - Identify Trends and Seasonality between February and June.
# MAGIC - Understand what drives&nbsp;January's healthy&nbsp;performance&nbsp;and August's uptrend.
# MAGIC

# COMMAND ----------

window_spec = Window.orderBy("month")
df_mom_dashboard = df.select(*df.columns)
df_mom_dashboard = df_mom_dashboard.withColumn("margin_mom_lag", lag("margin").over(window_spec))
df_mom_dashboard = df_mom_dashboard.withColumn("margin_mom", col("margin") - col("margin_mom_lag"))
display(df_mom_dashboard)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Per region month over month margin trend
# MAGIC
# MAGIC Observations
# MAGIC
# MAGIC - us-east1 peaked margin in January, hitting negative in May, and zero margin in the months in between.
# MAGIC - us-west1 fluctuated in negatives and did not reach a positive margin.
# MAGIC
# MAGIC Actions
# MAGIC
# MAGIC - Understand what drives us-east1&nbsp;January's healthy margin&nbsp;and remaining stability.
# MAGIC - Understand us-west1 overall unhealthy margins.&nbsp;&nbsp;
# MAGIC

# COMMAND ----------

window_spec = Window.partitionBy("region").orderBy("month")
df_dashboard = df.select(*df.columns)
df_dashboard = df_dashboard.withColumn("region_month_margin_mom_lag", lag("margin").over(window_spec))
df_dashboard = df_dashboard.withColumn("region_month_margin_mom", col("margin") - col("region_month_margin_mom_lag"))
 
display(df_dashboard)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Per product line month over month margin trend
# MAGIC
# MAGIC Observations:
# MAGIC - Both Product lines show a similar trend with SQL showing higher revenue than Jobs.
# MAGIC - Both Product lines share flat 0 revenue in May and June.
# MAGIC - SQL Product Line trends downward slower than Jobs.

# COMMAND ----------

window_spec = Window.partitionBy("product_line").orderBy("month")
df_dashboard = df.select(*df.columns)
df_dashboard = df_dashboard.withColumn("product_line_month_margin_mom_lag", lag("margin").over(window_spec))
df_dashboard = df_dashboard.withColumn("product_line_month_margin_mom", col("margin") - col("product_line_month_margin_mom_lag"))
display(df_dashboard)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Per region, per product line month over month margin trend
# MAGIC
# MAGIC Observations:
# MAGIC -
# MAGIC

# COMMAND ----------

product_line_window_spec = Window.partitionBy("product_line").orderBy("month")
region_window_spec = Window.partitionBy("region").orderBy("month")
df_dashboard = df.select(*df.columns)
df_dashboard = df_dashboard.withColumn("product_line_month_margin_mom_lag", lag("margin").over(product_line_window_spec))
df_dashboard = df_dashboard.withColumn("region_month_margin_mom_lag", lag("margin").over(region_window_spec))
df_dashboard = df_dashboard.withColumn("product_line_month_margin_mom", col("margin") - col("product_line_month_margin_mom_lag"))
df_dashboard = df_dashboard.withColumn("region_month_margin_mom", col("margin") - col("region_month_margin_mom_lag"))
display(df_dashboard)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. An easy visualization that shows best & worst performing regions & product lines.
# MAGIC

# COMMAND ----------

display(df_mom_dashboard)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save results into table

# COMMAND ----------

table_name = 'region_margin'
dbutils.fs.rm("dbfs:/user/hive/warehouse/"+table_name,True)
spark.sql("DROP TABLE IF EXISTS " + table_name)
df.write.format("parquet").saveAsTable(table_name)

# COMMAND ----------

# Validate SQL Table

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC REFRESH TABLE region_margin;
# MAGIC select * from region_margin;

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC
# MAGIC - For MOM: Evaluate details for Seasonality and negative revenue from **Feburary - July**
# MAGIC - For region: Evaluate details for Seasonality in **January - February**
# MAGIC - For Product line: Similar trends while SQL product line performs better than Job product line
# MAGIC - For Product line x Region MOM: East is more profitable than West in Jobs product line, while West is more profitable than East in SQL product line.
# MAGIC
# MAGIC # Suggested Actions
# MAGIC - Apply forecasting for margin 
# MAGIC - Automate data workflow for preprocessing, dashboard generation, and Multistep Margin Forecasting (Recursive)
# MAGIC - Identify Anomalies, Trends, Seasonality, Special Events & Data Leakage
# MAGIC
