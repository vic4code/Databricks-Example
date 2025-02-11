# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper()  # Create the DA object

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"
    loan_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Select columns of interest and replace spaces with underscores
    loan_df = loan_df.selectExpr("ID", "Age", "Experience", "Income", "`ZIP Code` as ZIP_Code", "Family", "CCAvg", "Education", "Mortgage", "`Personal Loan` as Personal_Loan", "`Securities Account` as Securities_Account", "`CD Account` as CD_Account", "Online", "CreditCard")
    loan_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/loan-dataset")
    # Save df as delta table using Delta API
    #loan_df.write.format("delta").mode("overwrite").saveAsTable("bank_loan")


DBAcademyHelper.add_method(create_features_table)
class payload():
    def __init__(self, data):
        self.data = data
    def as_dict(self):
        return self.data

# COMMAND ----------

DA = DBAcademyHelper()  # Create the DA object
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()

# COMMAND ----------

from pyspark.sql.types import IntegerType, LongType, StringType, MapType, ArrayType, DoubleType, StructType, StructField, DateType
from pyspark.sql.functions import from_json, col, to_date

import pandas as pd
import csv
inferences_pdf = pd.read_csv(
    f'{DA.paths.datasets.monitoring}/monitoring/lab_inference_df.csv',
    header=0,
    sep=","
)



inferences_df = spark.createDataFrame(inferences_pdf)

# Define the schema for the request_metadata and response columns as a MapType or Array
metadata_schema = MapType(StringType(), StringType())
response_schema = StructType([

    StructField("predictions", ArrayType(DoubleType()))
])

# Correct type casting for columns
inferences_df = inferences_df.withColumn("client_request_id", col("client_request_id").cast(StringType()))
inferences_df = inferences_df.withColumn("databricks_request_id", col("databricks_request_id").cast(StringType()))
inferences_df = inferences_df.withColumn("timestamp_ms", col("timestamp_ms").cast(LongType()))
inferences_df = inferences_df.withColumn("status_code", col("status_code").cast(IntegerType()))
inferences_df = inferences_df.withColumn("execution_time_ms", col("execution_time_ms").cast(LongType()))
inferences_df = inferences_df.withColumn("sampling_fraction", col("sampling_fraction").cast(
DoubleType()))

# Parse the date column into DateType
inferences_df = inferences_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

# Ensure `request_metadata`, `request`, and `response` are treated as strings before parsing
inferences_df = inferences_df.withColumn("request_metadata", col("request_metadata").cast(StringType()))
inferences_df = inferences_df.withColumn("request", col("request").cast(StringType()))
inferences_df = inferences_df.withColumn("response", col("response").cast(StringType()))

# Parse the JSON string in `request_metadata` into a MapType
inferences_df = inferences_df.withColumn("request_metadata", from_json("request_metadata", metadata_schema))
inferences_df = inferences_df.withColumn("response", from_json("response", response_schema))

# Drop the existing Delta table if it exists to prevent schema conflicts
spark.sql(f"DROP TABLE IF EXISTS {DA.catalog_name}.{DA.schema_name}.model_inference_table")

# Save the DataFrame as a Delta table in the specified schema and catalog
inferences_df.write.format("delta").mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.lab_model_inference_table")

print("Inference table created successfully.")