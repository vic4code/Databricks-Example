# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper()  # Create the DA object

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, col

# Create function to initialize and create the features table
def create_features_table(self):
    # Load dataset path from working directory
    dataset_path = f"{DA.paths.datasets.cdc_diabetes}/cdc-diabetes/diabetes_binary_5050split_BRFSS2015.csv"
    diabetes_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Convert all columns to double type
    for column in diabetes_df.columns:
        diabetes_df = diabetes_df.withColumn(column, col(column).cast("double"))

    # Add a unique ID column
    diabetes_df = diabetes_df.withColumn("id", monotonically_increasing_id())

    # Save the dataset as a Delta table
    diabetes_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/diabetes-dataset")

# add method the helper function
DBAcademyHelper.add_method(create_features_table)
class payload():
    def __init__(self, data):
        self.data = data
    def as_dict(self):
        return self.data

# COMMAND ----------

import time
import re
import io
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, pandas_udf, transform, size, element_at


def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumn("timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", F.array(F.element_at(F.col("request"), -1)))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded

# COMMAND ----------

DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()

# COMMAND ----------

from pyspark.sql.types import IntegerType, LongType, StringType, MapType, ArrayType, DoubleType, StructType, StructField, DateType
from pyspark.sql.functions import from_json, col, to_date

import pandas as pd
# Load the CSV file directly into a Spark DataFrame
inferences_pdf = pd.read_csv(
    f'{DA.paths.datasets.monitoring}/monitoring/inference_table.csv',
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
inferences_df = inferences_df.withColumn("sampling_fraction", col("sampling_fraction").cast(DoubleType()))

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
inferences_df.write.format("delta").mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.model_inference_table")

print("Inference table created successfully.")