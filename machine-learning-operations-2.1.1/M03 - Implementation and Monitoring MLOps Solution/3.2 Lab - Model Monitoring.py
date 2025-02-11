# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Model Monitoring
# MAGIC In this notebook, you will monitor the performance of a deployed machine learning model using Databricks. You will enable an inference table, send batched requests, and set up comprehensive monitoring.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _In this lab, you will need to complete the following tasks:_
# MAGIC - **Task 1:** Save the Training Data as Reference for Drift
# MAGIC - **Task 2:** Processing and Monitoring Inference Data
# MAGIC > - **2.1:** Monitoring the Inference Table  
# MAGIC > - **2.2:** Processing Inference Table Data 
# MAGIC > - **2.3:** Analyzing Processed Requests
# MAGIC - **Task 3:** Persisting Processed Model Logs
# MAGIC - **Task 4:** Setting Up and Monitoring Inference Data
# MAGIC > - **4.1:** Creating an Inference Monitor with Databricks Lakehouse Monitoring
# MAGIC > - **4.2:** Inspect and Monitor Metrics Tables
# MAGIC
# MAGIC
# MAGIC 📝 **Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): **`15.4.x-cpu-ml-scala2.12`**

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC 1. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC   - In the drop-down, select **More**.
# MAGIC   - In the **Attach to an existing compute resource** pop-up, select the first drop-down. You will see a unique cluster name in that drop-down. Please select that cluster.
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC 1. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC 1. Wait a few minutes for the cluster to start.
# MAGIC 1. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the Lab. Execute the following cell:
# MAGIC
# MAGIC 🚨 **_Please wait for the classroom setup to run, as it may take around 10 minutes to execute and create a model that you will be using for the Lab._**
# MAGIC
# MAGIC > **🚨Note:** If you encounter a "file not found" error when running this cell, simply re-run it. This issue may occur as we transition from DBFS to Unity Catalog.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.Lab

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load Data
# MAGIC Load the banking dataset into a Pandas DataFrame and prepare it for training and requesting.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType

# Load the Delta table into a Spark DataFrame
dataset_path = loan_pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Train / New Requests Split
# MAGIC Split the data into training and request sets.
# MAGIC
# MAGIC

# COMMAND ----------

# Split the data into train and request sets
train_df, request_df = loan_df.randomSplit(weights=[0.6, 0.4], seed=42)

# Convert to Pandas DataFrames
train_pd_df = train_df.toPandas()
request_pd_df = request_df.toPandas()
target_col = "Personal_Loan"
ID = "ID"

X_request = request_df.drop(target_col)
y_request = request_df.select(target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define Model Name and Display the Model Serving Endpoint
# MAGIC - Set the model name for registration in the Databricks Model Registry.
# MAGIC
# MAGIC - Display the Model Serving Endpoint URL for easy access.

# COMMAND ----------

model_name = f"{DA.catalog_name}.{DA.schema_name}.loan_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Save the Training Data as Reference for Drift
# MAGIC  
# MAGIC Save the training data as a Delta table to serve as a reference for detecting data drift.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Convert Data:** Convert the Pandas DataFrame to a Spark DataFrame.
# MAGIC 2. **Save Data:** Save the Spark DataFrame as a Delta table.
# MAGIC 3. **Read and Update Data:** Read the Delta table and update the data types to match the required schema.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import lit, col
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame, functions as F, types as T

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(train_pd_df).withColumn('model_id', lit(0)).withColumn("labeled_data", col("Personal_Loan").cast(DoubleType()))

# Save the spark_df as baseline_features table
(spark_df
 <FILL_IN>)

# Read the existing table into a DataFrame
baseline_features_df = <FILL_IN>

# Cast the labeled_data and CCAvg columns to INT
baseline_features_df = <FILL_IN>

# Overwrite the existing table with the updated DataFrame
(baseline_features_df
  <FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Monitoring and Processing Inference Data 
# MAGIC This task involves processing the logged data, monitoring the inference table, and analyzing the processed requests to ensure continuous availability and accuracy of model performance data.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1: Monitoring the Inference Table
# MAGIC In this task, you will monitor the inference table to ensure that data is populating correctly as the model starts receiving requests.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Monitor Table:** Monitor the inference table to ensure data is populating correctly.
# MAGIC   2. **Check Table:** Implement a loop to wait and check for the table population, retrying until the data appears.
# MAGIC

# COMMAND ----------


# Attempt to read the table
inference_df = spark.read.table(f"{DA.catalog_name}.{DA.schema_name}.lab_model_inference_table")
        
# Check if the table is not empty
if inference_df.count() > 0:
    # If successful and the table is not empty, display the DataFrame and break the loop
    <FILL_IN>
else:
    # If the table Not Found, print table Not Found
    print("Table is empty, trying again in 10 seconds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.2: Processing Inference Table Data
# MAGIC In this task, you will extract and analyze the data logged in the inference table to prepare it for monitoring.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Define Functions:** Define helper functions for JSON conversion.
# MAGIC   2. **Process Requests:** Process the raw requests and unpack JSON payloads.
# MAGIC   3. **Convert Data:** Convert timestamps and explode batched requests into individual rows.
# MAGIC

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T
import json
import pandas as pd

"""
Conversion helper functions.
"""
def convert_to_record_json(json_str: str) -> str:
    """
    Converts records from the four accepted JSON formats for Databricks
    Model Serving endpoints into a common, record-oriented
    DataFrame format which can be parsed by the PySpark function from_json.
    
    :param json_str: The JSON string containing the request or response payload.
    :return: A JSON string containing the converted payload in record-oriented format.
    """
    try:
        request = json.<FILL_IN>
    except json.JSONDecodeError:
        return <FILL_IN>
    output = []
    if isinstance(request, dict):
        obj_keys = set(request.keys())
        if "dataframe_records" in obj_keys:
            # Record-oriented DataFrame
            <FILL_IN>
        elif "dataframe_split" in obj_keys:
            # Split-oriented DataFrame
            dataframe_split = request["dataframe_split"]
            <FILL_IN>
        elif "instances" in obj_keys:
            # TF serving instances
            <FILL_IN>
        elif "inputs" in obj_keys:
            # TF serving inputs
            <FILL_IN>
        elif "predictions" in obj_keys:
            # Predictions
            <FILL_IN>
        return json.dumps(output)
    else:
        # Unsupported format, pass through
        <FILL_IN>


@F.pandas_udf(T.StringType())
def json_consolidation_udf(json_strs: pd.Series) -> pd.Series:
    """A UDF to apply the JSON conversion function to every request/response."""
    <FILL_IN>

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.types import TimestampType 

def process_requests(requests_raw: DataFrame) -> DataFrame:
    """
    Processes a stream of raw requests and:
        - Unpacks JSON payloads for requests
        - Extracts relevant features as scalar values (first element of each array)
        - Converts Unix epoch millisecond timestamps to Spark TimestampType
    """
    # Calculate the current timestamp in seconds
    current_ts = int(spark.sql("SELECT unix_timestamp(<FILL_IN>)").collect()[0][0])

    # Define the start timestamp for 30 days ago
    start_ts = current_ts - 30 * 24 * 60 * 60  # 30 days in seconds

    # Dynamically calculate the min and max values of timestamp_ms
    min_max = requests_raw.agg(
        F.min("timestamp_ms").alias("min_ts"),
        F.max("timestamp_ms").alias("max_ts")
    ).collect()[0]

    min_ts = <FILL_IN>  # Convert from milliseconds to seconds
    max_ts = <FILL_IN>  # Convert from milliseconds to seconds

    # Transform timestamp_ms to span the last month
    requests_timestamped = requests_raw.withColumn(
        'timestamp', 
        (start_ts + ((F.col(<FILL_IN>) / 1000 - min_ts) / (max_ts - min_ts)) * (current_ts - start_ts)).cast(TimestampType())
    ).drop("timestamp_ms")

    # Consolidate and unpack JSON.
    requests_unpacked = requests_timestamped \
        .withColumn("request", json_consolidation_udf(F.col("request"))) \
        .withColumn('request', F.from_json(F.col("request"), F.schema_of_json('[{"ID": 1.0,"Age": 40.0, "Experience": 10.0, "Income": 84.0, "ZIP_Code": 9302.0, "Family": 3.0, "CCAvg": 2.0, "Education": 2.0, "Mortgage": 0.0, "Securities_Account": 0.0, "CD_Account": 0.0, "Online": 1.0, "CreditCard": 1.0}]'))) \
        .withColumn("response", F.expr("transform(response.predictions, x -> x)"))  \
        .withColumn('response', F.col("response"))

    # Explode batched requests into individual rows.
    DB_PREFIX = "__db"
    requests_exploded = requests_unpacked \
        .withColumn(f"{DB_PREFIX}_request_response", F.arrays_zip(F.col("request"), F.col("response"))) \
        .withColumn(f"{DB_PREFIX}_request_response", F.explode(F.col(f"{DB_PREFIX}_request_response"))) \
        .select(F.col("*"), F.col(f"{DB_PREFIX}_request_response.request.*"), F.col(f"{DB_PREFIX}_request_response.response").alias("Personal_Loan")) \
        .drop(f"{DB_PREFIX}_request_response", "request", "response") \
        .withColumn('model_id', F.lit(<FILL_IN>))

    return requests_exploded

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3: Analyzing Processed Requests
# MAGIC
# MAGIC After processing and unpacking the logged data from the inference table, the next step is to analyze the requests that were successfully answered by the model, filtering and joining with additional label data for comprehensive analysis.
# MAGIC
# MAGIC **Steps:**
# MAGIC   1. **Process Data:** Filter and analyze the requests that were successfully answered by the model.
# MAGIC   2. **Join Data:** Join with additional label data for comprehensive analysis.
# MAGIC

# COMMAND ----------

# Process the inference data
model_logs_df = process_requests(<FILLIN>) # Let's ignore bad requests

# Ensure the ID column is added during processing and Display the model_logs_df
<FILL_IN>

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
loan_spark_df = spark.createDataFrame(<loan_pd_df>)

# Rename 'Personal_Loan' to 'labeled_data' in the loan_spark_df
label_spark_df = <FILL_IN>

# Join with model_logs_df
model_logs_df_labeled = model_logs_df.join(
    <FILL_IN>
)

# Display the joined DataFrame
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Persisting Processed Model Logs
# MAGIC
# MAGIC In this task, you will save the enriched model logs to ensure long-term availability for ongoing monitoring and analysis.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. **Convert Data Types:** Convert all columns to appropriate data types.
# MAGIC 2. **Save Logs:** Save the processed logs to a designated storage for long-term use.
# MAGIC 3. **Enable CDF:** Enable Change Data Feed (CDF) to facilitate efficient incremental processing of new data.
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F, types as T

# Convert all columns to IntegerType except those of type MAP<STRING, STRING>
for col_name, col_type in model_logs_df_labeled.dtypes:
    <FILL_IN>
        )

# Overwrite the existing table with the updated DataFrame
model_logs_df_labeled.write.format("delta") \
    <FILL_IN>")

# COMMAND ----------

# MAGIC %md
# MAGIC For efficient execution, enable CDF (Change Data Feeed) so monitoring can incrementally process the data.

# COMMAND ----------

spark.sql(f'ALTER TABLE {DA.catalog_name}.{DA.schema_name}.lab_model_logs SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Setting Up and Monitoring Inference Data
# MAGIC
# MAGIC This task includes setting up the monitoring of inference data and ensuring its continuous availability for analysis and monitoring.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.1: Creating an Inference Monitor with Databricks Lakehouse Monitoring
# MAGIC Set up monitoring to continuously track model performance and detect any anomalies or drift in real-time.
# MAGIC **Steps:**
# MAGIC   1. **Configure Monitor:** Configure and initiate the monitoring of model logs.
# MAGIC   2. **Create Monitor:** Create an inference monitor and validate its creation.
# MAGIC   3. **Verify Metrics:** Verify that metrics tables are created and populated.
# MAGIC

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()
table_name = f'{DA.catalog_name}.{DA.schema_name}.model_logs'
baseline_table_name = f"{DA.catalog_name}.{DA.schema_name}.baseline_features"

# ML problem type, either "classification" or "regression"
PROBLEM_TYPE = MonitorInferenceLogProblemType.<FILL_IN>

# Window sizes to analyze data over
GRANULARITIES = ["<FILL_IN>"]

# Directory to store generated dashboard
ASSETS_DIR = f"/Workspace/Users/{DA.username}/databricks_lakehouse_monitoring/model_logs"
# Optional parameters
SLICING_EXPRS = ["<FILL_IN>"]   # Expressions to slice data with
print(f"Creating monitor for model_logs")

info = w.quality_monitors.create(
  table_name=<FILL_IN>,
  inference_log=MonitorInferenceLog(
    <FILL_IN>
  ),
  baseline_table_name=baseline_table_name,
  slicing_exprs=SLICING_EXPRS,
  output_schema_name=f"{DA.catalog_name}.{DA.schema_name}",
  assets_dir=ASSETS_DIR
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Instructions for Accessing Your Monitor Dashboard
# MAGIC
# MAGIC Follow these steps to ensure that your quality monitor is correctly set up and to access the monitor dashboard:
# MAGIC
# MAGIC 1. **Wait for Monitor Creation**: Start by initiating a loop to check the monitor’s status. The loop will query the monitor’s status every 10 seconds until it changes from *pending* to *active*, confirming that the monitor has been successfully created.
# MAGIC
# MAGIC 2. **Check for Metric Refreshes**: Once the monitor is created, it automatically triggers a metric refresh. Ensure that this refresh process is completed successfully.
# MAGIC
# MAGIC 3. **Monitor Refresh Status**: Identify the first refresh operation and enter a loop to check its state. If the state is either *pending* or *running*, wait for 30 seconds before checking again. This loop will continue until the refresh state changes to *success*, confirming the refresh operation has completed successfully.
# MAGIC
# MAGIC 4. **Access the Monitor Dashboard**: Once the monitor is active and metrics have been refreshed, a URL to the monitor dashboard will be constructed using the workspace URL, catalog name, schema name, and specifying the `quality` tab. This URL is printed out, allowing you direct access to the dashboard where you can review quality metrics for your model logs.
# MAGIC
# MAGIC By following these steps, you can ensure your quality monitor is set up correctly and use the provided URL to conveniently access the dashboard to monitor your model's performance and data quality.
# MAGIC

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.<FILL_IN>
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"
# A metric refresh will automatically be triggered on creation
refreshes = w.quality_monitors.<FILL_IN>
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.<FILL_IN>
  time.sleep(30)

assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"

w.quality_monitors.<FILL_IN>
# Extract workspace URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Construct the monitor dashboard URL
monitor_dashboard_url = f"https://{workspace_url}/explore/data/{DA.catalog_name}/{DA.schema_name}/model_logs?o={DA.schema_name}&activeTab=quality"

print(f"Monitor Dashboard URL: {monitor_dashboard_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.2: Inspect and Monitor Metrics Tables
# MAGIC
# MAGIC In this task, you will learn how to inspect and monitor the metrics tables generated by the Databricks quality monitoring tools. These tables provide valuable insights into the performance and behavior of your models, including summary statistics and data drift detection.
# MAGIC
# MAGIC You will perform the following:
# MAGIC - **Inspect the Metrics Tables Using UI**: Locate and review the profile and drift metrics tables created by the monitoring process. These tables are saved in your default database and provide detailed metrics and visualizations.

# COMMAND ----------

# MAGIC %md
# MAGIC **Inspect the Metrics Tables Using UI**
# MAGIC
# MAGIC
# MAGIC By default, the metrics tables are saved in the default database.
# MAGIC
# MAGIC The `create_monitor` call created two new tables: the profile metrics table and the drift metrics table.
# MAGIC
# MAGIC - **Profile Metrics Table**: This table records summary statistics for each column in the monitored table.
# MAGIC - **Drift Metrics Table**: This table records metrics that compare current values in the monitored table to baseline values, identifying potential drift.
# MAGIC
# MAGIC These tables use the same name as the primary table to be monitored, with the suffixes `_profile_metrics` and `_drift_metrics`.
# MAGIC
# MAGIC > **Instructions:**
# MAGIC > 1. Go to the Table where the monitor is created: `(Table name=f'{DA.catalog_name}.{DA.schema_name}.model_logs')`.
# MAGIC > 2. Check the output tables:
# MAGIC >    - Locate the table with the suffix `_profile_metrics` to view summary statistics for each column.
# MAGIC >    - Locate the table with the suffix `_drift_metrics` to view metrics that compare current values to baseline values.
# MAGIC > 3. View the dashboard associated with these metrics tables.
# MAGIC > 4. Explore the different metrics and visualizations created. The dashboard provides insights into data distribution, potential data drift, and other key metrics.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore the dashboard!

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you successfully deployed a machine learning model and set up a monitoring framework to track its performance. You sent batched requests to the model endpoint and monitored the responses to detect any anomalies or drift. Additionally, you explored how to use Databricks Lakehouse Monitoring to continuously track and alert on model performance metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>