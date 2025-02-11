# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Feature Engineering with Feature Store
# MAGIC
# MAGIC Welcome to the "Feature Engineering with Feature Store" In this lesson, you will learn how to load and prepare a dataset for feature selection, explore and manipulate a feature table through Databricks UI, perform feature selection on specific columns, create a new feature table, access feature table details using both UI and API, merge two feature tables based on a common identifier, and efficiently delete unnecessary feature tables. Get ready to enhance your feature engineering skills—let's dive in!
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this Lab, you will learn how to:
# MAGIC
# MAGIC 1. Load and Prepare Dataset for Feature Selection
# MAGIC 2. Explore Feature Table through UI
# MAGIC 3. Access Feature Table Information
# MAGIC 4. Create Feature Table from Existing UC Table
# MAGIC 5. Enhance Feature Table with New Features
# MAGIC 6. Efficient Feature Table Deletion

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **16.0.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC 1. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC   - In the drop-down, select **More**.
# MAGIC   - In the **Attach to an existing compute resource** pop-up, select the first drop-down. You will see a unique cluster name in that drop-down. Please select that cluster.
# MAGIC   
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC 1. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC 1. Wait a few minutes for the cluster to start.
# MAGIC 1. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.2

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
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC

# COMMAND ----------

# Set the path of the dataset
shared_volume_name = 'cdc-diabetes' # From Marketplace
csv_name = 'diabetes_binary_5050split_BRFSS2015' # CSV file name
dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" # Full path
silver_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
display(silver_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task1: Feature Selection
# MAGIC
# MAGIC The dataset is loaded and ready. We are assuming that most of the data cleaning and feature computation is already done and data is saved to "silver" table.
# MAGIC
# MAGIC Select these features from the dataset; **"HighBP", "HighChol", "BMI", "Stroke", "PhysActivity", "GenHlth", "Sex", "Age", "Education", "Income". **
# MAGIC
# MAGIC Create a `UID` column to be used as primary key.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# select features we are interested in
silver_df = <FILL_IN>

# drop the target column
silver_df = <FILL_IN>

# create an UID column to be used as primary key
silver_df = <FILL_IN>

display(silver_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 2: Create a Feature Table
# MAGIC
# MAGIC
# MAGIC Create a feature table from the `silver_df` dataset. Define description and tags as you wish.
# MAGIC
# MAGIC New feature table name must be **`diabetes_features`**.
# MAGIC
# MAGIC **Note:** Don't define partition column.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = <FILL_IN>

diabetes_table_name = <FILL_IN>

fe.create_table(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 3: Explore Feature Table with the UI
# MAGIC
# MAGIC Now that the feature table is created, visit **Features** page from the left panel and review following information;
# MAGIC
# MAGIC * Check table columns, identify **primary key** and **partition** columns.
# MAGIC
# MAGIC * View **sample data**.
# MAGIC
# MAGIC * View table **details**. 
# MAGIC
# MAGIC * View **history**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Retrieve Feature Table Details
# MAGIC
# MAGIC Another way of accessing the feature table is using the API. Let's **list `features` and `primary_keys`** of the table.

# COMMAND ----------

ft = fe.<FILL_IN>
print(f"Features: {ft.<FILL_IN>}")
print(f"Primary Keys: {ft.<FILL_IN>}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Create a Feature Table from an Existing UC Table
# MAGIC
# MAGIC There is a table already created for you which includes diet related features. The table name is **`diet_features`**. Create a feature table for this existing table.

# COMMAND ----------

display(spark.sql("SELECT * FROM diet_features"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- set UID column to not null
# MAGIC <FILL_IN>
# MAGIC
# MAGIC -- set UID column as primary key constraint
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Add New Features to Existing Table
# MAGIC
# MAGIC Let's collect diet features and merge them to the existing `diabetes_features` table. As both tables has `UID` as unique identifier, we will merge them based on this column.

# COMMAND ----------

diet_features = spark.sql("SELECT * FROM diet_features")

# Update diabetes feature table by adding diet features table
fe.<FILL_IN>

# Read and display the merged feature table
display(fe.<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7: Delete a Feature Table
# MAGIC
# MAGIC We merged both feature tables and we no longer need the `diet_features` table. Thus, let's delete this table.

# COMMAND ----------

diet_table_name = f"{DA.catalog_name}.{DA.schema_name}.diet_features"

# drop the table
fe.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you demonstrated the use of Databricks Feature Store to perform feature engineering tasks. You executed the loading, preparation, and selection of features from a dataset, created a feature table, explored and accessed table details through both the UI and API, merged tables, and efficiently removed unnecessary ones. 
# MAGIC
# MAGIC This hands-on experience enhanced your feature engineering skills on the Databricks platform.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>