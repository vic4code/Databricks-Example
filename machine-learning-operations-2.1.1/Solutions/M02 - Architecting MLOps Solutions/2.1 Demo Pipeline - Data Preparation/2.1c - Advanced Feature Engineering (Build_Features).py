# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 3 - Advanced Feature Engineering
# MAGIC
# MAGIC This notebook is the third task in our MLOps pipeline workflow. Here, we will load the transformed loan data, create additional complex features, validate the engineered features, and save the advanced feature-engineered dataset for model training.
# MAGIC
# MAGIC **Objectives:**
# MAGIC - Load the transformed loan dataset from Task 2.
# MAGIC - Create additional features that can enhance model performance, including debt-to-income ratio, total assets indicator, and credit usage category.
# MAGIC - Validate the data schema and sample output to ensure correctness.
# MAGIC - Save the feature-engineered dataset in Delta format for efficient storage and retrieval in subsequent pipeline steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-2.1a

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
# MAGIC
# MAGIC ## Task Outline:
# MAGIC In this task, we will enhance the loan data further by adding complex features for improved model training.
# MAGIC
# MAGIC - **Load the Transformed Loan Data:** Load the data saved from Task 2 for advanced feature engineering.
# MAGIC - **Create Additional Complex Features:** Add new features to capture important relationships, such as debt-to-income ratio, total assets, and credit usage.
# MAGIC - **Validate and Inspect the Data:** Check the schema and sample output for accuracy.
# MAGIC - **Save the Engineered Dataset:** Store the dataset in Delta format, making it accessible for model training in the next stage.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1: Load the Transformed Loan Data
# MAGIC In this step, we load the transformed loan dataset saved from Task 2. This will serve as the base dataset for advanced feature engineering.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the path to the transformed data from Task 2.
# MAGIC - Use the Spark DataFrame API to load the data and display a sample to confirm it loaded correctly.

# COMMAND ----------

# Define the path to the transformed loan data from Task 2
transformed_data_path = f"{DA.catalog_name}.{DA.schema_name}.transformed_loan_data"

# Load the transformed data
loan_df = spark.read.table(transformed_data_path)

# Display the first few rows to inspect the data
display(loan_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Create Additional Complex Features
# MAGIC Here, we’ll create more sophisticated features, such as debt-to-income ratio, total assets indicator, and a credit usage category. These features will provide additional insights to improve model accuracy.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Use sample data modifications to ensure varied results in engineered features.
# MAGIC - Define new columns for debt-to-income ratio, total assets, and credit usage category based on conditions and calculations.

# COMMAND ----------

from pyspark.sql.functions import col, expr, when

# Display the initial schema for reference
loan_df.printSchema()

# Set sample rows to ensure diverse results in `debt_to_income_ratio` and `credit_usage_category`
loan_df = loan_df.withColumn("Income", when(col("ID") == "1", 20000)
                             .when(col("ID") == "2", 30000)
                             .when(col("ID") == "3", 15000)
                             .when(col("ID") == "4", 50000)
                             .when(col("ID") == "5", 45000)
                             .otherwise(col("Income").cast("double")))

loan_df = loan_df.withColumn("Mortgage", when(col("ID") == "1", 25000)
                             .when(col("ID") == "2", 18000)
                             .when(col("ID") == "3", 50000)
                             .when(col("ID") == "4", 10000)
                             .when(col("ID") == "5", 15000)
                             .otherwise(col("Mortgage").cast("double")))

loan_df = loan_df.withColumn("CCAvg", when(col("ID") == "1", 4.0)
                             .when(col("ID") == "2", 1.5)
                             .when(col("ID") == "3", 3.2)
                             .when(col("ID") == "4", 5.0)
                             .when(col("ID") == "5", 2.0)
                             .otherwise(col("CCAvg").cast("double")))

# Calculate debt-to-income ratio
loan_df = loan_df.withColumn("debt_to_income_ratio", 
                             when((col("Income") > 0) & (col("Mortgage") > 0), (col("Mortgage") / col("Income")) * 100)
                             .otherwise(0))

# Calculate total assets indicator
loan_df = loan_df.withColumn("total_assets", 
                             expr("CASE WHEN Securities_Account = '1' OR CD_Account = '1' THEN 1 ELSE 0 END"))

# Assign credit usage category based on Income and CCAvg
loan_df = loan_df.withColumn("credit_usage_category", 
                             when((col("CCAvg") > 3.0) & (col("Income") < 50000), "High Usage, Low Income")
                             .when((col("CCAvg") <= 3.0) & (col("Income") >= 50000), "Low Usage, High Income")
                             .otherwise("Moderate"))

# Display sample of rows to verify output
display(loan_df.select("ID", "Income", "Mortgage", "CCAvg", "debt_to_income_ratio", "total_assets", "credit_usage_category"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Validate and Inspect the Data
# MAGIC After creating these features, we validate the schema and inspect sample rows to confirm everything is correct and ready for model training.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Print the schema to verify new columns and data types.
# MAGIC - Display sample rows to inspect the values of the newly engineered features.

# COMMAND ----------

# Verify new columns by printing the schema
loan_df.printSchema()

# Display a sample of rows to check the engineered features
display(loan_df.select("Income", "debt_to_income_ratio", "total_assets", "credit_usage_category"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Persist the Engineered Dataset for Model Training
# MAGIC Finally, save the feature-engineered dataset in Delta format. This dataset will be used directly for model training in the next stage.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Convert column names to lowercase to avoid case sensitivity issues.
# MAGIC - Define the path to save the engineered data in Delta format.
# MAGIC - Drop any existing Delta table with the same name and save the DataFrame to Delta format.

# COMMAND ----------

from pyspark.sql.functions import col
from delta.tables import DeltaTable

# Standardize column names to lower case to avoid case sensitivity issues
loan_df = loan_df.toDF(*[col_name.lower() for col_name in loan_df.columns])

# Define the Delta table path
feature_engineered_data_path = f"{DA.catalog_name}.{DA.schema_name}.feature_engineered_loan_data"

# Drop the existing Delta table if it exists
if DeltaTable.isDeltaTable(spark, feature_engineered_data_path):
    spark.sql(f"DROP TABLE IF EXISTS {feature_engineered_data_path}")

# Save the updated DataFrame to Delta format
loan_df.write.format("delta").mode("overwrite").saveAsTable(feature_engineered_data_path)

print(f"Feature-engineered loan data saved to: {feature_engineered_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded the transformed loan data from Task 2.
# MAGIC - Created additional complex features, such as debt-to-income ratio, total assets indicator, and credit usage category, to enrich the dataset.
# MAGIC - Validated the schema and inspected sample data to confirm correctness.
# MAGIC - Saved the feature-engineered dataset in Delta format, preparing it for direct use in model training in the next stage of the MLOps pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>