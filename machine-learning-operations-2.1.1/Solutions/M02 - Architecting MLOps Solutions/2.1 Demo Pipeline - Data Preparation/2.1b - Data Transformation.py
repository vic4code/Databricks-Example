# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 2 - Data Transformation
# MAGIC
# MAGIC This notebook is the second task in our MLOps pipeline workflow. In this task, we will load the cleaned loan data, perform additional feature engineering, and save the transformed dataset for further steps in the pipeline.
# MAGIC
# MAGIC **Objectives:**
# MAGIC - Load the cleaned loan dataset from Task 1.
# MAGIC - Engineer new features to enhance the dataset for machine learning purposes.
# MAGIC - Validate the schema of the transformed data to ensure compatibility with downstream tasks.
# MAGIC - Save the transformed dataset in Delta format for optimized storage and retrieval.

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
# MAGIC In this task, we will follow a structured approach to enrich the loan data by engineering new features.
# MAGIC
# MAGIC 1. **Load the Cleaned Loan Dataset:** Load the dataset saved in Task 1 for further transformations.
# MAGIC - Feature Engineering: Add new features, such as income tiers and credit card spending categories, to enhance the dataset.
# MAGIC 2. **Schema and Data Validation:** Ensure the new features are added correctly and data integrity is maintained.
# MAGIC 3. **Save Transformed Data:** Save the transformed dataset in Delta format to ensure accessibility for subsequent steps in the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1: Load the Cleaned Loan Dataset
# MAGIC In this step, we will load the cleaned loan dataset saved in Task 1 to continue with feature engineering.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the path to the cleaned data from Task 1.
# MAGIC - Use the Spark DataFrame API to load the data and display the first few rows to verify it has been loaded correctly.

# COMMAND ----------

# Define the path to the cleaned loan data (from Notebook 1)
cleaned_data_path = f"{DA.catalog_name}.{DA.schema_name}.cleaned_loan_data"

# Load the cleaned data
loan_df = spark.read.table(cleaned_data_path)

# Display the first few rows
display(loan_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Feature Engineering
# MAGIC In this step, we will add new features to enhance the dataset for model training. These engineered features can improve model accuracy and interpretability.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Create new columns that categorize income levels and credit card spending, based on thresholds.
# MAGIC - Verify the transformations by displaying the dataset with the new features.

# COMMAND ----------

from pyspark.sql.functions import when, col

# Display the initial DataFrame schema for reference
loan_df.printSchema()

# Feature Engineering: categorize Income into tiers
loan_df = loan_df.withColumn("income_tier", 
                             when(col("Income") < 50000, "Low") \
                             .when((col("Income") >= 50000) & (col("Income") < 100000), "Medium") \
                             .otherwise("High"))

# Feature Engineering: categorize Credit Card Average (CCAvg) into spending categories
loan_df = loan_df.withColumn("cc_spending_category", 
                             when(col("CCAvg").cast("double") < 2, "Low") \
                             .when((col("CCAvg").cast("double") >= 2) & (col("CCAvg").cast("double") < 5), "Medium") \
                             .otherwise("High"))

# Display the transformed DataFrame with new features
display(loan_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Schema and Data Validation
# MAGIC To ensure that the new features have been added correctly and data integrity is maintained, we will validate the schema and display sample rows.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Verify the data types of the new columns by printing the schema.
# MAGIC - Display sample rows to inspect the values in the newly engineered columns.

# COMMAND ----------

# Verify new columns by printing the schema
loan_df.printSchema()

# Display sample rows to check the engineered features
# We will display "Income", "income_tier", "CCAvg", and "cc_spending_category"
display(loan_df.select("Income", "income_tier", "CCAvg", "cc_spending_category"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Save Transformed Data
# MAGIC After feature engineering, we save the transformed dataset in Delta format for efficient storage and retrieval in the next steps of the pipeline.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the path to save the transformed data in Delta format.
# MAGIC - Write the DataFrame to Delta format with overwrite mode to replace any existing table.

# COMMAND ----------

# Define the path to save the transformed loan data
transformed_data_path = f"{DA.catalog_name}.{DA.schema_name}.transformed_loan_data"

# Write the transformed DataFrame to Delta format
loan_df.write.format("delta").mode("overwrite").saveAsTable(transformed_data_path)

print(f"Transformed loan data with engineered features saved to: {transformed_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded the cleaned loan data saved from Task 1.
# MAGIC - Enhanced the dataset with new engineered features, such as income tiers and credit card spending categories.
# MAGIC - Validated the schema and inspected sample rows to ensure data integrity.
# MAGIC - Saved the transformed dataset in Delta format, preparing it for further processing in the machine learning pipeline.
# MAGIC
# MAGIC This transformed dataset will be used in subsequent steps of the pipeline to perform advanced feature engineering and model training.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>