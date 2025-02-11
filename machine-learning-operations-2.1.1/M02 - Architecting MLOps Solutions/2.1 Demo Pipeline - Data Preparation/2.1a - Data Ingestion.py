# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 1- Data Ingestion
# MAGIC
# MAGIC This notebook is the first task in our MLOps pipeline workflow. Here, we will ingest raw loan data, perform initial data cleaning, validate the schema to ensure consistency, and save the cleaned data as a Delta table for efficient retrieval in downstream tasks.
# MAGIC
# MAGIC **Objectives:**
# MAGIC - Ingest loan data from a CSV file.
# MAGIC - Perform initial exploration to understand data structure.
# MAGIC - Conduct data cleaning to handle missing values and standardize data types.
# MAGIC - Validate the data schema to ensure compatibility with future processing steps.
# MAGIC - Save the cleaned data in Delta format for optimized storage and retrieval.

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
# MAGIC ## Task Outline
# MAGIC
# MAGIC In this task, we will follow a structured approach to prepare the loan data for further analysis in the MLOps pipeline.
# MAGIC
# MAGIC 1. **Load Loan Data**: Ingest the raw loan data from a CSV file.
# MAGIC 2. **Initial Exploration**: Examine the dataset to understand its structure, data types, and any potential issues.
# MAGIC 3. **Data Cleaning**: Address missing values and ensure data types are consistent, creating a clean and standardized dataset.
# MAGIC 4. **Schema Validation**: Validate that the data schema matches expected formats for seamless downstream processing.
# MAGIC 5. **Save Cleaned Data**: Store the cleaned data in Delta format, ensuring efficient access for the next steps in the pipeline.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1: Load Loan Data
# MAGIC In this step, we will load the raw loan dataset from a CSV file. We will then inspect the data to understand its structure and ensure that it has been loaded correctly.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - Define the path to the CSV file containing loan data.
# MAGIC - Use the Spark DataFrame API to load the data and display it for verification.

# COMMAND ----------

# Define the path to the dataset
dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"

# Load the dataset as a Spark DataFrame
loan_df = spark.read.format("csv").option("header", "true").load(dataset_path)

# Display the DataFrame to inspect the data
display(loan_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Initial Exploration
# MAGIC Next, we’ll explore the dataset to get an overview of the data structure, column types, and any missing values.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Print the schema of the loaded DataFrame to understand column types.
# MAGIC - Generate basic descriptive statistics to summarize the numerical columns.
# MAGIC - Display the count of missing values for each column to identify gaps that may need handling in the cleaning process.

# COMMAND ----------

# Print the schema of the DataFrame
loan_df.printSchema()

# Display basic statistics for numerical columns
loan_df.describe().show()

# Show the count of missing values for each column
from pyspark.sql.functions import col, sum as spark_sum

missing_counts = loan_df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in loan_df.columns])
display(missing_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Data Cleaning
# MAGIC In this step, we’ll perform initial data cleaning by addressing missing values and standardizing column data types for consistency.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - Use default values to fill missing entries in key columns.
# MAGIC - Cast columns to appropriate data types (e.g., convert numerical columns to DoubleType).
# MAGIC - Rename columns to remove spaces, ensuring compatibility with downstream processing.

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql.functions import col

# Fill missing values with default values for identified columns
loan_df = loan_df.fillna({
    "Income": 0.0, 
    "Mortgage": 0.0, 
    "Education": "Unknown"
})

# Cast numerical columns to appropriate types and categorical columns as strings
loan_df = loan_df.withColumn("Income", col("Income").cast(DoubleType())) \
                 .withColumn("Mortgage", col("Mortgage").cast(DoubleType())) \
                 .withColumn("Education", col("Education").cast(StringType())) \
                 .withColumn("Age", col("Age").cast(IntegerType())) \
                 .withColumn("Experience", col("Experience").cast(IntegerType())) \
                 .withColumn("Family", col("Family").cast(IntegerType())) \
                 .withColumn("ZIP Code", col("ZIP Code").cast(IntegerType()))

# Rename columns to remove spaces for compatibility
for col_name in loan_df.columns:
    new_col_name = col_name.replace(" ", "_")
    loan_df = loan_df.withColumnRenamed(col_name, new_col_name)

# Verify the new column names
loan_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Schema Validation
# MAGIC Schema validation ensures that the data structure is consistent and compatible with downstream tasks. This step helps prevent errors in later stages by verifying data types.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the expected schema for the dataset.
# MAGIC - Iterate through each column to confirm that the actual data types match the expected types.
# MAGIC - Print any discrepancies for reference.

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StringType

# Define the expected schema for the dataset columns
expected_schema = {
    "Income": DoubleType(),
    "Age": IntegerType(),
    "Experience": IntegerType(),
    "Family": IntegerType(),
    "CCAvg": DoubleType(),
    "Education": StringType(),  # Assuming Education is categorical
    "Mortgage": DoubleType(),
    "Personal Loan": IntegerType(),
    "Securities Account": IntegerType(),
    "CD Account": IntegerType(),
    "Online": IntegerType(),
    "CreditCard": IntegerType()
}

# Validate each column's data type
for col_name, col_type in expected_schema.items():
    if col_name in loan_df.columns:
        actual_type = loan_df.schema[col_name].dataType
        if actual_type != col_type:
            print(f"Schema mismatch for column '{col_name}': Expected {col_type}, found {actual_type}")
    else:
        print(f"Column '{col_name}' not found in DataFrame")

# Rename the column to remove the space
loan_df = loan_df.withColumnRenamed("ZIP Code", "ZIP_Code")

print("Schema validation completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 5: Save Cleaned Data
# MAGIC Finally, save the cleaned and validated loan data as a Delta table. This format ensures efficient storage and retrieval for future tasks in the pipeline.
# MAGIC
# MAGIC **Instructions**
# MAGIC
# MAGIC - Define the path to save the cleaned data as a Delta table.
# MAGIC - Write the DataFrame to Delta format with overwrite mode to replace any existing table.

# COMMAND ----------

# Define the path to save the cleaned data
cleaned_data_path = f"{DA.catalog_name}.{DA.schema_name}.cleaned_loan_data"

# Write the cleaned DataFrame to Delta format
loan_df.write.format("delta").mode("overwrite").saveAsTable(cleaned_data_path)

print(f"Cleaned data saved to: {cleaned_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded raw loan data from a CSV file.
# MAGIC - Explored the dataset to understand its structure and identify missing values.
# MAGIC - Cleaned the data by filling missing values and standardizing column data types.
# MAGIC - Validated the schema to ensure data consistency for downstream tasks.
# MAGIC - Saved the cleaned data as a Delta table, optimizing it for efficient storage and retrieval.
# MAGIC
# MAGIC This dataset will be used in subsequent steps of the machine learning pipeline, enabling seamless data transformation and feature engineering in the next stages of the workflow.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>