# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 2 - Alert on Unusual Patterns
# MAGIC In this task, we will analyze the loan dataset for unusual patterns that may affect the performance of downstream machine learning tasks. Unusual patterns can include high cardinality in categorical features and skewed distributions in numerical features. Detecting these patterns early can help in adjusting feature engineering or model preparation steps.
# MAGIC
# MAGIC **Objectives:**
# MAGIC
# MAGIC - Identify columns with high cardinality, which may require transformation or encoding adjustments.
# MAGIC - Check for skewed distributions in numerical features, which may benefit from normalization or transformations.
# MAGIC - Set a flag to indicate whether unusual patterns were detected, enabling conditional paths in the MLOps pipeline.

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

# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.1demo

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
# MAGIC In this task, we will:
# MAGIC
# MAGIC - Load the loan dataset.
# MAGIC - Identify high cardinality columns.
# MAGIC - Detect skewed distributions in numerical columns.
# MAGIC - Set a flag for conditional task execution based on unusual patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1: Load Loan Data
# MAGIC In this step, we load the loan dataset from a CSV file and inspect a sample of the data to confirm the structure.

# COMMAND ----------

# Define the dataset path
dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"

# Load the loan dataset
data = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(dataset_path)

# Display the first few rows to inspect the data
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Check for High Cardinality in Categorical Features
# MAGIC High cardinality in categorical features (e.g., features with many unique values) can complicate encoding and increase model complexity. Here, we will flag columns with high cardinality.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define a threshold for high cardinality (e.g., > 100 unique values).
# MAGIC - Identify columns that exceed this threshold.
# MAGIC
# MAGIC

# COMMAND ----------

# High cardinality check (threshold set to > 100 unique values)
high_cardinality_columns = [col for col in data.columns if data.select(col).distinct().count() > 100]

print("High Cardinality Columns:")
print("========================")
print(high_cardinality_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Check for Skewed Distributions in Numerical Features
# MAGIC Skewed distributions in numerical features may lead to biased models and may require normalization or transformations. We will flag columns with high skewness.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Calculate the skewness of numerical columns.
# MAGIC - Identify columns with an absolute skewness > 1.5 as skewed.

# COMMAND ----------

from pyspark.sql.functions import skewness, abs as abs_

skewed_columns = []
numeric_columns = [col for col, dtype in data.dtypes if dtype in ['double', 'int']]

for col in numeric_columns:
    skewness_value = data.select(skewness(col)).collect()[0][0]
    if skewness_value is not None and abs(skewness_value) > 1.5:
        skewed_columns.append(col)

print("Skewed Columns:")
print("===============")
print(skewed_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Save Unusual Patterns Report
# MAGIC If unusual patterns are found, save a report that summarizes the findings. This report will help inform further data preprocessing steps.

# COMMAND ----------

# Save unusual patterns report
with open("./unusual_patterns_report.txt", "w") as f:
    f.write("Unusual Patterns Report\n")
    f.write("=======================\n")
    f.write(f"High Cardinality Columns: {high_cardinality_columns}\n")
    f.write(f"Skewed Columns: {skewed_columns}\n")

print("Unusual Patterns Report saved to ./unusual_patterns_report.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 5: Set Flag for Conditional Execution
# MAGIC If any unusual patterns are detected (high cardinality or skewed columns), we set a flag unusual_patterns_found to True. This flag will be used in the pipeline to decide whether to proceed with the regular workflow or an alternative investigation path.

# COMMAND ----------

# Import MLflow
import mlflow

# Set MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Define a function to check for unusual patterns
def check_unusual_patterns(report_path):
    # Initialize flags
    high_cardinality_found = False
    skewed_distribution_found = False
    
    try:
        # Load the unusual patterns report
        with open(report_path, "r") as f:
            unusual_patterns_report = f.read()
        
        # Check for high cardinality columns
        if "High Cardinality Columns:" in unusual_patterns_report:
            # Check if any columns are listed under high cardinality
            high_cardinality_columns = unusual_patterns_report.split("High Cardinality Columns:")[1].split("Skewed Columns:")[0].strip()
            high_cardinality_found = bool(high_cardinality_columns and high_cardinality_columns != "None")
        
        # Check for skewed distribution columns
        if "Skewed Columns:" in unusual_patterns_report:
            # Check if any columns are listed under skewed columns
            skewed_columns = unusual_patterns_report.split("Skewed Columns:")[1].strip()
            skewed_distribution_found = bool(skewed_columns and skewed_columns != "None")
        
        # Determine unusual pattern status
        if high_cardinality_found or skewed_distribution_found:
            dbutils.jobs.taskValues.set(key="unusual_pattern_status", value="unusual_pattern_detected")
            return "unusual_pattern_detected"
        else:
            dbutils.jobs.taskValues.set(key="unusual_pattern_status", value="no_unusual_pattern_detected")
            return "no_unusual_pattern_detected"
    
    except Exception as e:
        # Handle exceptions and return a default status
        print(f"Error occurred: {e}")
        dbutils.jobs.taskValues.set(key="unusual_pattern_status", value="error_in_checking")
        return "error_in_checking"

# Example usage
dbutils.jobs.taskValues.set(key="unusual_pattern_status", value="no_unusual_pattern_detected") # Default
report_path = "./unusual_patterns_report.txt" # Path to report
pattern_status = check_unusual_patterns(report_path)
print(f"Unusual pattern status: {pattern_status}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Analyzed the loan dataset for unusual patterns, including high cardinality and skewed distributions.
# MAGIC - Saved a report with the identified patterns for reference.
# MAGIC - Set a flag to enable conditional execution in the pipeline based on the detection of unusual patterns.
# MAGIC
# MAGIC This task ensures that any detected unusual patterns are accounted for in the MLOps workflow, allowing for tailored data processing and troubleshooting steps. This flag will guide the workflow to either save the final report or proceed with an investigation of the unusual patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>