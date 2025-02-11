# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 1- Data Quality Assessment
# MAGIC
# MAGIC In this task, we will assess the quality of the loan dataset by checking for missing values, duplicates, and potential outliers. This step is critical to ensure that our data is clean and ready for further analysis in subsequent tasks.
# MAGIC
# MAGIC **Objectives:**
# MAGIC
# MAGIC - Identify and report missing values in each column.
# MAGIC - Detect duplicate rows.
# MAGIC - Perform basic outlier detection on numerical columns.

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
# MAGIC - Check for missing values in each column.
# MAGIC - Identify duplicate rows in the dataset.
# MAGIC - Detect potential outliers in numerical columns.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1: Load Loan Data
# MAGIC In this step, we will load the loan dataset from a Delta table. We will then inspect the data to understand its structure and ensure it has been loaded correctly.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the path to the Delta table containing loan data.
# MAGIC - Use the Spark DataFrame API to load the data and display a sample for verification.

# COMMAND ----------

# Define the dataset path
dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"

# Load the loan dataset
loan_data = spark.read.format('csv').option('header', 'true').load(dataset_path)

# Display the first few rows to inspect the data
display(loan_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Data Quality Checks
# MAGIC We will perform three main data quality checks: identifying missing values, detecting duplicates, and finding potential outliers.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####Step 2.1: Check for Missing Values
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Use Spark functions to count missing values in each column.
# MAGIC - Display columns with missing values for further analysis.

# COMMAND ----------

from pyspark.sql.functions import count, when, col

# Check for missing values in each column
missing_values = loan_data.select([count(when(col(c).isNull(), c)).alias(c) for c in loan_data.columns])

# Display columns with missing values
display(missing_values)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Step 2.2: Detect Duplicate Rows
# MAGIC **Instructions:**
# MAGIC - Use the `groupBy` and `count` functions to detect duplicate rows.
# MAGIC - Count the number of duplicate rows in the dataset.

# COMMAND ----------

# Check for duplicate rows
duplicates = loan_data.groupBy(loan_data.columns).count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicates}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Step 2.3: Outlier Detection in Numerical Columns
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Calculate the mean and standard deviation for numerical columns.
# MAGIC - Identify outliers as values that are more than three standard deviations from the mean.

# COMMAND ----------

from pyspark.sql.functions import mean, stddev

# List to store columns with detected outliers
outlier_columns = []

# Detect outliers in numerical columns
for column, dtype in loan_data.dtypes:
    if dtype in ['double', 'int']:
        mean_val = loan_data.select(mean(col(column))).collect()[0][0]
        stddev_val = loan_data.select(stddev(col(column))).collect()[0][0]
        outliers = loan_data.filter((col(column) > mean_val + 3 * stddev_val) | (col(column) < mean_val - 3 * stddev_val))
        if outliers.count() > 0:
            outlier_columns.append(column)

# Display columns with outliers
print("Columns with outliers:", outlier_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Log Data Quality Report
# MAGIC We will save a data quality report with the results from the above checks, including missing values, duplicate rows, and columns with outliers.

# COMMAND ----------

# Save data quality report
with open("./data_quality_report.txt", "w") as f:
    f.write("Data Quality Report\n")
    f.write("===================\n")
    f.write("Missing Values:\n")
    missing_values_list = missing_values.collect()
    for row in missing_values_list:
        f.write(f"{row}\n")
    f.write(f"\nDuplicate rows: {duplicates}\n")
    f.write(f"Columns with outliers: {outlier_columns}\n")

print("Data Quality Report saved to ./data_quality_report.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Set Conditional Flag for Data Quality Issues
# MAGIC Finally, we will set a flag to indicate if any data quality issues were found, which will guide the next steps in the pipeline.

# COMMAND ----------

data_quality_issues = bool(
    sum(missing_values.selectExpr(f"sum(`{col}`)").collect()[0][0] for col in missing_values.columns) > 0 or 
    duplicates > 0 or 
    outlier_columns
)
print(f"Data Quality Issues Found: {data_quality_issues}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded and inspected the loan dataset.
# MAGIC - Checked for missing values, duplicates, and potential outliers.
# MAGIC - Generated a data quality report and saved it to the Databricks file system.
# MAGIC - Set a conditional flag to indicate the presence of data quality issues.
# MAGIC
# MAGIC This data quality assessment ensures that the dataset is prepared for further analysis. If data quality issues are found, they will trigger an alternative path in the workflow pipeline for additional review or corrective actions.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>