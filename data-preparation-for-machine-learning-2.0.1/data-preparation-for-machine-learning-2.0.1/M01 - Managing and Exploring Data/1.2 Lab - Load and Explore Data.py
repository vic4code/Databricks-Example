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
# MAGIC # LAB - Load and Explore Data
# MAGIC
# MAGIC
# MAGIC Welcome to the "Load and Explore Data" lab! In this session, you will learn essential skills in data loading and exploration using PySpark in a Databricks environment. Gain hands-on experience reading data from Delta tables, managing data permissions, computing summary statistics, and using data profiling tools to unveil insights in your Telco dataset. Let's dive into the world of data exploration!
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC
# MAGIC In this Lab, you will learn how to:
# MAGIC 1. Read data from delta table
# MAGIC 1. Manage data permissions
# MAGIC 1. Show summary statistics
# MAGIC 1. Use data profiler to explore data frame
# MAGIC     - Check outliers
# MAGIC     - Check data distributions
# MAGIC 1. Read previous versions of the delta table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
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
# MAGIC ## Lab Setup
# MAGIC
# MAGIC Before starting the Lab, follow these initial steps:
# MAGIC
# MAGIC 1. Run the provided classroom setup script. This script will establish necessary configuration variables tailored to each user. Execute the following code cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll make use of the object `DA`, which provides critical variables. Execute the code block below to see various variables that will be used in this notebook:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1: Read Data from Delta Table
# MAGIC
# MAGIC
# MAGIC + Use Spark to read data from the Delta table into a DataFrame.
# MAGIC
# MAGIC

# COMMAND ----------

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

# Read dataset with spark
telco_df = <FILL_IN>

table_name = "telco_missing"
table_name_bronze = f"{table_name}_bronze"

# Write it as delta table
telco_df.write.<FILL_IN>
telco_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2: Manage Data Permissions
# MAGIC
# MAGIC Establish controlled access to the Telco Delta table by granting specific permissions for essential actions.
# MAGIC
# MAGIC + Grant permissions for specific actions (e.g., read, write) on the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Write query to Grant Permission to all the users to access Delta Table
# MAGIC <FILL_IN>;

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3: Show Summary Statistics
# MAGIC
# MAGIC
# MAGIC Compute and present key statistical metrics to gain a comprehensive understanding of the Telco dataset.
# MAGIC
# MAGIC
# MAGIC + Utilize PySpark to compute and display summary statistics for the Telco dataset.
# MAGIC
# MAGIC + Include key metrics such as mean, standard deviation, min, max, etc.

# COMMAND ----------

# Show summary of the Data
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Use Data Profiler to Explore DataFrame
# MAGIC Use the Data Profiler and Visualization Editor tools.
# MAGIC
# MAGIC + Use the Data Profiler to explore the structure, data types, and basic statistics of the DataFrame.
# MAGIC     - **Task 4.1.1:** Identify columns with missing values and analyze the percentage of missing data for each column.
# MAGIC     - **Task 4.1.2:** Review the data types of each column to ensure they match expectations. Identify any columns that might need type conversion.
# MAGIC + Use Visualization Editor to Check Outliers and Data Distributions:
# MAGIC     - **Task 4.2.1:** Create a bar chart to visualize the distribution of churned and non-churned customers.
# MAGIC     - **Task 4.2.2:** Generate a pie chart to visualize the distribution of different contract types.
# MAGIC     - **Task 4.2.3:** Create a scatter plot to explore the relationship between monthly charges and total charges.
# MAGIC     - **Task 4.2.4:** Visualize the count of customers for each payment method using a bar chart.
# MAGIC     - **Task 4.2.5:** Compare monthly charges for different contract types using a box plot.
# MAGIC

# COMMAND ----------

# Display the data and Explore the Data Profiler and Visualization Editor
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5: Drop the Column
# MAGIC Remove a specific column, enhancing data cleanliness and focus.
# MAGIC
# MAGIC
# MAGIC + Identify the column that needs to be dropped. For example, let's say we want to drop the 'SeniorCitizen' column.
# MAGIC
# MAGIC
# MAGIC + Use the appropriate command or method to drop the identified column from the Telco dataset.
# MAGIC
# MAGIC
# MAGIC + Verify that the column has been successfully dropped by displaying the updated dataset.

# COMMAND ----------

# Drop SeniorCitizen Column 
telco_dropped_df = <FILL_IN>

# Overwrite the Delta table
telco_dropped_df.write.mode("overwrite")<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Time-Travel to First 
# MAGIC
# MAGIC
# MAGIC Revert the Telco dataset back to its initial state, exploring the characteristics of the first version.
# MAGIC
# MAGIC
# MAGIC + Utilize time-travel capabilities to revert the dataset to its initial version.
# MAGIC
# MAGIC
# MAGIC + Display and analyze the first version of the Telco dataset to understand its original structure and content.
# MAGIC

# COMMAND ----------

# Extract timestamp of first version (can also be set manually)
timestamp_v0 = spark.sql(<FILL_IN>)
(spark
        .read
        .option(<FILL_IN>)
        .table(<FILL_IN>)
        .printSchema()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Task 7: Read previous versions of the delta table
# MAGIC Demonstrate the ability to read data from a specific version of the Delta table.
# MAGIC
# MAGIC + Replace the timestamp in the code with the actual version or timestamp of interest.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show table versions
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conclusion
# MAGIC In this lab, you demonstrated how to explore and manipulate the dataset using Databricks, focusing on data exploration, management, and time-travel capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>