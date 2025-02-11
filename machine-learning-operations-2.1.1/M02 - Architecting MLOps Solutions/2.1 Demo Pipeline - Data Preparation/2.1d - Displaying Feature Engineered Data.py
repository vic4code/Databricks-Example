# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 4 - Displaying Feature Engineered Data
# MAGIC
# MAGIC This notebook is the final task in our MLOps pipeline workflow. Here, we load the feature-engineered loan dataset, display it in a structured format, create a visualization based on aggregated data, and save both the dataset and visualization for downstream tasks.
# MAGIC
# MAGIC **Objectives:**
# MAGIC - Load the feature-engineered dataset created in Task 3.
# MAGIC - Display and inspect the dataset to verify structure and content.
# MAGIC - Create a bar chart visualization showing total assets by credit usage category.
# MAGIC - Save the visualization as a PNG file and a sample of the dataset as a JSON file for API access or inspection.

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
# MAGIC Throughout this lab, we'll refer to the object DA. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Task Outline
# MAGIC In this task, we will:
# MAGIC
# MAGIC - **Load the Feature-Engineered Data:** Load the dataset created in Task 3, which includes advanced features.
# MAGIC - **Display and Inspect Data:** Select key columns for display and basic inspection.
# MAGIC - **Create Visualization:** Generate a bar chart showing total assets by credit usage category to explore the distribution of assets.
# MAGIC - **Save Outputs:** Save the visualization as a PNG file and a sample of the dataset as a JSON file for further inspection or API access.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1: Load the Feature-Engineered Data
# MAGIC We load the feature-engineered dataset created in Task 3 to inspect its structure and content.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define the path to the feature-engineered data.
# MAGIC - Load the data into a Spark DataFrame and display the relevant columns to verify the data is as expected.

# COMMAND ----------

# Import necessary libraries
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Define the path to the feature-engineered loan data
feature_engineered_data_path = f"{DA.catalog_name}.{DA.schema_name}.feature_engineered_loan_data"

# Load the feature-engineered data
loan_df = spark.read.table(feature_engineered_data_path)

# Select the columns to display and analyze
output_df = loan_df.select("income", "debt_to_income_ratio", "total_assets", "credit_usage_category")
display(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Display and Inspect Data
# MAGIC After loading the data, convert it to a Pandas DataFrame for more flexibility in data visualization and exporting.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Convert the Spark DataFrame to a Pandas DataFrame to enable plotting with Matplotlib.
# MAGIC - Select a subset of the data for further analysis and inspection.

# COMMAND ----------

# Convert the DataFrame to Pandas for visualization and saving
output_df_pd = output_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 3: Create Visualization
# MAGIC Using the Pandas DataFrame, we will create a bar chart to show the total assets by credit usage category. This will help us understand the distribution of assets across different credit usage levels.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Group the data by credit usage category and sum the total assets.
# MAGIC - Plot the aggregated data as a bar chart and save it as a PNG file.

# COMMAND ----------

# Aggregated data for visualization: Total Assets Indicator by Credit Usage Category
total_assets_by_category = (
    output_df_pd.groupby("credit_usage_category")["total_assets"]
    .sum()
    .reset_index()
)

# Visualization: Total Assets Indicator vs. Credit Usage Category
plt.figure(figsize=(10, 6))
total_assets_by_category.set_index("credit_usage_category")["total_assets"].plot(
    kind="bar", color="salmon", edgecolor="black"
)
plt.title("Total Assets Indicator by Credit Usage Category")
plt.xlabel("Credit Usage Category")
plt.ylabel("Total Assets Indicator (Sum)")

# Save the visualization as a PNG file
image_path = "./total_assets_by_credit_usage_category.png"
plt.savefig(image_path, format="png", bbox_inches="tight")
plt.show()

print(f"Visualization saved as PNG at: {image_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Save Outputs
# MAGIC Finally, we will save a sample of the data as a JSON file for easy inspection and API access. This will include both the sample data and a link to the visualization.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Convert a sample of the display data to JSON format.
# MAGIC - Save the JSON output to a file for future API access or inspection.
# MAGIC - Save the PNG visualization image to a specified file path.

# COMMAND ----------

# Convert the sample display data to JSON format for API access and inspection
output_data = output_df_pd.head(5).to_dict(orient="records")

# Print the JSON-formatted output for the final task
print("Notebook_Output:", json.dumps(output_data, indent=4))

# Save the visualization data and sample output to a JSON file
output_json_path = "./feature_engineered_output_with_visualization_data.json"
with open(output_json_path, "w") as json_file:
    json.dump({"sample_data": output_data}, json_file, indent=4)

print(f"JSON output saved to: {output_json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded the feature-engineered dataset from Task 3.
# MAGIC - Displayed and inspected the data to confirm it is ready for model training.
# MAGIC - Created a bar chart visualization to show the distribution of total assets across different credit usage categories.
# MAGIC - Saved the visualization as a PNG and a sample of the data as a JSON file, preparing the outputs for further use or inspection.
# MAGIC
# MAGIC This completes the data preparation and visualization steps in our MLOps pipeline workflow. The prepared dataset is now ready for downstream machine learning tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>