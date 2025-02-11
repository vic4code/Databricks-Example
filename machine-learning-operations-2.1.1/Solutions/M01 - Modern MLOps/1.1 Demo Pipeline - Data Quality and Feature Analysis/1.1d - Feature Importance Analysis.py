# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Task 3 - Feature Importance Analysis
# MAGIC
# MAGIC In this task, we will perform feature importance analysis on the loan dataset. This analysis helps us understand which features have the most significant influence on predicting the target variable (e.g., Personal Loan). Feature importance analysis is useful for feature selection, model interpretability, and refining the dataset for model training.
# MAGIC
# MAGIC **Objectives:**
# MAGIC
# MAGIC - Train a Random Forest model to analyze feature importance.
# MAGIC - Identify the top features contributing to the model's predictions.
# MAGIC - Save a report with the most important features.

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
# MAGIC ## Task Outline:
# MAGIC In this task, we will:
# MAGIC
# MAGIC - Load the loan dataset.
# MAGIC - Preprocess the data by encoding categorical variables and preparing the feature set.
# MAGIC - Train a Random Forest model to assess feature importance.
# MAGIC - Identify and save the top important features.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 1: Load Loan Data
# MAGIC In this step, we load the loan dataset from a Delta table and display a sample to verify the data structure.

# COMMAND ----------

# Define the dataset path
dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"

# Load the loan dataset
data = spark.read.format('csv').option('header', 'true').load(dataset_path)

# Display the first few rows to inspect the data
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 2: Preprocess the Data
# MAGIC For this feature importance analysis, we will prepare the dataset by separating features and the target variable. We'll also perform encoding on categorical features.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Drop any unnecessary columns (e.g., ID and ZIP_Code).
# MAGIC - Define Personal Loan as the target variable and the remaining columns as features.
# MAGIC - Convert categorical variables to numeric format for compatibility with the Random Forest model.

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType
from sklearn.model_selection import train_test_split

# Define target and features
target = "Personal Loan"
X = data.drop("ID", "ZIP Code", target)  # Drop target and unnecessary columns
y = data.select(col(target).cast("int")).toPandas()[target]  # Convert target to integer and collect as pandas Series

# Encode categorical variables
categorical_cols = [field.name for field in X.schema.fields if isinstance(field.dataType, StringType)]

for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
    X = indexer.fit(X).transform(X).drop(col_name)

# Convert X to pandas DataFrame for train_test_split
X = X.toPandas()

# Split the dataset for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 3: Train a Random Forest Model for Feature Importance
# MAGIC Using the Random Forest model, we can determine the importance of each feature in predicting the target variable.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Train a Random Forest model on the training data.
# MAGIC - Extract feature importances and identify the top features.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(5)

print("Top 5 Important Features")
print("========================")
print(important_features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Step 4: Save Feature Importance Report
# MAGIC We will save the top features identified by the model to a CSV file for reference and further analysis in the MLOps pipeline.

# COMMAND ----------

import json

# Convert important features to DataFrame
important_features_df = important_features.reset_index()
important_features_df.columns = ["Feature", "Importance"]

# Convert the sample display data to JSON format for API access and inspection
output_data = important_features_df.head(5).to_dict(orient="records")

# Print the JSON-formatted output for the final task
print("Notebook_Output:", json.dumps(output_data, indent=4))

# Save the visualization data and sample output to a JSON file
output_json_path = "./feature_engineered_output_with_visualization_data.json"
with open(output_json_path, "w") as json_file:
    json.dump({"sample_data": output_data}, json_file, indent=4)

print(f"JSON output saved to: {output_json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this notebook, we:
# MAGIC
# MAGIC - Loaded and preprocessed the loan dataset.
# MAGIC - Trained a Random Forest model to analyze feature importance.
# MAGIC - Identified and saved the top features influencing the Personal Loan target variable.
# MAGIC
# MAGIC The feature importance analysis provides insights into which features most impact the model's predictions, enabling better model interpretability and guiding feature selection for future modeling steps. The saved report will be used in subsequent steps of the pipeline for further validation and monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>