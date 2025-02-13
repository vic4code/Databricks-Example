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
# MAGIC # LAB- Evaluation with Mosaic AI Agent Evaluation
# MAGIC
# MAGIC In this lab, you will have the opportunity to evaluate a RAG chain model **using Mosaic AI Agent Evaluation Framework.**
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will complete the following tasks:*
# MAGIC
# MAGIC - **Task 1**: Define a custom Gen AI evaluation metric.
# MAGIC
# MAGIC - **Task 2**: Conduct an evaluation test using the Agent Evaluation Framework.
# MAGIC
# MAGIC - **Task 3**: Analyze the evaluation results through the user interface.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -q -U  mlflow==2.15.1 databricks-agents databricks-vectorsearch langchain==0.2.11 langchain-community==0.2.10
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04

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
# MAGIC ## Lab Overview
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC In this lab, you will work with the same dataset utilized in the demos. This dataset contains sample queries along with their corresponding expected responses, which are generated using synthetic data.

# COMMAND ----------

display(DA.eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Model
# MAGIC
# MAGIC A RAG chain has been created and registered for use in this lab. The model details are provided below.
# MAGIC
# MAGIC **📌 Note:** If you are using your own workspace to run this lab, you must manually execute **`00 - Build Model / 00-Build Model`**.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

schema_name = f"ws_{spark.conf.get('spark.databricks.clusterUsageTags.clusterOwnerOrgId')}"
model_uri = f"models:/genai_shared_catalog_eval_gov.{schema_name}.rag_app/1"
model_name = f"genai_shared_catalog_eval_gov.{schema_name}.rag_app"

print(f"ws_{spark.conf.get('spark.databricks.clusterUsageTags.clusterOwnerOrgId')}")
print(model_uri)
print(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Define A Custom Metric
# MAGIC
# MAGIC For this task, define a custom metric to evaluate whether the generated **"ANSWER"** from the RAG chain is easily readable by a non-expert user.

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric_from_prompt

# Prompt for LLM as judge to determine if the generated response is easily readable by non-academic or expert readers
eval_prompt = "Your task is to determine whether the generated response easily readble by non-academic or expert readers. This was the content: '{retrieved_context}'"

# Use Llama-3 as LLM
llm="endpoints:/databricks-meta-llama-3-1-70b-instruct"

# Define the metric
is_readable = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2 - Run Evaluation Test
# MAGIC
# MAGIC Next, run an evaluation using the custom metric you defined. Ensure that you select **Mosaic AI Agent Evaluation** as the evaluation type.
# MAGIC

# COMMAND ----------

with <FILL_IN>(run_name="lab_04_agent_evaluation"):
    eval_results = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Review Evaluation Results
# MAGIC
# MAGIC Review the evaluation results in the **Experiments** section. Examine the following information regarding this evaluation:
# MAGIC
# MAGIC - Token usage
# MAGIC
# MAGIC - Model metrics
# MAGIC
# MAGIC - Results of the custom metric defined earlier ("readability")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you evaluated a RAG chain using the Mosaic AI Evaluation Framework library. You began by loading the dataset and RAG model. Then, you defined a custom metric and initiated the evaluation process. Finally, you reviewed the results through the user interface.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>