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
# MAGIC # Evaluation with Mosaic AI Agent Evaluation
# MAGIC
# MAGIC In previous demonstrations, we utilized `mlflow` for evaluation purposes. Mosaic AI Agent Evaluation builds upon MLflow, offering additional features and enhancements. It enables the definition of custom evaluation metrics, facilitates straightforward model deployment, and provides an easy-to-use **Review App**.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC - Load a model from the model registry and use it to evaluate an evaluation dataset.
# MAGIC - Define custom evaluation metrics.
# MAGIC - Deploy the model along with the Review App to gather human feedback.

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demo, we will begin by reviewing **the dataset** that will be used for evaluation. Next, we will **load a RAG chain** model from the model registry and utilize it for evaluation purposes. To illustrate custom evaluation, we will define a custom metric and incorporate it into the evaluation workflow. Upon completing the evaluation, we will **deploy the model** and demonstrate how to use the integrated "Review App" to gather **human feedback**.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Evaluation Dataset
# MAGIC
# MAGIC This dataset includes sample queries and their corresponding expected responses. The expected responses are generated using synthetic data. In a real-world project, these responses would be crafted by experts.

# COMMAND ----------

display(DA.eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Model
# MAGIC
# MAGIC A RAG chain has been created and registered for us. If you're interested in the code, you can explore the `00 - Build Model` folder. Please note that building RAG chains is beyond the scope of this course. For more information on these topics, you can refer to the related course, **"Generative AI Solution Development"** available on the Databricks Academy.

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
# MAGIC ## Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Custom Metrics
# MAGIC Although the Agents Evaluation framework automatically calculates common evaluation metrics, there are instances where we may need to assess the model using custom metrics. In this section, we will define a custom metric to evaluate whether the **retrieval model** generates responses containing personally identifiable information (PII).

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric_from_prompt

# Define a custom assessment to detect PII in the retrieved chunks. 
has_pii_prompt = "Your task is to determine whether the retrieved content has any PII information. This was the content: '{retrieved_context}'"

has_pii = make_genai_metric_from_prompt(
    name="has_pii",
    judge_prompt=has_pii_prompt,
    model="endpoints:/databricks-meta-llama-3-1-70b-instruct",
    metric_metadata={"assessment_type": "RETRIEVAL"},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Evaluation Test
# MAGIC
# MAGIC Please note that in the code below, we are logging the evaluation process using MLflow to enable viewing the results through the MLflow UI.

# COMMAND ----------

with mlflow.start_run(run_name="rag_eval_with_agent_evaluation"):
    eval_results = mlflow.evaluate(
        data = DA.eval_df,
        model = model_uri,
        model_type = "databricks-agent",
        extra_metrics=[has_pii]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Evaluation Results
# MAGIC
# MAGIC We have two options for reviewing the evaluation results. The first option is to examine the metrics and tables directly using the results object. The second option is to review the results through the user interface (UI).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Review Results Table

# COMMAND ----------

display(eval_results.metrics)

# COMMAND ----------

display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Results via the UI
# MAGIC
# MAGIC To view the results in the UI, follow these steps:
# MAGIC
# MAGIC - Click on the **"Experiment"** link displayed at the top of the previous code block's output for a simpler method.
# MAGIC
# MAGIC - Alternatively, you can navigate to "Experiments" in the left panel and locate the experiment registered with the title of this notebook.
# MAGIC
# MAGIC - View the overall metrics in the **Model Metrics** tab.
# MAGIC
# MAGIC - Examine detailed results for each assessment in the **Evaluation Results** tab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect Human Feedback via Databricks Review App
# MAGIC
# MAGIC The Databricks Review App stages the LLM in an environment where expert stakeholders can engage with it—allowing for conversations, questions, and more. This setup enables the collection of valuable feedback on your application, ensuring the quality and safety of its responses.
# MAGIC
# MAGIC **Stakeholders can interact with the application bot and provide feedback on these interactions. They can also offer feedback on historical logs, curated traces, or agent outputs.**
# MAGIC
# MAGIC **🚨 Note:** This step is intended for the course instructor only. If you are using your own environment, feel free to comment out the cells and run them to deploy the model and access the Review App.

# COMMAND ----------

# import time
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
# import mlflow
# from databricks import agents

# # Deploy the model with the agent framework
# deployment_info = agents.deploy(
#     model_name, 
#     model_version=1, 
#     scale_to_zero=True)

# # Wait for the Review App and deployed model to be ready
# w = WorkspaceClient()
# print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")

# while ((w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY) or (w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS)):
#     print(".", end="")
#     time.sleep(30)

# print("\nThe endpoint is ready!", end="")

# COMMAND ----------

# print(f"Endpoint URL    : {deployment_info.endpoint_url}")
# print(f"Review App URL  : {deployment_info.review_app_url}")

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
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we began by defining a custom metric to be used as an additional metric within the Agent Evaluation Framework. Next, we conducted an evaluation run and reviewed the results using both the API and the user interface. In the final step, we deployed the model through Model Serving and demonstrated how the Review App can be utilized to collect human feedback.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hepful Resources
# MAGIC
# MAGIC * **The Databricks Generative AI Cookbook ([https://ai-cookbook.io/](https://ai-cookbook.io/))**: Learning materials and production-ready code to take you from initial POC to high-quality production-ready application using Mosaic AI Agent Evaluation and Mosaic AI Agent Framework on the Databricks platform.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>