# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo: Setting Up and Managing Workflow Jobs using UI
# MAGIC
# MAGIC In this demo, we'll set up a Databricks Workflow that automates a series of MLOps tasks such as data quality assessment, feature importance analysis, and alerting on unusual patterns. These tasks help ensure data readiness and provide insights before moving to model training. We’ll also introduce a conditional path based on the detection of unusual patterns.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC In this demo, we will:
# MAGIC
# MAGIC - Create and configure a Databricks Workflow job with multiple Python script tasks.
# MAGIC - Set dependencies and conditional paths between tasks.
# MAGIC - Enable email notifications for successful job runs.
# MAGIC - Manually trigger the workflow.
# MAGIC - Monitor the job's execution and completion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `15.4.x-cpu-ml-scala2.12`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Using a Serverless Cluster for This Notebook
# MAGIC
# MAGIC Instructors have the flexibility to use a **Serverless cluster** for this notebook if preferred. Serverless clusters can provide faster startup times and simplified resource management, making them a convenient option for running the notebook efficiently.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Create a Databricks Workflow Job in the UI
# MAGIC
# MAGIC 1. **Navigate to Workflows**:
# MAGIC    - In your Databricks workspace, click on the **Workflows** icon in the left sidebar.
# MAGIC    
# MAGIC 2. **Create a New Job**:
# MAGIC    - Click on **Create Job** in the upper-right corner of the Workflows page.
# MAGIC    - Name the job "MLOps Workflow: Data Quality and Feature Analysis" or something similar for easy identification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Add Tasks to the Job:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1: Data Quality Assessment
# MAGIC
# MAGIC 1. **Create First Task**:
# MAGIC    - Name the task `Data_Quality_Assessment`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for data quality assessment (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1a - Data Quality Assessment`).
# MAGIC    - Select an `Serverless` cluster for this task.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will check for missing values, duplicates, and outliers in the dataset and generate a data quality report.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.2: Alert on Unusual Patterns
# MAGIC
# MAGIC 1. **Create Second Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Alert_Unusual_Patterns`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for alerting on unusual patterns (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1b - alert_unusual_patterns`).
# MAGIC    - Use the same `Serverless` cluster as the previous task.
# MAGIC    - Set **Depends on** to `Data_Quality_Assessment` to ensure this task runs after data quality checks.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will check for unusual patterns, such as high cardinality and skewed distributions, setting a flag if any unusual patterns are detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3: Conditional Path Setup
# MAGIC
# MAGIC 1. **Create Condition for Unusual Patterns**:
# MAGIC    - Click on **Add Task --> If/else condition**.
# MAGIC    - Name the condition task `Alert_Unusual_Patterns_True`.
# MAGIC    - Set **Type** to `If/else condition`.
# MAGIC    - **Condition**: Set the expression to `{{tasks.Alert_Unusual_Patterns.values.unusual_pattern_status}} == unusual_pattern_detected`.
# MAGIC    - Set **Depends on** to `Alert_Unusual_Patterns` to ensure this condition is evaluated after the unusual patterns check.
# MAGIC    - This condition will branch based on whether unusual patterns are detected (`True`) or not (`False`).
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This conditional setup directs the workflow to either investigate unusual patterns (if detected) or proceed with feature importance analysis if no patterns are detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.4: Investigate and Resolve Unusual Patterns and Analyse Feature Importance
# MAGIC
# MAGIC 1. **Create Third Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Investigate_Unusual_Patterns`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for investigating unusual patterns (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1c -Investigate and Resolve Unusual Patterns`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to `Alert_Unusual_Patterns_True == True`, ensuring it only runs if unusual patterns are detected.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will execute only if unusual patterns are detected (when the condition is `True`).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.5: Feature Importance Analysis (False Path or After Investigation)
# MAGIC
# MAGIC 1. **Create Fourth Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Feature_Importance`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for feature importance analysis (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1d - Feature Importance Analysis`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to:
# MAGIC      - `Alert_Unusual_Patterns_True == False` (to run if no unusual patterns are detected).
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will run only if there are no unusual patterns or after the unusual patterns investigation is completed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.6: Save Report (Final Task)
# MAGIC
# MAGIC 1. **Create Fifth Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Save_Report`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for saving the final report (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1e - Save Report Notebook (Success Path)`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to both:
# MAGIC      - `Feature_Importance` and `Investigate_Unusual_Patterns`.
# MAGIC    - Set **Run if dependencies** to "At least one succeeded" to ensure it saves the report regardless of the path taken.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will save the final report once all prior steps are successfully completed, regardless of whether unusual patterns were detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.7: Enable Email Notifications
# MAGIC
# MAGIC 1. **Set up Notifications**:
# MAGIC    - In the job's configuration, navigate to the **Notifications** section.
# MAGIC    - Enable email notifications by adding your email to receive updates on job completion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Trigger the Workflow Manually
# MAGIC
# MAGIC 1. **Run the Job**:
# MAGIC    - Go to the job in the Databricks UI and click on **Run Now** in the top-right corner to manually trigger the job. This will execute all tasks in the workflow according to their dependencies and conditions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4: Monitor the Workflow Execution
# MAGIC
# MAGIC 1. **Navigate to the Runs Tab**:
# MAGIC    - In the job interface, go to the **Runs** tab to view active and completed executions of the job.
# MAGIC
# MAGIC 2. **Observe Task Execution**:
# MAGIC    - Each task’s status is displayed in the **Runs** tab, where you can see which tasks are currently executing or have completed.
# MAGIC    - Click on each task to view its execution details and outputs, allowing you to troubleshoot and verify each stage.
# MAGIC    - Check the logs to see if the workflow followed the correct path based on the unusual pattern detection condition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, you learned how to:
# MAGIC - Configure and execute a Databricks Workflow job with multiple tasks for data quality, feature importance, and unusual pattern checks.
# MAGIC - Use dependencies and conditional paths to control the flow of tasks based on the data conditions.
# MAGIC - Set up email notifications to stay updated on job execution.
# MAGIC - Trigger the workflow manually and monitor its execution.
# MAGIC
# MAGIC This workflow serves as a preliminary step to ensure data quality and feature insights before moving on to model training. By automating these MLOps setup tasks and handling conditional paths, you can ensure a robust pipeline that adapts based on data characteristics, providing insights and addressing issues early in the MLOps process.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>