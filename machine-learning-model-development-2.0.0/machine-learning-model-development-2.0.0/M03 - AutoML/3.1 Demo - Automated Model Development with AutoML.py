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
# MAGIC # Automated Model Development with AutoML
# MAGIC
# MAGIC In this demo, we will demonstrate how to initiate AutoML experiments both through the user-friendly AutoML UI and programmatically using the AutoML API. 
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Start an AutoML experiment via the AutoML UI.
# MAGIC
# MAGIC * Start an AutoML experiment via the AutoML API.
# MAGIC
# MAGIC * Open and edit a notebook generated by AutoML.
# MAGIC
# MAGIC * Identify the best model generated by AutoML based on a given metric.
# MAGIC
# MAGIC * Modify the best model generated by AutoML.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
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
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.1

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
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prepare Data
# MAGIC
# MAGIC For this demonstration, we will utilize a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC A table with all features is already created for you.
# MAGIC
# MAGIC **Table name: `customer_churn`**
# MAGIC
# MAGIC To get started, execute the code block below and review the dataset schema.

# COMMAND ----------

churn_data = spark.sql("SELECT * FROM customer_churn")
display(churn_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Experiment with UI
# MAGIC
# MAGIC Databricks AutoML supports experimentation via the UI and the API. Thus, **in the first section of this demos we will demonstrate how to create an experiment using the UI**. Then, show how to create the same experiment via the API.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create AutoML Experiment
# MAGIC
# MAGIC Let's initiate an AutoML experiment to construct a baseline model for predicting customer churn. The target field for this prediction will be the `Churn` field.
# MAGIC
# MAGIC Follow these step-by-step instructions to create an AutoML experiment:
# MAGIC
# MAGIC 1. Navigate to the **Experiments** section in Databricks.
# MAGIC
# MAGIC
# MAGIC 2. Click on **Start training** Under **Classification**.
# MAGIC
# MAGIC   ![automl-create-experiment-v1](files/images/machine-learning-model-development-2.0.0/automl-create-experiment-v1.png)
# MAGIC
# MAGIC 3. Choose a cluster to execute the experiment.
# MAGIC
# MAGIC 4. Select the **catalog > schema > `customers_churn` table**, which was created in the previous step, as the input training dataset.
# MAGIC
# MAGIC 5. Specify **`Churn`** as the prediction target.
# MAGIC
# MAGIC 6. Deselect the **CustomerID** field as it's not needed as a feature.
# MAGIC
# MAGIC 7. In the **Advanced Configuration** section, set the **Timeout** to **5 minutes**.
# MAGIC
# MAGIC 8. Enter a name for your experiment. Let's enter `Churn_Prediction_AutoML_Experiment` as experiment name.
# MAGIC
# MAGIC ![automl-input-fields-v1](files/images/machine-learning-model-development-2.0.0/automl-input-fields-v1.png)
# MAGIC
# MAGIC **Optional Advanced Configuration:**
# MAGIC <img src ="https://files.training.databricks.com/images/automl-advanced-configuration-optional-v1.png"> 
# MAGIC - You have the flexibility to choose the **evaluation metric** and your preferred **training framework**.
# MAGIC
# MAGIC - If your dataset includes a timeseries field, you can define it when splitting the dataset.
# MAGIC
# MAGIC 9. Click on **Start AutoML**.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### View the Best Run
# MAGIC
# MAGIC Once the experiment is finished, it's time to examine the best run:
# MAGIC
# MAGIC 1. Access the completed experiment in the **Experiments** section.
# MAGIC <img src = "https://files.training.databricks.com/images/automl-completed-experiment-v1.png" width = 1000>
# MAGIC
# MAGIC 2. Identify the best model run by evaluating the displayed **metrics**. Alternatively, you can click on **View notebook for the best model** to access the automatically generated notebook for the top-performing model.
# MAGIC <img src ="https://files.training.databricks.com/images/automl-metrics-v1.png" width = 1000>
# MAGIC
# MAGIC 3. Utilize the **Chart** tab to compare and contrast the various models generated during the experiment.
# MAGIC
# MAGIC You can find all details for the run  on the experiment page. There are different columns such as the framework used (e.g., `Scikit-Learn`, `XGBoost`), evaluation metrics (e.g., `Accuracy`, `F1 Score`), and links to the corresponding notebooks for each model. This allows you to make informed decisions about selecting the best model for your specific use case.

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Notebook
# MAGIC
# MAGIC ####**Instruction for viewing the notebook of the best run:**
# MAGIC
# MAGIC
# MAGIC
# MAGIC + **Click on the `"View notebook for best model"` link.**
# MAGIC
# MAGIC + **Review the notebook that created the best model.**
# MAGIC
# MAGIC <img src ="https://files.training.databricks.com/images/automl-best-model-notebook-v1.png" width= 1000>
# MAGIC
# MAGIC
# MAGIC + **Edit the notebook as required.**
# MAGIC     + Identify the best model generated by AutoML based on a given metric and modify it as needed. The best model details, including the associated run ID, can be found in the MLflow experiment logs. Use the run ID to load the best model, make modifications, and save the modified model for deployment or further use.

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Experiment with API
# MAGIC
# MAGIC In this section we will use AutoML API to start and AutoML job and retrieve the experiment results.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start an Experiment

# COMMAND ----------

from databricks import automl
from datetime import datetime
automl_run = automl.classify(
    dataset = spark.table("customer_churn"),
    target_col = "Churn",
    exclude_cols=["CustomerID"], # Exclude columns as needed
    timeout_minutes = 5
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search for the Best Run
# MAGIC
# MAGIC The search for the best run in this experiment, we need to first **get the experiment ID** and then **search for the runs** by experiment.

# COMMAND ----------

import mlflow
# Get the experiment path by experiment ID
exp_path = mlflow.get_experiment(automl_run.experiment.experiment_id).name
# Find the most recent experiment in the AutoML folder
filter_string=f'name LIKE "{exp_path}"'
automl_experiment_id = mlflow.search_experiments(
  filter_string=filter_string,
  max_results=1,
  order_by=["last_update_time DESC"])[0].experiment_id

# COMMAND ----------

from mlflow.entities import ViewType

# Find the best run ...
automl_runs_pd = mlflow.search_runs(
  experiment_ids=[automl_experiment_id],
  filter_string=f"attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.val_f1_score DESC"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Print information about the best trial from the AutoML experiment.**
# MAGIC

# COMMAND ----------

print(automl_run.best_trial)

# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation**
# MAGIC
# MAGIC
# MAGIC - **`print(automl_run.best_trial)`**: This prints information about the best trial or run from the AutoML experiment.
# MAGIC
# MAGIC     - **Model:** Specifies the machine learning model that performed the best. 
# MAGIC
# MAGIC     - **Model path:** The MLflow artifact URL of the model trained in this trial.
# MAGIC
# MAGIC     - **Preprocessors:** Description of the preprocessors run before training the model.
# MAGIC
# MAGIC     - **Training duration:** Displays the duration it took to train the best model.
# MAGIC
# MAGIC     - **Evaluation metric score:** Shows the value of the evaluation metric used to determine the best model. 
# MAGIC
# MAGIC     - **Evaluation metric:** Score of primary metric, evaluated for the validation dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC **Import notebooks for other runs in AutoML.**
# MAGIC
# MAGIC For classification and regression experiments, AutoML generated notebooks for data exploration and the best trial in your experiment are automatically imported to your workspace. Generated notebooks for other experiment trials are saved as MLflow artifacts on DBFS instead of auto-imported into your workspace. 
# MAGIC
# MAGIC For all trials besides the best trial, the **`notebook_path`** and **`notebook_url`** in the TrialInfo Python API are not set. If you need to use these notebooks, you can manually import them into your workspace with the AutoML experiment UI or the **`automl.import_notebook`** Python API.
# MAGIC
# MAGIC **🚨 Notice:** `destination_path` takes Workspace as root.

# COMMAND ----------

# Create the Destination path for storing the best run notebook
destination_path = f"/Users/{DA.username}/imported_notebooks/demo-3.1-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Get the path and url for the generated notebook
result = automl.import_notebook(automl_run.trials[1].artifact_uri, destination_path)
print(f"The notebook is imported to: {result.path}")
print(f"The notebook URL           : {result.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we show how to use AutoML UI and AutoML API for creating classification model and how we can retrieve the best run and access the generated notebook, and how we can modify the parameters of the best model. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>