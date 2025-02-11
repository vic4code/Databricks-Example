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
# MAGIC # LAB - Hyperparameter Tuning with Hyperopt
# MAGIC
# MAGIC Welcome to the Hyperparameter Tuning with Hyperopt lab! In this hands-on session, you'll gain practical insights into **optimizing machine learning models using Hyperopt**. Throughout the lab, we'll cover key steps, from loading the dataset and creating training/test sets to **defining a hyperparameter search space and running optimization trials with Spark**. The primary objective is to equip you with the skills to fine-tune models effectively using Spark, hyperopt and MLflow.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC 1. Load the dataset and create training/test sets.
# MAGIC
# MAGIC 1. Define the hyperparameter search space for optimization.
# MAGIC
# MAGIC 1. Define the optimization function to fine-tune the model.
# MAGIC
# MAGIC 1. Run hyperparameter tuning trials using Spark.
# MAGIC
# MAGIC 1. Show the best run's info.
# MAGIC
# MAGIC 1. Search for runs using the MLflow API.
# MAGIC
# MAGIC 1. Search for runs using the MLflow UI.

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

# MAGIC %run ../Includes/Classroom-Setup-2.2

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
# MAGIC ## Prepare Dataset
# MAGIC
# MAGIC In this lab you will be using a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC In this lab will create and tune a model that will predict customer churn based on **`Churn`** field. 
# MAGIC
# MAGIC A table with all features is already created for you.
# MAGIC
# MAGIC **Table name: `customer_churn`**

# COMMAND ----------

import mlflow.data
from sklearn.model_selection import train_test_split

# load data from the feature table
table_name = f"{DA.catalog_name}.{DA.schema_name}.customer_churn"
dataset = mlflow.data.load_delta(table_name=table_name)
pd = dataset.df.drop("CustomerID").toPandas()

# split dataset to train/test 
target_col = "Churn"
X_all = pd.drop(labels=target_col, axis=1)
y_all = pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1:  Define Hyperparameter Search Space
# MAGIC
# MAGIC Define the parameter search space for hyperopt. Define these hyperparameters and search space;
# MAGIC * **`max_depth`:** 2 to 30
# MAGIC * **`max_features`**: 5 to 10
# MAGIC
# MAGIC Note that both parameters are discrete values.

# COMMAND ----------

from hyperopt import hp

# define param search space
dtc_param_space = {
  'max_depth': hp.choice('dtree_max_depth',
                          [None, hp.uniformint('dtree_max_depth_int', 2, 30)]),
  'max_features': hp.choice('dtree_max_features',
                          [None, hp.uniformint('dtree_max_features_int', 5, 10)]),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Define Optimization Function
# MAGIC
# MAGIC Next, define an optimization function that will be used by hyperopt for minimizing the loss. 
# MAGIC
# MAGIC Make sure to follow instructions;
# MAGIC
# MAGIC * Make sure to enable MLflow run as **`nested`** experiment. 
# MAGIC
# MAGIC * For each run log the cross-validation results for `accuracy`, `precision`, `recall` and `f1`
# MAGIC
# MAGIC * Use **3-fold** cross validation
# MAGIC
# MAGIC * Minimize loss based on the **`precision`** score

# COMMAND ----------

from math import sqrt

import mlflow
import mlflow.data
import mlflow.sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

from hyperopt import STATUS_OK

def tuning_objective(params):
  # start an MLFlow run
  with mlflow.start_run(nested=True) as mlflow_run:
    
    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        disable=False,
        log_input_examples=True,
        silent=True,
        exclusive=False)

    # set up our model estimator
    dtc = DecisionTreeClassifier(**params)
    
    # cross-validated on the training set
    validation_scores = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(dtc, 
                                X_train, 
                                y_train, 
                                cv=3,
                                scoring=validation_scores)
    # log the average cross-validated results
    cv_score_results = {}
    for val_score in validation_scores:
      cv_score_results[val_score] = cv_results[f'test_{val_score}'].mean()
      mlflow.log_metric(f"cv_{val_score}", cv_score_results[val_score])

    # return the negative of our cross-validated precision score as the loss
    return {
      "loss": -cv_score_results['precision'],
      "status": STATUS_OK,
      "run": mlflow_run
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Run Trials in Hyperopt
# MAGIC
# MAGIC After defining the *objective function*, we are ready to run this function with hyperopt. 
# MAGIC
# MAGIC * Use `SparkTrails` and run *3 trails* in parallel.
# MAGIC
# MAGIC * Use **TPE** algorithm for optimization.
# MAGIC
# MAGIC * Use maximum 3 evaluations.

# COMMAND ----------

from hyperopt import SparkTrials, fmin, tpe

# set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/LAB-2-Hyperparameter-Tuning-with-Hyperopt")

trials = SparkTrials(parallelism=3)
with mlflow.start_run(run_name="Model Tuning with Hyperopt Demo") as parent_run:
  # caal objective function using tpe and maximum eval to 3
  fmin(tuning_objective,
      space=dtc_param_space,
      algo=tpe.suggest,
      max_evals=3,
      trials=trials)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Show the Best Run Info

# COMMAND ----------

# get best trail and show the info
best_run = trials.best_trial["result"]['run']
best_run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Search for the Best Run with MLflow API
# MAGIC
# MAGIC We just got the best run based on the loss metric in the previous step. Sometimes we might need to search for runs using custom filters such as by parent run or by another metric. 
# MAGIC
# MAGIC In this step, search for runs of `parent_run` experiment and use following filters;
# MAGIC
# MAGIC * Filter by runs which has `FINISHED`
# MAGIC
# MAGIC * Order by **cross validation precision** score from **high to low**.

# COMMAND ----------

from mlflow.entities import ViewType

# search over all runs
hpo_runs_pd = mlflow.search_runs(
  experiment_ids=[parent_run.info.experiment_id],
  filter_string=f"attributes.status = 'FINISHED'",
  order_by=["metrics.cv_precision DESC"],
)

display(hpo_runs_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Search for the Best Run with MLflow UI
# MAGIC
# MAGIC Another way of searching for runs is to simply use the MLflow UI. In this section, we will need to review the experiment and runs and filter runs based on the same filters that are defined in the previous step but this time using the UI.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In conclusion, you have successfully completed the Hyperparameter Tuning with Hyperopt lab, gaining practical insights into optimizing machine learning models. Throughout this hands-on session, you've mastered key steps, from defining a hyperparameter search space to executing optimization trials with Spark. Additionally, you searched for and analyzed the best model runs through both the MLflow API and the user-friendly MLflow UI. The primary objective was to empower you with the skills to fine-tune models effectively using Spark, Hyperopt, and MLflow. As you conclude this lab, you are now adept at these techniques. Congratulations on your achievement!

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>