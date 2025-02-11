# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Machine Learning Model Development
# MAGIC
# MAGIC This comprehensive course provides a practical guide to developing traditional machine learning models on Databricks, emphasizing hands-on demonstrations and workflows using popular ML libraries. This course focuses on executing common tasks efficiently with AutoML and MLflow. Participants will delve into key topics, including regression and classification models, harnessing Databricks' capabilities to track model training, leveraging feature stores for model development, and implementing hyperparameter tuning. Additionally, the course covers AutoML for rapid and low-code model training, ensuring that participants gain practical, real-world skills for streamlined and effective machine learning model development in the Databricks environment.
# MAGIC
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Time | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 120m    | **[Model Development Workflow]($./M01 - Model Development Workflow)**    | Model Development Workflow <br>[Training Regression Models]($./M01 - Model Development Workflow/1.1a Demo - Training Regression Models)<br>[Training Classification Models]($./M01 - Model Development Workflow/1.1b Demo - Training Classification Models)<br>[Model Tracking with MLflow]($./M01 - Model Development Workflow/1.2 Demo - Model Tracking with MLflow)<br>[Lab: Model Development Tracking with MLflow]($./M01 - Model Development Workflow/1.3 Lab - Model Development Tracking with MLflow)|
# MAGIC | 60m    | **[Hyperparameter Tuning with Hyperopt]($./M02 - Hyperparameter Tuning)** | Hyperparameter Tuning </br> [Hyperparameter Tuning with Hyperopt]($./M02 - Hyperparameter Tuning/2.1 Demo - Hyperparameter Tuning with Hyperopt) </br> [Lab: Hyperparameter Tuning with Hyperopt]($./M02 - Hyperparameter Tuning/2.2 Lab - Hyperparameter Tuning with Hyperopt) | 
# MAGIC | 60m    | **[AutoML]($./M03 - AutoML)** |AutoML </br> [Automated Model Development with AutoML]($./M03 - AutoML/3.1 Demo - Automated Model Development with AutoML) </br> [Lab: AutoML]($./M03 - AutoML/3.2 Lab - AutoML)|

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s):  **16.0.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>