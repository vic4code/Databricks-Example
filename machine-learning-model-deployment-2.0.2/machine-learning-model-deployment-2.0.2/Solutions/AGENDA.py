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
# MAGIC ## Machine Learning Model Deployment
# MAGIC
# MAGIC This course is designed to introduce three primary machine learning deployment strategies and illustrate the implementation of each strategy on Databricks. Following an exploration of the fundamentals of model deployment, the course delves into batch inference, offering hands-on demonstrations and labs for utilizing a model in batch inference scenarios, along with considerations for performance optimization. The second part of the course comprehensively covers pipeline deployment, while the final segment focuses on real-time deployment. Participants will engage in hands-on demonstrations and labs, deploying models with Model Serving and utilizing the serving endpoint for real-time inference.
# MAGIC
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Time | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 22 minutes    | **Model Deployment Fundamentals**    | Model Deployment Strategies </br> Model Deployment with MLflow|
# MAGIC | 53 minutes    | **[Batch Deployment]($./M02 - Batch Deployment)**    | Introduction to Batch Deployment <br>[Demo: Batch Deployment]($./M02 - Batch Deployment/2.1 Demo - Batch Deployment) </br> [Lab: Batch Deployment]($./M02 - Batch Deployment/2.2 Lab - Batch Deployment)|
# MAGIC | 40 minutes    | **[Pipeline Deployment]($./M03 - Pipeline Deployment)**    | Introduction to Pipeline Deployment <br>[Demo: Pipeline Deployment]($./M03 - Pipeline Deployment/3.1a Demo - Pipeline Deployment) </br> [Demo: Inference Pipeline]($./M03 - Pipeline Deployment/3.1b Demo - Inference Pipeline)|
# MAGIC | 112 minutes    | **[Real-time Deployment and Online Stores]($./M04 - Real-time Deployment and Online Stores)**    | Introduction to Real-time Deployment </br> Databricks Model Serving <br>[Demo: Real-time Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.1 Demo - Real-time Deployment with Model Serving) </br> [Demo: Custom Model Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.2 Demo - Custom Model Deployment with Model Serving)</br> [Lab: Real-time Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.3 Lab - Real-time Deployment with Model Serving)|

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