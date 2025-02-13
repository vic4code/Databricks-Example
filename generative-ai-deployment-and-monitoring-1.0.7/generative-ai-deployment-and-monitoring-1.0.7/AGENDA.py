# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Generative AI Application Deployment and Monitoring
# MAGIC
# MAGIC This course introduces learners to operationalizing, deploying, and monitoring generative artificial intelligence (AI) applications.. First, learners will develop knowledge and skills in the deployment of generative AI applications using tools like Model Serving. Next, the course will discuss operationalizing generative AI applications following modern LLMOps best practices and recommended architectures. And finally, learners will be introduced to the idea of monitoring generative AI applications and their components using Lakehouse Monitoring.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Course Agenda
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **Model Deployment Fundamentals**    | **Lecture -** Model Management <br> **Lecture -** Deployment Methods|
# MAGIC | **[Batch Deployment]($./01 - Batch Deployment)** | **Lecture -** Introduction to Batch Deployment </br> [**Demo:** Batch Inference using SLM]($./01 - Batch Deployment/1.0 - Batch Inference using SLM) </br> [**Lab:** Batch Inference using SLM]($./01 - Batch Deployment/1.LAB-Batch Inference using SLM) | 
# MAGIC | **[Realtime Model Deployment]($./02 - Realtime Model Deployment)** | **Lecture -** Introduction to Real-time Deployment </br> **Lecture -** Databricks Model Serving </br> [**Demo:** Serving External Models with Model Serving]($./02 - Realtime Model Deployment/2.1 - Serving External Models with Model Serving) </br> [**Demo:** Deploying an LLM Chain to Databricks Model Serving]($./02 - Realtime Model Deployment/2.2 - Deploying an LLM Chain to Databricks Model Serving) </br> [**Lab:** Custom Model Deployment and A/B Testing]($./02 - Realtime Model Deployment/2.LAB - Custom Model Deployment and A-B Testing) | 
# MAGIC | **[AI System Monitoring]($./03 - AI System Monitoring)** | **Lecture -** AI Application Monitoring </br> [**Demo:** Online Monitoring a RAG Chain]($./03 - AI System Monitoring/3.1 - Online Monitoring a RAG Chain) </br> [**Lab:** Online Monitoring]($./03 - AI System Monitoring/3.LAB - Online Monitoring)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>