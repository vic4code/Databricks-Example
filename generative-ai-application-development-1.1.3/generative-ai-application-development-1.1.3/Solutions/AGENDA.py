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
# MAGIC ## Generative AI Application Development
# MAGIC
# MAGIC This course provides participants with information and practical experience in building advanced LLM (Large Language Model) applications using multi-stage reasoning LLM chains and agents. In the initial section, participants will learn how to decompose a problem into its components and select the most suitable model for each step to enhance business use cases. Following this, participants will construct a multi-stage reasoning chain utilizing LangChain and HuggingFace transformers. Finally, participants will be introduced to agents and will design an autonomous agent using generative models on Databricks.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **[Foundations of Compound AI Systems]($./01 - Foundations of Compound AI Systems)**    | **Lecture -** Foundations of Compound AI Systems </br> **Lecture -** 01 - Foundations of Compound AI Systems <br>[Demo: Deconstruct and Plan a Use Case]($./01 - Foundations of Compound AI Systems/1.1 - Deconstruct and Plan a Use Case) <br>[Lab: Planning an AI System for Product Quality Complaints]($./01 - Foundations of Compound AI Systems/1.LAB - Planning an AI System for Product Quality Complaints)|
# MAGIC | **[Building Multi-stage Reasoning Chains]($./02 - Building Multi-stage Reasoning)** | **Lecture -** Introduction to Multi-stage Reasoning Chains </br> [Demo: Building Multi-stage Reasoning Chain in Databricks]($./02 - Building Multi-stage Reasoning/2.1 - Building Multi-stage AI Systems in Databricks) </br> [Lab: Building Multi-stage AI System]($./02 - Building Multi-stage Reasoning/2.LAB - Building Multi-stage AI System) | 
# MAGIC | **[Agents and Cognitive Architectures]($./03 - Agents and Cognitive Architectures)** | **Lecture -** Introduction to Agents  </br> [Demo: Agent Design in Databricks]($./03 - Agents and Cognitive Architectures/3.1 - Agent Design in Databricks) </br> [Lab: Create a ReAct Agent]($./03 - Agents and Cognitive Architectures/3.LAB - Create a ReAct Agent) | 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>