# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
#DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()

print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")