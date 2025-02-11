# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col, rand, when
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Read the dataset
    dataset_path = f"{DA.paths.datasets.banking}/banking/loan-clean.csv"
    loan_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Add random missing values
    for col_name in loan_df.columns:
        loan_df = loan_df.withColumn(col_name, when(rand() < 0.1, None).otherwise(col(col_name)))

    # Select columns of interest and replace spaces with underscores
    loan_df = loan_df.selectExpr("ID", "Age", "Experience", "Income", "`ZIP Code` as ZIP_Code", "Family", "CCAvg", "Education", "Mortgage", "`Personal Loan` as Personal_Loan", "`Securities Account` as Securities_Account", "`CD Account` as CD_Account", "Online", "CreditCard")

    # Add random 2 duplicate rows
    duplicate_rows = loan_df.orderBy(rand()).limit(2)
    loan_df = loan_df.union(duplicate_rows)

    # Define the path to save the cleaned data
    loan_data_path = f"{DA.catalog_name}.{DA.schema_name}.loan_data_path"

    # Write the cleaned DataFrame to Delta format
    loan_df.write.format("delta").mode("overwrite").saveAsTable(loan_data_path)

    print(f"Cleaned data saved to: {loan_data_path}")

DBAcademyHelper.add_method(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper()  # Create the DA object
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()