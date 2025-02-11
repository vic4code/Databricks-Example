# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Machine Learning Pipeline Workflow with Databricks SDK
# MAGIC
# MAGIC In this demo, we configure and initiate a Databricks workflow to run a series of tasks representing an MLOps pipeline. Using the Databricks REST API, we will create, execute, and monitor a series of four notebooks in sequence.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC - **Authentication Setup**
# MAGIC   - Configure and authenticate access to the Databricks REST API.
# MAGIC
# MAGIC - **Pipeline Configuration**
# MAGIC   - Define and initialize a JSON payload for pipeline tasks.
# MAGIC
# MAGIC - **Executing the Pipeline**
# MAGIC   - Trigger the workflow using the Databricks REST API.
# MAGIC
# MAGIC - **Monitoring Task Progress**
# MAGIC   - Track job status and task completion using REST API calls.
# MAGIC
# MAGIC - **Notifications on Completion**
# MAGIC   - Set up email notifications for workflow completion.
# MAGIC
# MAGIC - **Retrieving and Displaying Outputs**
# MAGIC   - Access and visualize JSON data and output files from completed tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC 1. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC   - In the drop-down, select **More**.
# MAGIC   - In the **Attach to an existing compute resource** pop-up, select the first drop-down. You will see a unique cluster name in that drop-down. Please select that cluster.
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC 1. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC 1. Wait a few minutes for the cluster to start.
# MAGIC 1. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo.
# MAGIC
# MAGIC **Important:** Ensure that you have completed the [**`0 - Generate Tokens`**]($./0 - Generate Tokens) notebook beforehand. This will set up the necessary authentication credentials.
# MAGIC
# MAGIC **If you have not done so, complete the [**`0 - Generate Tokens`**]($./0 - Generate Tokens) notebook now, then start this notebook again from the beginning to ensure proper setup.**
# MAGIC
# MAGIC Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.1

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
# MAGIC ## Pipeline Configuration and Setup
# MAGIC In this section, we set up the environment and define paths for each notebook in the pipeline. We also configure the API headers for communication with the Databricks REST API.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC Here, we initialize the foundational configuration by defining:
# MAGIC
# MAGIC - **Databricks Instance URL:** Constructs the instance URL using the workspace configuration.
# MAGIC - **API Token:** Defines the token for authorization, which is essential for authenticated API requests.
# MAGIC - **API Headers:** Sets the headers for the API requests, including the authorization token and content type to ensure secure and correctly formatted requests.

# COMMAND ----------

import requests
import json

# Databricks instance and API token
databricks_instance = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
api_token = token.strip("{}").strip("'")

# Define headers for API requests
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve Cluster ID
# MAGIC
# MAGIC Here, we retrieve the current cluster ID of the environment. This ID is necessary to specify the cluster on which each notebook task will run within the pipeline.

# COMMAND ----------

# Retrieve the current cluster ID
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print("cluster_id -", cluster_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Notebook Paths for Pipeline Tasks
# MAGIC In this section, we dynamically determine the paths to each notebook involved in the pipeline. This setup ensures all task notebooks are linked correctly for sequential execution.
# MAGIC
# MAGIC - C**urrent Notebook Path:** Retrieves the path of the current notebook.
# MAGIC - **Base Path:** Defines a base path for locating all related notebooks in the workflow pipeline.
# MAGIC - **Task Notebook Paths:** Specifies the paths for each task notebook (Data Ingestion, Data Transformation, Advanced Feature Engineering, and Displaying Feature Engineered Data) to ensure they’re accessible within the pipeline.

# COMMAND ----------

# Get the current notebook path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
# Base path for related notebooks
base_path = notebook_path.rsplit('/', 1)[0] + "/2.1 Demo Pipeline - Data Preparation"

# Paths for specific process notebooks
notebook1 = f"{base_path}/2.1a - Data Ingestion"
print(notebook1)
notebook2 = f"{base_path}/2.1b - Data Transformation" 
print(notebook2)
notebook3 = f"{base_path}/2.1c - Advanced Feature Engineering (Build_Features)" 
print(notebook3)
notebook4 = f"{base_path}/2.1d - Displaying Feature Engineered Data"
print(notebook4)
# Store the notebook paths in a list for easy iteration
notebook_paths = [notebook1, notebook2, notebook3, notebook4]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Pipeline Tasks Creation
# MAGIC
# MAGIC In this section, we use the `Databricks REST API` to define tasks for each notebook in the pipeline. Each task is dependent on the previous one, ensuring sequential execution. An email notification is sent upon successful job completion.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC The **`create_job_with_tasks`** function allows for a modular approach to building workflows. Each task is specified by a unique **`task_key`**, and dependencies are established between tasks. 
# MAGIC
# MAGIC > **Important:** Email notifications ensures that you are alerted as soon as the job completes.

# COMMAND ----------

# Function to create a job with tasks, including email notifications for success
def create_job_with_tasks(job_name, notebook_paths, cluster_id):
    tasks = []
    for i, notebook_path in enumerate(notebook_paths):
        task = {
            "task_key": f"task_{i+1}",
            "notebook_task": {
                "notebook_path": notebook_path
            },
            "existing_cluster_id": cluster_id
        }
        
        # Add dependencies for each subsequent task
        if i > 0:
            task["depends_on"] = [{"task_key": f"task_{i}"}]
        
        tasks.append(task)
    
    # Job payload with email notifications
    job_payload = {
        "name": f"{DA.username}-testing-workflow-job",
        "tasks": tasks,
        "email_notifications": {
            "on_success": [DA.username],  # Sends an email notification on success to the user
            "no_alert_for_skipped_runs": False
        },
        "notification_settings": {
            "no_alert_for_skipped_runs": False,
            "no_alert_for_canceled_runs": False
        }
    }
    
    # Create the job using the Databricks REST API
    response = requests.post(
        f"{databricks_instance}/api/2.1/jobs/create",
        headers=headers,
        data=json.dumps(job_payload)
    )
    
    if response.status_code == 200:
        job_id = response.json().get("job_id")
        print(f"Job '{job_name}' created with ID: {job_id}")
        return job_id
    else:
        print(f"Error creating job: {response.text}")
        return None

# Define the job name and create the job with tasks
job_name = "ML Pipeline Workflow"
job_id = create_job_with_tasks(job_name, notebook_paths, cluster_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Running the Pipeline Job
# MAGIC This section triggers the workflow job after all tasks are defined, initiating the pipeline's execution.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC The `run_job` function uses the Databricks REST API to start the workflow job by specifying the `job_id`. The output `run_id` is important for tracking this specific job execution in future requests. Use this function to automate the process and ensure timely task execution.

# COMMAND ----------

def run_job(job_id):
    # Send a POST request to start the job with the given job_id
    response = requests.post(
        f"{databricks_instance}/api/2.1/jobs/run-now",
        headers=headers,
        data=json.dumps({"job_id": job_id})
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract the run_id from the response
        run_id = response.json().get("run_id")
        print(f"Job '{job_name}' started with Run ID: {run_id}")
        return run_id
    else:
        # Print the error message if the request failed
        print(f"Error starting job: {response.text}")
        return None

# Start the job and get the run_id
run_id = run_job(job_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accessing the Workflow
# MAGIC After the job is started, you can access the job details and log history directly from the Databricks Jobs interface. This final cell provides a direct link to the workflow job, allowing instructors or users to review task execution details.
# MAGIC
# MAGIC **Instruction:**
# MAGIC
# MAGIC The link generated is a direct access point to monitor the job execution and review logs or any error messages. This link is valuable for troubleshooting or assessing job performance metrics.

# COMMAND ----------

# Print the link to the workflow
workspace_id = databricks_instance.split("//")[1].split(".")[0]
print(f"Job started successfully. You can monitor the progress at: https://{databricks_instance.split('//')[1]}/jobs/{job_id}?o={workspace_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Retrieving Task Outputs and Visualization
# MAGIC This function polls the job status to ensure it completes successfully, then retrieves the JSON data output and visualizes the PNG file generated in the final notebook.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC The `get_final_task_output` function checks the job's state and result status, displaying the **JSON** data and visualization **PNG** file. It also sends an email upon job completion, useful for monitoring and collaboration.

# COMMAND ----------

# Function to wait for the job to complete and retrieve the output from JSON and PNG files
import time
import os
def get_final_task_output(run_id, 
                          json_output_path="2.1 Demo Pipeline - Data Preparation/feature_engineered_output_with_visualization_data.json", 
                          png_output_path="2.1 Demo Pipeline - Data Preparation/total_assets_by_credit_usage_category.png"):
    from PIL import Image
    import matplotlib.pyplot as plt

    # Polling to check job status
    while True:
        response = requests.get(
            f"{databricks_instance}/api/2.1/jobs/runs/get",
            headers=headers,
            params={"run_id": run_id}
        )
        
        if response.status_code == 200:
            job_status = response.json().get("state", {}).get("life_cycle_state")
            result_state = response.json().get("state", {}).get("result_state")
            print(f"Job status: {job_status}, Result state: {result_state}")
            
            if job_status == "TERMINATED":
                if result_state == "SUCCESS":
                    # Attempt to read the JSON output file once the job is successful
                    try:
                        # Display JSON file contents
                        with open(json_output_path, 'r') as file:
                            task_output = json.load(file)
                            print("\n=======\nOutput of Final Task (JSON Data):\n")
                            print(json.dumps(task_output, indent=4))
                            
                        # Confirm the existence of the PNG image file
                        if os.path.exists(png_output_path):
                            # Display the bar graph from the PNG file
                            try:
                                img = Image.open(png_output_path)
                                plt.imshow(img)
                                plt.axis('off')
                                plt.show()
                            except Exception as e:
                                print(f"Error displaying the bar graph: {e}")
                            print(f"\nVisualization image saved at: {png_output_path}")
                        else:
                            print("Visualization image file not found.")
                    except Exception as e:
                        print(f"Error reading final task output: {e}")
                else:
                    print("Job did not complete successfully.")
                break
        else:
            print(f"Error getting job status: {response.text}")
            break

        time.sleep(20)  # Wait before polling again

# Retrieve and print the output of the final task
get_final_task_output(run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conclusion
# MAGIC In this notebook, we demonstrated how to configure and execute a Databricks workflow to implement an MLOps pipeline. Using Databricks' REST API, we created a sequence of tasks to handle data ingestion, transformation, feature engineering, and data visualization. The workflow was automated to run each notebook in sequence, and a monitoring function provided real-time updates, including a visualization output and email notifications upon completion.
# MAGIC
# MAGIC This pipeline represents a scalable approach to managing machine learning workflows, aligning with the MLOps principles.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>