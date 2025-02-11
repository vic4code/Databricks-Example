# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# MAGIC %run ../Includes/_stream_factory

# COMMAND ----------

# MAGIC %run ../Includes/_pipeline_config

# COMMAND ----------

class Token:
    def __init__(self, config_path='./var/credentials.cfg'):
        self.config_path = config_path
        self.token = self.get_credentials()

    def get_credentials(self):
        import configparser
        import os

        # Check if the file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"The configuration file was not found at {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)

        if 'DEFAULT' not in config:
            raise KeyError("The section 'DEFAULT' was not found in the configuration file.")

        if 'db_token' not in config['DEFAULT']:
            raise KeyError("The key 'db_token' was not found in the 'DEFAULT' section of the configuration file.")

        token = config['DEFAULT']['db_token']
        
        # Print the token for debugging purposes
        print(f"db_token: {token}")

        return token

# COMMAND ----------

# Use the Token class to get the token
token_obj = Token()
token = token_obj.token

# COMMAND ----------

def init_DA(name, reset=False, pipeline=False):
    
    DA.init()

    if pipeline:
        DA.paths.stream_source = f"{DA.paths.working_dir}/stream_source"
        DA.paths.storage_location = f"{DA.paths.working_dir}/storage_location"
        DA.pipeline_name = f"{DA.unique_name(sep='-')}: ETL Pipeline - {name.upper()}"
        DA.lookup_db = f"{DA.unique_name(sep='_')}_lookup"

        if reset:  # create lookup tables
            import pandas as pd

            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {DA.lookup_db}")
            DA.clone_source_table(f"{DA.lookup_db}.date_lookup", source_name="date-lookup")
            DA.clone_source_table(f"{DA.lookup_db}.user_lookup", source_name="user-lookup")

            df = spark.createDataFrame(pd.DataFrame([
                ("device_id_not_null", "device_id IS NOT NULL", "validity", "bpm"),
                ("device_id_valid_range", "device_id > 110000", "validity", "bpm"),
                ("heartrate_not_null", "heartrate IS NOT NULL", "validity", "bpm"),
                ("user_id_not_null", "user_id IS NOT NULL", "validity", "workout"),
                ("workout_id_not_null", "workout_id IS NOT NULL", "validity", "workout"),
                ("action_not_null","action IS NOT NULL","validity","workout"),
                ("user_id_not_null","user_id IS NOT NULL","validity","user_info"),
                ("update_type_not_null","update_type IS NOT NULL","validity","user_info")
            ], columns=["name", "condition", "tag", "topic"]))
            df.write.mode("overwrite").saveAsTable(f"{DA.lookup_db}.rules")
        
    return DA

None

# COMMAND ----------

import os

# set CWD to folder containing notebook; this can go away for DBR14.x+ as its default behavior
os.chdir('/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()) + '/')

@DBAcademyHelper.add_method
def load_credentials(self):
    import configparser
    c = configparser.ConfigParser()
    c.read(filenames=f"var/credentials.cfg")
    try:
        token = c.get('DEFAULT', 'db_token')
        host = c.get('DEFAULT', 'db_instance')
        os.environ["DATABRICKS_HOST"] = host
        os.environ["DATABRICKS_TOKEN"] = token
    except:
        token = ''
        host = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'

    return token, host    

@DBAcademyHelper.add_method
def get_credentials(self):
    import ipywidgets as widgets

    (current_token, current_host) = self.load_credentials()

    @widgets.interact(host=widgets.Text(description='Host:',
                                        placeholder='Paste workspace URL here',
                                        value = current_host,
                                        continuous_update=False),
                      token=widgets.Password(description='Token:',
                                             placeholder='Paste PAT here',
                                             value = current_token,
                                             continuous_update=False)
    )
    def _f(host='', token=''):
        from urllib.parse import urlparse,urlunsplit

        u = urlparse(host)
        host = urlunsplit((u.scheme, u.netloc, '', '', ''))

        if host and token:
            os.environ["DATABRICKS_HOST"] = host
            os.environ["DATABRICKS_TOKEN"] = token

            contents = f"""
[DEFAULT]
db_token = {token}
db_instance = {host}
            """
            os.makedirs('var', exist_ok=True)
            with open("var/credentials.cfg", "w") as f:
                print(f"Credentials stored ({f.write(contents)} bytes written).")

None

# COMMAND ----------

@DBAcademyHelper.add_method
def update_bundle(self):
    import ipywidgets as widgets
    import os
    
    os.remove("my_project/requirements-dev.txt")
    os.remove("my_project/pytest.ini")
    os.remove("my_project/tests/main_test.py")
    os.rmdir("my_project/tests")
    os.remove("my_project/scratch/exploration.ipynb")
    os.remove("my_project/scratch/README.md")
    os.rmdir("my_project/scratch")
    
    current_cluster = self.client.clusters.get_current()['cluster_id']

    import subprocess
    import json

    def get_policy_id(policy_name):
        # Run the shell command to list cluster policies
        result = subprocess.run(
            ["databricks", "cluster-policies", "list", "--output", "JSON"],
            capture_output=True, text=True
        )

        # Check for errors
        if result.returncode != 0:
            print("Error executing command:", result.stderr)
            return None

        # Load JSON output
        try:
            policies = json.loads(result.stdout)
        except json.JSONDecodeError:
            print("Failed to decode JSON from command output")
            return None

        # Search for the specified policy
        for policy in policies:
            if policy['name'] == policy_name:
                return policy['policy_id']

        return None

    # Specify the policy name
    policy_name = "DBAcademy DLT UC"

    # Get the policy ID
    policy_id = get_policy_id(policy_name)

    if policy_id:
        print("")
    else:
        print(f"Policy named '{policy_name}' not found")

    contents = f"""
# The main job for my_project.
resources:
  jobs:
    my_project_job:
      name: my_project_job

      schedule:
        # Run every day at 8:37 AM
        quartz_cron_expression: '44 37 8 * * ?'
        timezone_id: Europe/Amsterdam

      tasks:
        - task_key: notebook_task
          existing_cluster_id: {current_cluster}
          notebook_task:
            notebook_path: ../src/notebook.ipynb
        
        - task_key: refresh_pipeline
          depends_on:
            - task_key: notebook_task
          pipeline_task:
            pipeline_id: ${{resources.pipelines.my_project_pipeline.id}}
        
        - task_key: main_task
          depends_on:
            - task_key: refresh_pipeline
          existing_cluster_id: {current_cluster}
          spark_python_task:
            python_file: "../src/my_project/main.py"
            """
    with open("my_project/resources/my_project_job.yml", "w") as f:
      print(f"Updated the bundle yml file ({f.write(contents)} bytes written).")

  
    pipelinecontents = f"""
# The DLT pipeline for my_project.
resources:
  pipelines:
    my_project_pipeline:
      name: my_project_pipeline
      target: my_project_${{bundle.environment}}
      clusters:
        - label: default
          policy_id: {policy_id}
          num_workers: 1
        - label: maintenance
          policy_id: {policy_id}
      libraries:
        - notebook:
            path: ../src/dlt_pipeline.ipynb

      configuration:
        bundle.sourcePath: /Workspace/${{workspace.file_path}}/src
            """The 
    with open("my_project/resources/my_project_pipeline.yml", "w") as f:
      print(f"Updated the bundle yml file ({f.write(pipelinecontents)} bytes written).")

# COMMAND ----------

LESSON = "generate_tokens"
DA = init_DA(LESSON, reset=False)

@DBAcademyHelper.add_method
def create_credentials_file(self, db_token, db_instance):
    contents = f"""
db_token = "{db_token}"
db_instance = "{db_instance}"
    """
    self.write_to_file(contents, "credentials.py")

None