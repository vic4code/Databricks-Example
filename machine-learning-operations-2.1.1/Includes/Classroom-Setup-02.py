# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper()  # Create the DA object
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

# COMMAND ----------

# Import necessary libraries
import os
import configparser
import ipywidgets as widgets
from urllib.parse import urlparse, urlunsplit

# Token Class Definition
class Token:
    def __init__(self, config_path='./M02 - Architecting MLOps Solutions/var/credentials.cfg'):
        self.config_path = config_path
        self.token = self.get_credentials()

    def get_credentials(self):
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

def init_DA(name, reset=True, pipeline=False):

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

# COMMAND ----------

# Initialize Lesson with Configurations
LESSON = "generate_tokens"
DA = init_DA(LESSON, reset=False)

# COMMAND ----------

# Create Credentials File Method
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
def create_credentials_file(self, db_token, db_instance):
    contents = f"""
db_token = "{db_token}"
db_instance = "{db_instance}"
    """
    self.write_to_file(contents, "credentials.py")
None

# Get Credentials Method
@DBAcademyHelper.add_method
def get_credentials(self):
    (current_token, current_host) = self.load_credentials()

    @widgets.interact(host=widgets.Text(description='Host:',
                                        placeholder='Paste workspace URL here',
                                        value=current_host,
                                        continuous_update=False),
                      token=widgets.Password(description='Token:',
                                             placeholder='Paste PAT here',
                                             value=current_token,
                                             continuous_update=False)
    )
    def _f(host='', token=''):
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
            # Save credentials in the new location
            os.makedirs('./var', exist_ok=True)
            with open("./var/credentials.cfg", "w") as f:
                print(f"Credentials stored ({f.write(contents)} bytes written).")

None

# COMMAND ----------

# Initialize the next lesson and load credentials
LESSON = "using_cli"
DA = init_DA(LESSON, reset=False, pipeline=True)

# Load Credentials
(db_token, db_instance) = DA.load_credentials()