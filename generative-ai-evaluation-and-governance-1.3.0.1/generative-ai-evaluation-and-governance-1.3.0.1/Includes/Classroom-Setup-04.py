# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic init, schemas and catalogs

# define host and token
DA.scope_name = f"genai_training_{DA.unique_name(sep='_')}"
w = WorkspaceClient()
try:
    w.secrets.delete_scope(DA.scope_name)
except:
    pass

try:
    w.secrets.create_scope(DA.scope_name)
    w.secrets.put_secret(DA.scope_name, key="depl_demo_host", string_value='https://' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').getOrElse(None))
    w.secrets.put_secret(DA.scope_name, key="depl_demo_token", string_value=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None))
except:
    print(f'unable to create secret scope {DA.scope_name}')

# COMMAND ----------

import pandas as pd
from io import StringIO

eval_set = """request,expected_response,evolution_type,episode_done
"What are the limitations of symbolic planning in task and motion planning, and how can leveraging large language models help overcome these limitations?","Symbolic planning in task and motion planning can be limited by the need for explicit primitives and constraints. Leveraging large language models can help overcome these limitations by enabling the robot to use language models for planning and execution, and by providing a way to extract and leverage knowledge from large language models to solve temporally extended tasks.",simple,TRUE
"What are some techniques used to fine-tune transformer models for personalized code generation, and how effective are they in improving prediction accuracy and preventing runtime errors? ","The techniques used to fine-tune transformer models for personalized code generation include ﬁne-tuning transformer models, adopting a novel approach called Target Similarity Tuning (TST) to retrieve a small set of examples from a training bank, and utilizing these examples to train a pretrained language model. The effectiveness of these techniques is shown in the improvement in prediction accuracy and the prevention of runtime errors.",simple,TRUE
How does the PPO-ptx model mitigate performance regressions in the few-shot setting?,"The PPO-ptx model mitigates performance regressions in the few-shot setting by incorporating pre-training and fine-tuning on the downstream task. This approach allows the model to learn generalizable features and adapt to new tasks more effectively, leading to improved few-shot performance.",simple,TRUE
How can complex questions be decomposed using successive prompting?,"Successive prompting is a method for decomposing complex questions into simpler sub-questions, allowing language models to answer them more accurately. This approach was proposed by Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner in their paper 'Successive Prompting for Decomposing Complex Questions', presented at EMNLP 2022.",simple,TRUE
"Which entity type in Named Entity Recognition is likely to be involved in information extraction, question answering, semantic parsing, and machine translation?",Organization,reasoning,TRUE
What is the purpose of ROUGE (Recall-Oriented Understudy for Gisting Evaluation) in automatic evaluation methods?,"ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used in automatic evaluation methods to evaluate the quality of machine translation. It calculates N-gram co-occurrence statistics, which are used to assess the similarity between the candidate text and the reference text. ROUGE is based on recall, whereas BLEU is based on accuracy.",simple,TRUE
"What are the challenges associated with Foundation SSL in CV, and how do they relate to the lack of theoretical foundation, semantic understanding, and explicable exploration?","The challenges associated with Foundation SSL in CV include the lack of a profound theory to support all kinds of tentative experiments, and further exploration has no handbook. The pretrained LM may not learn the meaning of the language, relying on corpus learning instead. The models cannot reach a better level of stability and match different downstream tasks, and the primary method is to increase data, improve computation power, and design training procedures to achieve better results. The lack of theoretical foundation, semantic understanding, and explicable exploration are the main challenges in Foundation SSL in CV.",simple,TRUE
How does ChatGPT handle factual input compared to GPT-3.5?,"ChatGPT handles factual input better than GPT-3.5, with a 21.9% increase in accuracy when the premise entails the hypothesis. This is possibly related to the preference for human feedback in ChatGPT's RLHF design during model training.",simple,TRUE
What are some of the challenges in understanding natural language commands for robotic navigation and mobile manipulation?,"Some challenges in understanding natural language commands for robotic navigation and mobile manipulation include integrating natural language understanding with reinforcement learning, understanding natural language directions for robotic navigation, and mapping instructions and visual observations to actions with reinforcement learning.",simple,TRUE
"How does chain of thought prompting elicit reasoning in large language models, and what are the potential applications of this technique in neural text generation and human-AI interaction?","The context discusses the use of chain of thought prompting to elicit reasoning in large language models, which can be applied in neural text generation and human-AI interaction. Specifically, researchers have used this technique to train language models to generate coherent and contextually relevant text, and to create transparent and controllable human-AI interaction systems. The potential applications of this technique include improving the performance of language models in generating contextually appropriate responses, enhancing the interpretability and controllability of AI systems, and facilitating more effective human-AI collaboration.",simple,TRUE
"Using the given context, how can the robot be instructed to move objects around on a tabletop to complete rearrangement tasks?","The robot can be instructed to move objects around on a tabletop to complete rearrangement tasks by using natural language instructions that specify the objects to be moved and their desired locations. The instructions can be parsed using functions such as parse_obj_name and parse_position to extract the necessary information, and then passed to a motion primitive that can pick up and place objects in the specified locations. The get_obj_names and get_obj_pos APIs can be used to access information about the available objects and their locations in the scene.",reasoning,TRUE
"How can searching over an organization's existing knowledge, data, or documents using LLM-powered applications reduce the time it takes to complete worker activities?","Using AI tools to search through a company's information can significantly cut down the time workers need to finish tasks. These tools quickly find and provide the necessary details by scanning large amounts of data efficiently.",simple,TRUE
"""

obj = StringIO(eval_set)
DA.eval_df = pd.read_csv(obj)

# COMMAND ----------

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")