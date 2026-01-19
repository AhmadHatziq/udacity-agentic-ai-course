# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent 
import os
from dotenv import load_dotenv
import sys 

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
product_spec = None 
with open("Product-Spec-Email-Router.txt", "r") as f: 
    product_spec = f.read() 

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A Development Plan for a product contains all these components"
)

# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key = openai_api_key, knowledge = knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always in the format: 'As a [type of user], I want [an action or feature] so that [benefit/value].' "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"Product Specification: {product_spec}" # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
)

# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_product_manager, knowledge=knowledge_product_manager)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
product_manager_evaluator_persona = "You are an evaluation agent that checks the answers of other worker agents performing as a product manager"
product_manager_evaluator_criteria = "The answer should be user stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=product_manager_evaluator_persona, 
                                                   evaluation_criteria=product_manager_evaluator_criteria, worker_agent=product_manager_knowledge_agent, 
                                                   max_interactions=10)

# Program Manager - Knowledge Augmented Prompt Agent
# NTS: Product manager vs Program manager: Product manager is higher 
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups." \
    "Features consist of the feature name, description, functionalities and user benefits." \
    "Use the provided information when it is relevant." \
    "If the information is incomplete, ambiguous or missing, rely on your foundational general knowledge and fill in the gaps with the most reasonable inference"
)
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_program_manager, knowledge=knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents performing as a Program Manager (responsible for defining features of a product)"

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
program_manager_evaluation_criteria = (
                      "The answer should be product features that follow the following structure: "\
                      "Feature Name: A clear, concise title that identifies the capability\n" \
                      "Description: A brief explanation of what the feature does and its purpose\n" \
                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
                      "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona_program_manager_eval, 
                                                   evaluation_criteria=program_manager_evaluation_criteria, worker_agent=program_manager_knowledge_agent, 
                                                   max_interactions=10)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story or feature." \
    "Tasks consists of a title, related user sotry, description, acceptance criteria, effort, dependencies and suggestions of technologies to use" \
    "Use the provided information when it is relevant." \
    "If the information is incomplete, ambiguous or missing, rely on your foundational general knowledge and fill in the gaps with the most reasonable inference"
)

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent =  KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_dev_engineer, knowledge=knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents. In this case, a development agent defining dev tasks for a product"
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
dev_engineer_evaluation_criteria = (
                     "The answer should be tasks following this exact structure. Must have the fields Task ID, Title, Related User Story, Description, Acceptance Criteria, Effort, Dependencies, Technologies: " \
                     "Task ID: Generate a urandom unique string for each task\n" \
                     "Task Title: Brief description of the specific development work\n" \
                     "Related User Story: Reference to the parent user story\n" \
                     "Description: Detailed explanation of the technical work required\n" \
                     "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                     "Estimated Effort: Time or complexity estimation\n" \
                     "Dependencies: Any tasks that must be completed first\n" \
                     "Technologies Needed: Any specific technologies (give examples) that can fulfill the task eg Oracle DB 19c, Jira, Python Django\n"
)
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona_dev_engineer_eval, 
                                                   evaluation_criteria=dev_engineer_evaluation_criteria, worker_agent=development_engineer_knowledge_agent, 
                                                   max_interactions=10)

# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). 
# Assign this list to the routing_agent's 'agents' attribute.
agents = [
    {
        "name": "Product Manager Agent",
        "description": "Responsible for defining product personas and USER STORIES only. Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x) 
    },
    {
        "name": "Program Manager Agent",
        "description": "Responsible for defining the FEATURES for a product. Only concerned with features.",
        "func": lambda x: program_manager_support_function(x) 
    },
    {
        "name": "Development Engineer Agent",
        "description": "Responsible for only defining the development or implementation TASKS for a product. Does not define features or user stories or group stories.",
        "func": lambda x: dev_engineer_support_function(x)
    }
]
routing_agent = RoutingAgent(openai_api_key, agents)

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
def product_manager_support_function(input_query: str) -> str:

    # Call the agent 
    first_response = product_manager_knowledge_agent.respond(input_query)

    # Evaluate the response. Will internally loop 
    eval_result_dict = product_manager_evaluation_agent.evaluate(first_response)

    # Return final validated response 
    return str(eval_result_dict["final_response"])

def program_manager_support_function(input_query: str) -> str: 
    first_response = program_manager_knowledge_agent.respond(input_query)
    eval_result_dict = program_manager_evaluation_agent.evaluate(first_response)
    return str(eval_result_dict["final_response"])

def dev_engineer_support_function(input_query: str) -> str: 
    first_response = development_engineer_knowledge_agent.respond(input_query)
    eval_result_dict = development_engineer_evaluation_agent.evaluate(first_response)
    return str(eval_result_dict["final_response"])

# End of support function definitions

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = f"""
You are planning a development workflow for this product: 

# Product Spec: 
{product_spec}

Return a high-level list of steps to:
- What user stories to define 
- Define Features related to those user stories
- Define Development Tasks to support those features

Your response must only be a list of steps, with each newline representing a single step. 
Each step must not refer to any other step, unless it is mentioned fully.  
Eg follow the format below: 
1. User Story - Identify personas such as Customer Support, Privileged User
2. User Story - Identify personas such as IT Administrator, SME (Subject Matter Expert)
3. Feature - Email ingestion system 
4. Feature - Integration with corporate Active Directory to support RBAC (Role Based Access Control) 
5. Feature - Email classification system 
6. Feature - High availability system on the cloud 
7. Task - Implement database using techologies MongoDB on AWS Cloud
8. Task - Create codebase using Python Django framework 
9. Task - xxx
""" 

# "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).

workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"==== WORKFLOW STEPS: {workflow_steps} ======")

'''
# For testing, iterate over 1 feature, user story and task 
# single_user_story = next((x for x in workflow_steps if "story" in x.lower()), None)
# single_feature = next((x for x in workflow_steps if "feature" in x.lower()), None)
single_task = next((x for x in workflow_steps if "task" in x.lower()), None)
workflow_steps = [single_task]
# workflow_steps = [single_user_story, single_feature, single_task]
print("\n Final workflow steps to execute: ", workflow_steps)
'''

# Loop through the steps and assign to routing agent 
completed_steps = []
counter = 0  
for step in workflow_steps: 
    
    print("===== WORKFLOW STEP: =======\n", step, "\n==============\n")

    # Get best agent's response 
    best_agent_response = routing_agent.get_best_agent(step)
    print("===== WORKFLOW RESULT ====== \n", best_agent_response, "\n============\n")
    completed_steps.append(best_agent_response)

    # Debugging counter 
    counter += 1 
    # if counter >= 5: 
    #    break 

# Print the workflow & completed steps: 
print(f"============= COMPLETED STEPS: ==========")
for i in range(len(completed_steps)): 
    current_step = completed_steps[i]
    workflow_step = workflow_steps[i]

    print(f"========== STEP: {i+1} - {workflow_step} ============") 
    print(current_step)
    print(f"=================================")

