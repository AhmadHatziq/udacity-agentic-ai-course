from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent 
import os
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
product_spec = None 
with open("Product-Spec-Email-Router.txt", "r") as f: 
    product_spec = f.read() 

print(f"Obtained product spec var as: \n {product_spec}")

program_manager_evaluation_criteria = (
                     "The answer should be tasks following this exact structure: " \
                     "Task ID: A unique identifier for tracking purposes\n" \
                     "Task Title: Brief description of the specific development work\n" \
                     "Related User Story: Reference to the parent user story\n" \
                     "Description: Detailed explanation of the technical work required\n" \
                     "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                     "Estimated Effort: Time or complexity estimation\n" \
                     "Dependencies: Any tasks that must be completed first"
)

print(f"Program Manager Criteria: \n{program_manager_evaluation_criteria}")