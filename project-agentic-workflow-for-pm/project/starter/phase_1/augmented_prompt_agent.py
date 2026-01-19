# TODO: 1 - Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(
    openai_api_key = openai_api_key, 
    persona = persona
)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(input_text = prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt. The agent used its own internal foundational (general knowledge to answer)
# - How the system prompt specifying the persona affected the agent's response. The agent's response starts with "Dear students". 
print(f""" \n
Q. What knowledge the agent likely used to answer the prompt. 
A. The agent used its own internal foundational (general knowledge to answer)

Q. How the system prompt specifying the persona affected the agent's response.
A. The agent's response starts with "Dear students". 
      """)
