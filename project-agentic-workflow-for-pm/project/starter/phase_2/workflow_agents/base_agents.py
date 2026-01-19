# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
import sys 
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(
            base_url = "https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
            )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # TODO: 3 - Specify the model to use (gpt-3.5-turbo)
            messages=[
                # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content.strip()
        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.persona = persona 
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(
            base_url = "https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
            )

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {"role": "system", "content": f"Forget any previous conversational context and assume the following persona: {self.persona}."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()  # TODO: 4 - Return only the textual content of the response, not the full JSON payload.

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    The Knowledge Augmented Prompt Agent is designed to generate responses
    by combining a defined persona with explicit, provided knowledge.
    """
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge 

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(
            base_url = "https://openai.vocareum.com/v1",
            api_key=self.openai_api_key
            )
        
        # Construct system prompt with knowledge 
        system_prompt = (
            f"You must follow ALL instructions below:\n"
            f"1. Forget all previous context.\n"
            f"2. You are now assuming the following persona: {self.persona}\n"
            f"3. You MUST answer strictly using ONLY the knowledge provided below. Do not use any other knowledge.\n"
            f"4. If the knowledge does not contain the answer, reply exactly: 'The knowledge provided is insufficient.'\n"
            # f"3. You MUST answer using ONLY the knowledge provided below, not your own, when it is relevant.\n"
            # f"4. If the information is incomplete, ambiguous or missing, rely on your foundational general knowledge and fill in the gaps with the most reasonable inference\n"
            # f"5. If the knowledge does not contain the answer, reply: 'The knowledge provided is insufficient.'\n\n"
            f"--- KNOWLEDGE START ---\n"
            f"{self.knowledge}\n"
            f"--- KNOWLEDGE END ---"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 2 - Construct a system message including:
                #           - The persona with the following instruction:
                #             "You are _persona_ knowledge-based assistant. Forget all previous context."
                #           - The provided knowledge with this instruction:
                #             "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                #           - Final instruction:
                #             "Answer the prompt based on this knowledge, not your own."
                {"role": "system", "content": system_prompt},
                
                # TODO: 3 - Add the user's input prompt here as a user message.
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float", 
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, chunk_id, start, prev_start = [], 0, 0, -1
        counter = 0 
        # print("Len text: ", len(text))
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            prev_start = start
            start = end - self.chunk_overlap
            chunk_id += 1

            counter += 1 
            # print(f"start: {start}, end: {end}, chunk_id: {chunk_id}")

            # Terminate if "start" does not change to avoid infinite loop 
            if start == prev_start: 
                # print("Value of start did not change, breaking to avoid infinite loop.")
                break  
            
            # Safety break 
            if counter == 50: 
                break 

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a local CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )
        print(f"QUERY: {prompt}")
        print(f"Best relevant chunk from embedding cosine similarity: \n{best_chunk}")

        return response.choices[0].message.content
    
class EvaluationAgent:
    """
    This agent will assess responses from another agent, against a set of criteria, 
    and refine the response via iterative feedback. 
    Returns the evaluation results, a dict with keys: 
        - "final_response"
        - "eval_result"
        - "iteration_count"
    """
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key, self.persona, self.evaluation_criteria, self.worker_agent, self.max_interactions = openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        MAX_INTERACTIONS = self.max_interactions
        for i in range(MAX_INTERACTIONS): # TODO: 2 - Set loop to iterate up to the maximum number of interactions:
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate) # TODO: 3 - Obtain a response from the worker agent
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"  # TODO: 4 - Insert evaluation criteria here
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
                messages= [
                    {"role": "system", "content": "Your role is to evaluate a response. Respond Yes or No, and the reason why it does or doesn't meet the criteria."}, 
                    {"role": "user", "content": eval_prompt}
                ], 
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    # TODO: 6 - Define the message structure sent to the LLM to generate correction instructions (use temperature=0)
                    messages=[
                        {"role": "system", "content": "Your role is to correct a response."}, 
                        {"role": "user", "content": instruction_prompt}
                    ], 
                    temperature=0 
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
            "final_response": response_from_worker, 
            "eval_result": evaluation, 
            "iteration_count": i
        }   

class RoutingAgent():
    """
    Directs user prompts to the most appropriate specialized agent from its 'agents', based on 
    semantic similarity between prompts and descriptions of each agent. 
    """

    def __init__(self, openai_api_key, agents):
        """
        Each agent will have the keys: name, description, func (lambda func for response)
        """
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents # TODO: 1 - Define an attribute to hold the agents, call it agents
        
    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)

        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float", 
        )

        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding 

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def get_best_agent(self, user_prompt): 
        """
        Based on user prompt, find best agent (based on cosine similarity of embedding vectors). 
        Returns best agent's response function. 
        """
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_prompt)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(str(agent['description']))

            if agent_emb is None:
                continue
            
            # Calculate cosine similarity between input_embedding and agent description embedding 
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score: 
                best_score = similarity 
                best_agent = agent 

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")

        # Call best_agent's function on user input and return 
        return best_agent["func"](user_prompt)

class ActionPlanningAgent:
    """
    Given user prompt, will get a list of steps required to execute the user's tasks. 
    """

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge 

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)

        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        system_prompt = f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}
            ], 
            temperature=0 
        )
        response_text = response.choices[0].message.content  # TODO: 4 - Extract the response text from the OpenAI API response

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps
