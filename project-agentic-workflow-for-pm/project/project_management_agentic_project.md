This project implements a two-phase agentic workflow system for technical project management. It builds a reusable library of AI “agents” (Phase 1), then orchestrates them into a multi-step workflow that converts a product specification into user stories → features → engineering tasks (Phase 2).

The code is designed for the Udacity workspace setup and uses OpenAI via Vocareum (base_url="https://openai.vocareum.com/v1").

```
Repository Structure
project/
├─ README.md
├─ project_overview.md
├─ reflection.md
├─ requirements.txt
├─ chunks-*.csv
├─ embeddings-*.csv
└─ starter/
   ├─ README.md
   ├─ phase_1/
   │  ├─ README.md
   │  ├─ workflow_agents/
   │  │  └─ base_agents.py
   │  ├─ direct_prompt_agent.py
   │  ├─ augmented_prompt_agent.py
   │  ├─ knowledge_augmented_prompt_agent.py
   │  ├─ rag_knowledge_prompt_agent.py
   │  ├─ evaluation_agent.py
   │  ├─ routing_agent.py
   │  ├─ action_planning_agent.py
   │  ├─ phase_1_output.txt
   │  └─ phase_1_output.docx
   └─ phase_2/
      ├─ README.md
      ├─ Product-Spec-Email-Router.txt
      ├─ agentic_workflow.py
      ├─ phase_2_output.txt
      ├─ testing_1.py
      └─ workflow_agents/
         └─ base_agents.py
```

Dependencies

# Defined in requirements.txt:
- openai==1.78.1
- python-dotenv==1.1.0
- pandas==2.2.3

# Environment Setup

Create a .env file (location depends on how you run it; Phase 2 uses load_dotenv() and reads from environment):
```
OPENAI_API_KEY=your_key_here
```

# Phase 1: Agent Library (workflow_agents/base_agents.py)

Phase 1 implements a reusable set of agent classes that demonstrate common agentic patterns:

1) DirectPromptAgent
- Sends the user’s prompt directly to the chat model (gpt-3.5-turbo)
- No system prompt, no extra context
- Returns only the response text

2) AugmentedPromptAgent

- Adds a persona via a system prompt

- Forces the model to “forget previous context”

- Returns only the response text

3) KnowledgeAugmentedPromptAgent

Adds:

- persona

- explicit knowledge block

Instructs the model to answer using only the given knowledge, otherwise return an “insufficient knowledge” response.

4) RAGKnowledgePromptAgent

A lightweight Retrieval-Augmented Generation flow implemented locally using embeddings:

- Splits long text into chunks (chunk_text)

- Saves chunks to chunks-<timestamp_uuid>.csv

- Computes embeddings using text-embedding-3-large, saves to embeddings-<timestamp_uuid>.csv

- For a query, embeds the prompt, computes cosine similarity, selects best chunk, and answers using only that chunk

The repo includes some pre-generated chunks-*.csv and embeddings-*.csv outputs.

5) EvaluationAgent

A “judge + refine loop” wrapper around another agent:

- Calls a worker agent to generate an answer

- Evaluates it against criteria using an LLM call (temperature 0)

- If it fails, generates correction instructions and re-prompts the worker

- Loops up to max_interactions

- Returns:

    - final_response

    - eval_result

    - iteration_count

6) RoutingAgent

A semantic router:

- Embeds the user prompt
- Embeds each candidate agent route description
- Uses cosine similarity to select the “best” route
- Calls that route’s function (func) with the prompt

7) ActionPlanningAgent

A step extractor:

- Given a workflow prompt, returns a list of steps (newline-split)

- Uses a system prompt describing how to extract only steps from known “knowledge”

# Phase 2: Agentic Workflow Orchestration (phase_2/agentic_workflow.py)

Phase 2 combines the agents into a project-management workflow for the provided product specification:

Inputs

- Product-Spec-Email-Router.txt (loaded into product_spec)

- A high-level workflow prompt that asks for:

1. User stories

2. Features grouped from stories

3. Development tasks supporting features

## Workflow Design

1. ActionPlanningAgent extracts a list of steps from the workflow prompt.

2. A RoutingAgent routes each step to the most appropriate “role team”:

- Product Manager team → produces user stories

- Program Manager team → produces features

- Development Engineer team → produces tasks

3. Each “team” is implemented as:

- KnowledgeAugmentedPromptAgent (role persona + role knowledge)

- plus an EvaluationAgent enforcing a strict output format

# Output

- For each extracted step:

    - prints the step
    
    - routes it
    
    - prints the routed agent’s validated response
- At the end, prints a combined view of completed steps.

# How to Run

From the Phase 2 folder:
```
cd project/starter/phase_2
python agentic_workflow.py
```

Phase 1 scripts can be run individually in project/starter/phase_1/ to test each agent.

# Notes / Gotchas

The OpenAI client is configured to use Vocareum:

base_url="https://openai.vocareum.com/v1"

Models used:

Chat: gpt-3.5-turbo

Embeddings: text-embedding-3-large

EvaluationAgent currently returns the latest worker response + last evaluation text; it does not return the improved response separately (it does update the prompt iteratively, but the returned field is the last worker response from the loop).