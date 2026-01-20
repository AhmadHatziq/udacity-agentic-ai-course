# udacity-agentic-ai-course
Project files used for the Agentic AI Nanodegree from Udacity 

There are 5 projects for this nanodegree: 
1. [Trip Planner Agent (ReACT)](./project-trip-planner/project_trip_planner.md)
2. [Agentic Workflow for Project Management (Multi Agent Workflow with Orchestration)](./project-agentic-workflow-for-pm/project/project_management_agentic_project.md)
3. [Video Game Agent with access to RAG tool, uses state machines](./project-udaplay/project_udaplay.md)
4. [Multi-Agent Workflow for a Paper Company, uses cosine similarity to find similar sales quotes](./project-paper-company/project_paper_company.md) 
5. [Chatbot for scraping websites and obtaining LLM pricing data (MCP Server Tools, Client)](./project-pricescout/developer_guide.md)

# 1) Trip Planner Agent 
- A notebook project building a trip-planning agent with constraints + evaluatio + ReACT-style loop 

## Core Idea (Architecture)
- `ItineraryAgent` produces a structured `TravelPlan` JSON using:
    - vacation info (budget, dates, traveler interests)
    - weather data
    - an activities list
- Then an evaluation layer checks things like:
    - total cost correctness
    - matching real events (avoid hallucination)
    - interest coverage
    - weather compatibility
- Then an ItineraryRevisionAgent uses a ReAct loop:
    - THOUGHT → ACTION (tool call) → OBSERVATION
    - iterates until it satisfies traveler feedback (e.g., “2 activities per day”)

## Key components:
- `project_starter.ipynb`: main implementation (prompts, agent behavior, tools, evaluation).
- `project_lib.py`: helper library + ChatAgent, mocked APIs for weather/activities.

# 2) Agentic Workflow for Project Management 

## Core idea (Architecture)
- Phase 1: Build a small library of reusable agent classes (prompt agent, persona agent, knowledge agent, RAG agent, evaluation agent, routing agent, action planning agent).
- Phase 2: Orchestrate multiple “job-role” agents (Product Manager, Program Manager, Dev Engineer) to transform a product spec into:
    - user stories → features → engineering tasks
- Agent orchestration + Evaluation Loops + Routing. Mostly prompt-based but structured and modular 

## Key Components 
- `workflow_agents/base_agents.py`: implementations of the agent types (incl. embeddings + chunking for RAG agent).
- `starter/phase_2/agentic_workflow.py`: the orchestrator:
    - ActionPlanningAgent creates a step list
    - RoutingAgent chooses the correct role-agent per step
    - Each role-agent is paired with an EvaluationAgent to validate outputs

## Outputs / Artifacts 
- `phase_1_output.txt/.docx`, `phase_2_output.txt`
- Product spec: `Product-Spec-Email-Router.txt`
- Embeddings/chunk CSVs (generated RAG assets)

# 3) Udaplay Agent (AI Game Research Agent with RAG)
- A RAG-powered “game research assistant” that answers questions from a local ChromaDB, with optional web-search tooling.
- Implements state machines, modular libraries, tool calling, memory, RAG integration 

## Core Idea (Architecture)
- Part 1: ingest game JSON documents → embed → store in ChromaDB (persistent).
- Part 2: an agent that:
    - retrieves from vector DB (retrieve_game)
    - evaluates retrieval quality (evaluate_retrieval)
    - optionally uses web search if local knowledge is insufficient (game_web_search)
    - maintains state across runs

## Key Components 
- `lib/agents.py`: a clean state-machine-driven agent:
    - prepare messages → LLM step → tool step → loop/terminate
    - short-term memory stores “Run” objects per session
- `lib/vector_db.py, lib/rag.py, lib/tooling.py, lib/state_machine.py`: the internal framework pieces
- `games/*.json`: local knowledge base
- Notebooks: Udaplay_01_starter_project.ipynb, Udaplay_02_starter_project.ipynb

# 4) Multi Agent System for Paper Company (Munder Difflin)
- A multi-agent business operations simulator for a paper company: inventory → quoting → ordering → reporting.

## Core Idea (Architecture)
- Uses `smolagents ToolCallingAgent` + tool functions and a SQLite backend to process “customer quote requests”.
- The system parses messy natural-language orders into a validated schema, checks inventory, decides restocking, generates quotes, logs transactions, and produces a financial report.

## Key Components 
- `project_starter.py` (main): huge single-file “app” containing:
    - SQLite init + tables (inventory, transactions, quote_requests, quotes)
    - Structured extraction with client.beta.chat.completions.parse(...) into Pydantic models (ParsedOrder, LineItem)
    - Tool functions like:
        - extract order items/qty/dates
        - compare current stock vs customer needs vs min stock levels
        - compute pricing / discounts / quote logic
        - log transactions + final report

## Artifacts 
- `final_report.json, results.json, test_results.csv`

# 5) Price Scout Project (LLM Inference Client, MCP Server)
- An MCP server + client chatbot that scrapes pricing pages for inference providers and helps compare token costs.

## Core Architecture 
- MCP server (FastMCP) exposes tools:
- `scrape_websites(...)`: uses Firecrawl to scrape markdown/html, caches to disk, stores metadata JSON
- `extract_scraped_info(identifier)`: reads cached files and returns scraped content for a given provider/url/domain

Intended to feed into a local DB / comparison flow (see `test.db`)

## Key Comppnents 
- `starter_server.py`: working MCP server tool implementations (scrape + extract).
- `starter_client.py`: chatbot client that calls MCP tools (and likely LLM to summarize/compare).
- `scraped_content/ + scraped_metadata.json`: cached provider content.
- `developer_guide.md, instructions.md`: internal developer notes.