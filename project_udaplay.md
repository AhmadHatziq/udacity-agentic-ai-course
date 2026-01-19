# UdaPlay (project-udaplay) — Codebase Summary

## 1) What this project is
UdaPlay is an “AI game research agent” starter project built around:
- **Offline RAG** using a **ChromaDB vector store** (persistent option included)
- A lightweight **agent loop** that can call tools (OpenAI tool-calling style)
- A simple **state machine framework** to orchestrate multi-step workflows
- Optional **memory** (short-term session history + long-term vector memory)

Repository entry points are mainly the **Jupyter notebooks**, with reusable logic inside `lib/`.

---

## 2) Top-level files

### `README.md`
Explains the two-part structure:
1. **Part 1: Offline RAG** — build + query a ChromaDB collection of game documents (name, platform, genre, publisher, description, year).
2. **Part 2: Agent** — an agent that uses internal knowledge (RAG), can do web search when needed, keeps state, and returns structured outputs.

---

## 3) Notebooks (how you’re expected to run the project)

### `Udaplay_01_starter_project.ipynb`
**Goal:** Build the offline RAG index.
Typical flow:
- Environment setup (expects `.env` style variables, e.g. API keys)
- Create/init a **Vector DB** via ChromaDB
- Create a **collection** with an embedding function (OpenAI embeddings via Chroma helper)
- Load and transform game data into documents
- Add documents into the collection
- Query the collection for similarity search results

This notebook is where the vector store is created/populated.

---

### `Udaplay_02_starter_project.ipynb`
**Goal:** Build the agent that uses tools + RAG.
Typical flow:
- Setup + imports
- Define or register **tools** (e.g., “retrieve game” tool)
- Initialize a **VectorStore** (from the same persisted Chroma directory/collection)
- Build an **Agent** that:
  - formats messages,
  - calls the LLM,
  - executes tools when the LLM requests them,
  - loops until termination
- Run test prompts and inspect output / runs

This notebook is where orchestration (agent + tooling + retrieval) comes together.

---

### `query-chroma-db.ipynb`
**Goal:** Quick experiments querying ChromaDB directly.
Includes examples of:
- Loading ChromaDB “the normal way”
- Loading via `VectorStoreManager`
- Using the predefined `RAG` class with a stored collection

Useful for debugging collection persistence, query behavior, and retrieval quality.

---

## 4) Library modules (`lib/`)

### `lib/__init__.py`
Empty/minimal initializer for the package.

---

### `lib/state_machine.py`
**Core orchestration framework** used by both `Agent` and `RAG`.

Key components:
- `Step`: wraps a piece of logic and updates TypedDict state safely
  - Supports step functions that accept `(state)` or `(state, resource)`
- `EntryPoint`, `Termination`: special steps to start/end a workflow
- `Transition`: connects steps; can be conditional
- `Snapshot`: captures state after each executed step (timestamped)
- `Run`: stores all snapshots and metadata for a full execution trace
- `StateMachine`: adds steps, connects transitions, and executes sequentially
  - Notes: parallel branching is explicitly **not implemented**.

Why it matters:
- Provides a deterministic, inspectable execution trace (snapshots)
- Encourages building agent/RAG flows as explicit step graphs

---

### `lib/messages.py`
**Typed message objects** for chat + tools.

Defines:
- `BaseMessage` + concrete types:
  - `SystemMessage`, `UserMessage`, `AIMessage`, `ToolMessage`
- `TokenUsage`: tracks prompt/completion/total tokens for a call

Why it matters:
- Standardizes message formatting for the `LLM` wrapper
- `AIMessage` supports `tool_calls` (OpenAI tool call objects)

---

### `lib/tooling.py`
**Tool wrapper + decorator** that converts Python functions into OpenAI-style tool schemas.

Defines:
- `Tool`: wraps a function and produces `.dict()` compatible with OpenAI tool schema
- `ToolCall`: type alias of OpenAI’s chat completion tool-call type
- `@tool` decorator: convenience wrapper to register tools easily

Notable behavior:
- Infers JSON schema types from Python type hints:
  - `Literal` → enum
  - `Optional[T]` → schema for `T`
  - `list[T]`, `dict[str, T]` → array/object schemas
  - primitives → string/integer/number/boolean

---

### `lib/llm.py`
**OpenAI client wrapper** used everywhere to call the LLM.

Defines:
- `LLM.invoke(...)` which accepts:
  - a string, a single `BaseMessage`, or a list of `BaseMessage`
  - optional `response_format` for structured outputs via `client.beta.chat.completions.parse`
- Automatically includes tools + `tool_choice="auto"` if any tools are registered
- Returns an `AIMessage` with optional `tool_calls` and `token_usage`

---

### `lib/agents.py`
**Agent implementation** using the state machine + tool execution loop.

Key pieces:
- `AgentState` (TypedDict): fields like `user_query`, `instructions`, `messages`,
  `current_tool_calls`, `total_tokens`, `session_id`
- `Agent`:
  - Maintains `ShortTermMemory` for session runs
  - Builds a state machine with steps:
    1. `message_prep`: ensure system message exists, append user message
    2. `llm_processor`: call `LLM.invoke`, capture tool calls + token usage
    3. `tool_executor`: execute requested tools and append `ToolMessage` results
    4. loop back to LLM until no more tool calls, then terminate
  - `invoke(query, session_id)`: runs the workflow and stores the `Run` in memory
  - Helpers: `get_session_runs`, `reset_session`

Important implementation detail:
- Tool execution expects `call.function.arguments` JSON, loads it, then calls the matching `Tool`.

---

### `lib/rag.py`
**RAG pipeline** implemented as a state machine.

Key pieces:
- `RAGState`: `question`, `documents`, `distances`, `messages`, `answer`
- Steps:
  1. `retrieve`: vector similarity query via `VectorStore.query`
  2. `augment`: construct a prompt with retrieved context + question
  3. `generate`: call LLM and output the final `answer`
- Uses `Resource(vars={ "llm": ..., "vector_store": ... })` to pass dependencies into steps

---

### `lib/vector_db.py`
**ChromaDB abstraction layer**.

Defines:
- `VectorStore`: wrapper around a Chroma collection
  - `add(Document|Corpus|List[Document])`: normalizes into Chroma batch add
  - `query(query_texts, n_results=3, where=None, where_document=None)`
  - `get(ids=None, where=None, limit=None)`
- `VectorStoreManager`: handles:
  - embedding function creation (OpenAI embeddings via Chrom
