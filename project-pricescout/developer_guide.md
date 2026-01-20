# developers.md

## Project Overview

This project implements an **MCP-based agentic system** composed of:

### 1. MCP Client / Chat Orchestrator
- Talks to Claude (Anthropic)
- Discovers and calls MCP tools
- Handles retries, logging, and tool execution
- Extracts structured data and stores it in SQLite via MCP

### 2. MCP Server (Firecrawl Scraper)
- Exposes web-scraping tools via MCP
- Scrapes pricing/inference pages using Firecrawl
- Caches scraped content on disk
- Provides structured access to scraped data

The two components communicate **only via MCP**, not direct imports.


## High-level Architecture

```
User (CLI)
│
▼
ChatSession (Anthropic + MCP Client)
│
├─▶ MCP Server: SQLite (read/write_query)
│
└─▶ MCP Server: Firecrawl Scraper
├─ scrape_websites
└─ extract_scraped_info
```

# PART 1 — MCP CLIENT (Chat / Orchestrator)

## Key Responsibilities

- Initialize MCP servers
- Discover tools and map them to servers
- Run Claude tool-calling loops
- Execute MCP tools with retries and timeout
- Extract structured pricing data
- Store extracted data into SQLite

---

## Helper Functions

### `dump_response(resp) -> None`

Debug utility to log Anthropic response objects safely:
- Tries `model_dump()`
- Falls back to `dict()`
- Else uses `repr()`

---

### `anthropic_messages_to_string(messages: list[dict]) -> str`

Converts full Anthropic message history into a readable transcript.

Supports:
- Assistant text blocks
- `tool_use` blocks
- `tool_result` blocks

Used for downstream parsing or debugging.

---

## Typed Structures

### `ToolDefinition (TypedDict)`

Represents a tool in Claude-compatible format:
- `name: str`
- `description: str`
- `input_schema: dict`

---

## Class: `Configuration`

### Responsibility
Loads environment variables and server configuration.

### Methods
- `load_env()` — loads `.env`
- `load_config(path)` — loads `server_config.json`
  - Requires top-level key `mcpServers`

### Property
- `anthropic_api_key`
  - Raises if missing

---

## Class: `Server`

### Responsibility
Wraps a **single MCP server connection**.

### Key Attributes
- `name`
- `config`
- `session: ClientSession`
- `exit_stack: AsyncExitStack`

### Methods

#### `initialize()`
- Starts MCP server over stdio
- Creates `ClientSession`
- Calls `session.initialize()`

#### `list_tools() -> List[ToolDefinition]`
- Calls `session.list_tools()`
- Converts MCP tool schema → Claude tool schema

#### `execute_tool(...)`
Executes an MCP tool with:
- Retry loop
- Logging per attempt
- **Mandatory 60-second read timeout**

```python
await self.session.call_tool(
    tool_name,
    arguments,
    read_timeout_seconds=timedelta(seconds=60),
)
```

`cleanup()` 
- Closes all stdio and session resources safely.

## Class: `DataExtractor`

### Responsibility
Extracts structured pricing plans from Claude output and stores them in SQLite.

### Methods

#### `setup_data_tables()`
Creates the `pricing_plans` table via SQLite MCP tool.

#### `_get_structured_extraction(prompt) -> str`
- Calls Claude with a strict JSON-only prompt  
- Returns text output only

#### `extract_and_store_data(...)`
- Prompts Claude to emit valid JSON  
- Parses extracted plans  
- Inserts rows into SQLite via MCP  

⚠️ **Note:** Current SQL inserts use f-strings and are not parameterized.

---

## Class: `ChatSession`

### Responsibility
Main orchestration layer.

### Key Attributes
- `servers`
- `available_tools`
- `tool_to_server` mapping
- `sqlite_server`
- `data_extractor`

### Methods

#### `process_query(query) -> str`
Core execution loop:
1. Sends user query to Claude  
2. Detects `tool_use` blocks  
3. Executes tools via MCP  
4. Sends `tool_result` blocks back to Claude  
5. Repeats until no more tool calls  
6. Extracts and stores structured data  

#### `chat_loop()`
CLI loop:
- `quit`
- `show data`
- Otherwise → `process_query()`

#### `show_stored_data()`
Reads recent pricing rows from SQLite MCP server and prints them.

#### `start()`
- Initializes all MCP servers  
- Discovers tools  
- Enables data extraction  
- Starts chat loop  

### Entry Point
```python
if __name__ == "__main__":
    asyncio.run(main())
```

# PART 2 — MCP SERVER (Firecrawl Scraper)

## Purpose

This MCP server:
- Scrapes pricing and inference pages from AI providers
- Caches scraped content locally
- Exposes structured access via MCP tools

---

## Global Setup

### Environment
- Uses `.env`
- Requires `FIRECRAWL_API_KEY`

### Storage
All scraped data is stored under:

```
scraped_content/
├─ scraped_metadata.json
├─ provider_markdown.txt
└─ provider_html.txt
```


