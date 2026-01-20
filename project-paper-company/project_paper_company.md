# Project Summary — Munder Difflin Multi-Agent System (Paper Company)

## What this project is
A Python-based **multi-agent workflow** that simulates how a fictional paper company handles **customer quote requests**, checks **inventory**, decides on **restocking**, applies **discount logic** using historical quotes, records **transactions** in SQLite, and produces **customer-facing order confirmations** plus **financial reports**.

Frameworks/tools used:
- **smolagents** `ToolCallingAgent` for agent orchestration + tool calling
- **OpenAI-compatible API** via Vocareum proxy (`https://openai.vocareum.com/v1`)
- **SQLite + SQLAlchemy + pandas** for data storage and querying
- **pydantic** for schema-constrained parsing of customer requests
- **TF-IDF + cosine similarity** (scikit-learn) for vector-ish retrieval over quote history

---

## Repository layout (top-level)
```
project-paper-company/
├─ project_starter.py
├─ project_starter_backup_20251222.py
├─ README.md
├─ report.MD
├─ requirements.txt
├─ .env_example
├─ munder_difflin.db
├─ quotes.csv
├─ quote_requests.csv
├─ quote_requests_sample.csv
├─ quote_requests_sample - backup.csv
├─ results.json
├─ test_results.csv
├─ final_report.json
└─ workflow.svg
```

## Key code: `project_starter.py`
### Purpose
This is the **main application**. It:
1) Defines a product catalog (`paper_supplies`)
2) Builds/initializes the SQLite DB (`munder_difflin.db`)
3) Implements inventory & quote search helpers
4) Defines **tool functions** (decorated with `@tool`) that agents can call
5) Defines **4 agents** (Inventory, Quote, Sales, Orchestrator)
6) Runs test scenarios from `quote_requests_sample.csv` and writes output artifacts

### Runtime dependencies & configuration
- Loads `.env` (expects `UDACITY_OPENAI_API_KEY`)
- Uses model ID `gpt-4o-mini` via `OpenAIServerModel` and OpenAI client
- Uses SQLite file: `sqlite:///munder_difflin.db`

### Data model / database tables created by `init_database()`
**Tables**
- `transactions`
  - columns: `id`, `item_name`, `transaction_type` (`stock_orders` or `sales`), `units`, `price`, `transaction_date`
- `quote_requests` (loaded from `quote_requests.csv`)
  - includes an `id` assigned sequentially
- `quotes` (loaded from `quotes.csv`)
  - `request_id`, `total_amount`, `quote_explanation`, `order_date`, plus optional metadata unpacked:
    - `job_type`, `order_size`, `event_type` (from `request_metadata` if present)
- `inventory`
  - seeded from `paper_supplies` with generated fields:
    - `current_stock`, `min_stock_level`, plus `item_name`, `category`, `unit_price`

**Seeding behavior**
- Sets an `initial_date = 2025-01-01T00:00:00` for baseline rows.
- Inserts a “starting cash” row using a dummy `sales` transaction of `50000.0`.
- Inserts one `stock_orders` transaction per inventory item to represent initial stock purchase.

### Core helper functions (non-agent)
- `generate_sample_inventory(paper_supplies, coverage=..., seed=...)`
  - randomly generates stock + minimum stock levels per item (seeded for reproducibility)
- `create_transaction(item_name, transaction_type, quantity, price, date)`
  - writes a stock order or sale to `transactions` and returns last insert row id
- Inventory queries:
  - `get_all_inventory(as_of_date)` → dict of positive stock per item
  - `get_stock_level(item_name, as_of_date)` → DataFrame for one item
  - `get_stock_level_multiple_items(item_names, as_of_date)` → DataFrame for many items (uses SQLAlchemy expanding bind params)
  - `get_supplier_delivery_date(input_date_str, quantity)` → lead-time logic (0/1/4/7 days by quantity tiers)
- Finance:
  - `get_cash_balance(as_of_date)` = sales revenue − stock purchases
  - `generate_financial_report(as_of_date)` → cash, inventory valuation, total assets, top sellers
- Quote history search:
  - `search_quote_history(search_terms, limit)` → SQL `LIKE` join of `quotes` + `quote_requests`
  - `search_quote_history_tfidf(search_terms, limit)` → TF-IDF over joined rows + cosine similarity scoring

### Pydantic schemas for request parsing
- `LineItem`: `{ item_name: Literal[<all known catalog names>], qty: int (>0) }`
- `ParsedOrder`: `{ requested_by: Optional[str], requested_on: Optional[str], line_items: List[LineItem] }`

### LLM-assisted parsing & pricing helpers
- `extract_items_from_request(request_text)`:
  - Uses **structured output** (`client.beta.chat.completions.parse`) constrained to `ParsedOrder`
  - Includes rule: if “ream” appears, 1 ream = 500 sheets
- `get_unit_price(item_names)`:
  - Pulls unit prices from the in-code `paper_supplies` list

---

## Multi-agent design in `project_starter.py`
### Agents
1) **InventoryAgent** (does not mutate DB)
   - Tools:
     - stock lookup (single/multiple/all)
     - min stock extraction
     - stock-vs-requirements comparison (LLM)
     - supplier delivery estimation
     - deadline feasibility check

2) **QuotingAgent** (does not mutate DB)
   - Tools:
     - history search via SQL
     - history search via TF-IDF similarity
     - infer discount thresholds from historical quotes (LLM)
     - apply discounts with guardrails (LLM; “never increase price”)

3) **SalesAgent** (mutates DB)
   - Tools:
     - confirm restock (`stock_orders` transaction)
     - confirm sale (`sales` transaction)
     - compute cash balance
     - compute financial report
     - raise alert (prints alert text; simulates escalation)

4) **Orchestrator** (the entrypoint agent)
   - Exposes tool: `handle_customer_request(customer_request, date_of_request=None, due_date=None)`
   - Workflow:
     1) Parse customer request into structured `ParsedOrder`
     2) Ask InventoryAgent for an **inventory feasibility report**
     3) Ask QuotingAgent for a **pricing/discount report**
     4) Ask SalesAgent to:
        - perform required restock + sales transactions
        - check financial health (cash/inventory thresholds)
        - draft the customer response message (with totals + transaction IDs)

### Notable orchestration constraints (prompted rules)
- InventoryAgent is forced to return **strict JSON array** with a fixed schema.
- SalesAgent runs a **two-phase process**:
  - Phase A: tool calls to mutate DB + run financial checks
  - Phase B: write final customer message (includes line items, discounts, totals, deadlines, ship date, transaction IDs)

---

## Backup code: `project_starter_backup_20251222.py`
A snapshot/older version of the main script (useful for diffing changes or recovering earlier logic). The project’s “active” implementation appears to be `project_starter.py`.

---

## Documentation
### `README.md`
- Explains the project goal (multi-agent automation for inventory, quoting, fulfillment)
- Setup:
  - `pip install -r requirements.txt`
  - install `smolagents` separately
  - create `.env` with `UDACITY_OPENAI_API_KEY`
- Primary run path: execute `run_test_scenarios()` to process sample requests and write outputs

### `report.MD`
A human-written report describing:
- The 4-agent architecture and responsibilities
- Added enhancements:
  - multi-item stock lookup
  - TF-IDF quote search
- Example “success” request outputs and what the customer responses look like

### `workflow.svg`
A diagram artifact representing the agent workflow/data flow (referenced in the report).

---

## Data & artifacts
### Input data
- `quotes.csv`
  - historical quotes; includes `quote_explanation`, `total_amount`, possibly `request_metadata`
- `quote_requests.csv`
  - source “customer inquiries” loaded into DB at init
- `quote_requests_sample.csv`
  - test cases used by `run_test_scenarios()`
- `quote_requests_sample - backup.csv`
  - backup copy of the sample requests

### Database
- `munder_difflin.db`
  - SQLite file used by the script (also included as an artifact in the zip)

### Output artifacts
- `results.json`
  - list of processed requests with customer replies + financial snapshots (from the test run)
- `test_results.csv`
  - tabular run log of request_id/date, cash/inventory, response, and embedded financial report object
- `final_report.json`
  - final roll-up financial report at end of the run

---

## How to run (as designed by the repo)
1) Create `.env` from `.env_example` and set:
   - `UDACITY_OPENAI_API_KEY=...`
   - `OPENAI_BASE_URL=https://openai.vocareum.com/v1` (if your environment uses it)
2) Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install smolagents`
3) Run:
   - `python project_starter.py`
4) Outputs appear in:
   - `results.json`, `test_results.csv`, `final_report.json`
   - updated `munder_difflin.db`

---

## Quick “mental model” of the system
Customer request text
→ (Orchestrator parses to structured items/dates)
→ InventoryAgent checks stock + reordering feasibility
→ QuotingAgent finds similar past quotes + computes safe discounts
→ SalesAgent writes transactions + checks health + drafts customer reply
→ Outputs: customer response + updated DB + financial report artifacts