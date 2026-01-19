# AgentsVille Trip Planner (LLM Agent + Tools + Evals)

This project is a Jupyter-notebook-based “AI travel agent” that generates a day-by-day itinerary for a fictional city called **AgentsVille**. It demonstrates agent prompting patterns (role-based prompting, Chain-of-Thought style planning, ReAct loops) and uses **mocked APIs** (weather + activities) plus **evaluation functions** to iteratively improve the itinerary.

---

## Project Structure

- `project_starter.ipynb`
  - The main notebook where you:
    1) define the trip details,
    2) load mocked weather + activities,
    3) generate an initial itinerary using an LLM,
    4) run eval checks,
    5) revise the plan via a ReAct loop using “tools”,
    6) generate a fun narrative summary (optional).

- `project_lib.py`
  - Utility code used by the notebook:
    - A lightweight `ChatAgent` wrapper around the OpenAI chat completion API
    - Pretty-print helpers (`print_in_box`)
    - Mocked “external API” functions for weather and activities
    - A text-to-speech narration helper (optional)

---

## What the Notebook Does (`project_starter.ipynb`)

### 1) Environment / Client Setup
The notebook initializes an OpenAI client (optionally using a Vocareum base URL) and selects a model via an `OpenAIModel` enum.

### 2) Define Trip Requirements (Input)
You define a trip in a structured JSON-like format and validate it using **Pydantic** models such as:
- `Traveler` (name + interests, etc.)
- `VacationInfo` (travelers, destination, arrival/departure, budget)

This step ensures the agent always receives well-formed trip constraints.

### 3) Pull Context: Weather + Activities (Mock APIs)
The notebook calls mocked functions (from `project_lib.py`) to retrieve:
- weather forecasts per day (`call_weather_api_mocked`)
- available activities per day (`call_activities_api_mocked`)

These are loaded in bulk so the LLM can plan using *real available events* rather than hallucinating.

### 4) Generate Initial Itinerary (ItineraryAgent)
`ItineraryAgent` extends the shared `ChatAgent` and uses a dedicated system prompt to produce a structured itinerary output.

Key behavior:
- Sends the `VacationInfo` JSON to the LLM
- Receives a JSON response and validates it into a `TravelPlan` Pydantic model
- Strips ```json fences if the model returns fenced JSON

The output model typically includes:
- trip metadata (destination, dates)
- per-day plan (`ItineraryDay`) with:
  - day weather
  - a list of recommended activities (`ActivityRecommendation`)
- computed totals (e.g., total cost)

### 5) Evaluate the Itinerary (Evals)
The notebook defines several evaluation functions that raise errors if the plan violates constraints, for example:
- dates match arrival/departure
- itinerary activities match the “actual” available events (no hallucinated events)
- interests are satisfied
- total cost is within budget
- total cost arithmetic is correct (sum of activity prices)
- weather compatibility (LLM-assisted check for outdoor/indoor mismatch)

These evals are then aggregated into an `EvaluationResults` summary.

### 6) Define Tools (for ReAct)
To support iterative improvement, the notebook defines “tools” as Python functions the agent can call:
- `calculator_tool`: reliable arithmetic
- `get_activities_by_date_tool`: fetch activities for a specific date
- `run_evals_tool`: run all evals and return failures
- `final_answer_tool`: signal that the agent is done and provide the final plan

The tools are described to the LLM so it can choose actions during the ReAct loop.

### 7) Revise the Itinerary (ItineraryRevisionAgent + ReAct Loop)
`ItineraryRevisionAgent` also extends `ChatAgent` and uses a ReAct-style system prompt:

- The LLM emits:
  - `THOUGHT: ...`
  - `ACTION: {"tool_name": "...", "arguments": {...}}`

- Python parses the JSON action, runs the matching tool function, and appends:
  - `OBSERVATION: ...`

This repeats up to `max_steps` times until the agent calls `final_answer_tool`.

There is also traveler feedback integrated as an additional evaluation constraint (example: “at least two activities per day”).

### 8) Fun Add-on: Trip Narration (Optional)
The project can generate a narrative summary of the final trip and (optionally) a short audio narration using OpenAI TTS, via `narrate_my_trip` in `project_lib.py`.

---

## What the Library File Does (`project_lib.py`)

### `Interest` enum
A fixed list of interest tags (art, hiking, technology, etc.) used by travelers and activity matching.

### `ChatAgent`
A small wrapper that:
- stores chat history (`messages`)
- prints prompts/responses in a boxed format for readability
- calls OpenAI chat completions via `do_chat_completion`

### Mocked Data + Mocked APIs
- `ACTIVITY_CALENDAR`: a predefined list of events in AgentsVille (IDs, times, descriptions, prices, related interests)
- `WEATHER_FORECAST`: a predefined forecast for a specific date range
- `call_activities_api_mocked(date, city, activity_ids)`: filters events by date/city/IDs
- `call_activity_by_id_api_mocked(activity_id)`: fetch one event
- `call_weather_api_mocked(date, city)`: returns forecast for that date

These mocks allow deterministic testing without real external services.

### `narrate_my_trip(...)`
Uses an LLM to write a narrative trip summary, and optionally generates audio via OpenAI’s speech API.

---

## How to Run

1. Open `project_starter.ipynb` in Jupyter / VSCode.
2. Ensure you have an OpenAI-compatible API key available (environment variable or the notebook’s client config).
3. Run cells top-to-bottom.
4. Fill in any `TODO` sections in the notebook prompts if they are part of the assignment.
5. Inspect:
   - the initial itinerary output,
   - eval failures,
   - the revised itinerary after the ReAct tool loop.

---

## Expected Output

- A validated `TravelPlan` object with:
  - correct date coverage
  - activities that exist in the mocked calendar
  - costs computed correctly and within budget
  - activities appropriate for the day’s weather
  - traveler feedback satisfied (e.g., >= 2 activities/day)

---

## Common “Gotchas” (Worth Knowing)

- The itinerary must use ONLY events in the mocked activities dataset (or evals will fail).
- LLM math is unreliable, so total cost should be checked/fixed via the calculator tool.
- Weather compatibility checks may require careful prompt wording since it’s an LLM-based eval.
- Output must be valid JSON that matches the Pydantic schema (or parsing/validation fails).

---

## Ideas for Extension

- Replace mocks with real APIs (weather + events) and add caching.
- Add ranking/scoring for activities (match interests, minimize travel time, maximize variety).
- Add constraints like “no more than X expensive activities per day”.
- Add “explainability”: why each activity was chosen given interests + weather + budget.
