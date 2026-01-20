import asyncio
import json
import logging
import os
import shutil
import traceback
import re
import ast

from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def dump_response(resp) -> None:
    """
    Helper function to dump response objects. 
    """
    try:
        if hasattr(resp, "model_dump"):
            logger.info("model_dump: %s", resp.model_dump())
        elif hasattr(resp, "dict"):
            logger.info("dict: %s", resp.dict())
        else:
            logger.info("repr: %r", resp)
    except Exception as e:
        logger.warning("Failed dumping response: %s", e)

def anthropic_messages_to_string(messages: list[dict]) -> str:
    """
    Helper function to render Anthropic messages (text/tool_use/tool_result) into a single transcript string
    suitable for downstream parsing.
    """
    lines: list[str] = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content")

        # content can be a string or a list of blocks
        if isinstance(content, str):
            lines.append(f"[{role.upper()}]\n{content}")
            continue

        if isinstance(content, list):
            # Each item is either a dict (your tool_result payload)
            # or an Anthropic content block object (text/tool_use)
            for block in content:
                # tool_result payload you created is a dict
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    lines.append(
                        "[TOOL_RESULT]\n"
                        f"tool_use_id={block.get('tool_use_id')}\n"
                        f"{block.get('content')}"
                    )
                    continue

                # Anthropic SDK block objects
                btype = getattr(block, "type", None)
                if btype == "text":
                    lines.append(f"[{role.upper()}]\n{getattr(block, 'text', '')}")
                elif btype == "tool_use":
                    lines.append(
                        "[TOOL_USE]\n"
                        f"name={getattr(block, 'name', '')}\n"
                        f"id={getattr(block, 'id', '')}\n"
                        f"input={json.dumps(getattr(block, 'input', {}), ensure_ascii=False)}"
                    )
                else:
                    # fallback for other block types
                    lines.append(f"[{role.upper()}:{btype}]\n{repr(block)}")

    return "\n\n".join(lines).strip()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        try: 
            with open(file_path, "r") as f:
                config = json.load(f)

                if "mcpServers" not in config: 
                    raise ValueError(f"JSON is invalid as missing key 'mcpServers'")
                return config 

        except json.JSONDecodeError as e:
            print(f"Invalid JSON syntax in config file: {e}")
            raise

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        # complete params
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        # complete
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_result = await self.session.list_tools()

        mcp_tools = getattr(tools_result, "tools", None)
        if mcp_tools is None:
            # Fallback in case the library returns a dict 
            if isinstance(tools_result, dict) and "tools" in tools_result:
                mcp_tools = tools_result["tools"]
            else:
                raise RuntimeError(
                    f"Unexpected list_tools() response type from server {self.name}: {type(tools_result)}"
                )

        tools: List[ToolDefinition] = []
        for t in mcp_tools:
            # MCP tool fields: name, description, inputSchema
            name = getattr(t, "name", None) if not isinstance(t, dict) else t.get("name")
            description = getattr(t, "description", "") if not isinstance(t, dict) else t.get("description", "")
            input_schema = getattr(t, "inputSchema", None) if not isinstance(t, dict) else t.get("inputSchema")

            if not name:
                continue
            if input_schema is None:
                # Anthropic requires an input_schema; default to empty object schema.
                input_schema = {"type": "object", "properties": {}}

            tools.append(
                {
                    "name": name,
                    "description": description or "",
                    "input_schema": input_schema,
                }
            )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 60.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        # complete
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        read_timeout_seconds = timedelta(seconds=60)

        attempt = 0
        while attempt < retries:
            try:
                # logging.info(f"Executing Tool: {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments, read_timeout_seconds=read_timeout_seconds)
                # logging.info(f"Returning tool {tool_name} result: {result}")
                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""
    
    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client
        
    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:
            
            await self.sqlite_server.execute_tool("write_query", {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            })
            
            logging.info("✓ Data extraction tables initialized")
            
        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024,
                model='claude-sonnet-4-20250514',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            text_content = ""
            if response.content: 
                for content in response.content:
                    if content.type == 'text':
                        text_content += content.text
            if not response.content: 
                logging.error(
                    "Structured extraction: empty/None response.content. "
                    f"response_type={type(response)} "
                    f"response_repr={repr(response)[:500]}"
                )
            
            return text_content.strip()
            
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")

            return '{"error": "extraction failed"}'
    
    async def extract_and_store_data(self, user_query: str, llm_response: str, 
                                   source_url: str = None) -> None:
        """Extract structured data from LLM response and store it."""
        try:            
            extraction_prompt = f"""
            You are a strict JSON serializer.

            Task:
            Extract pricing plans from the input text and output ONLY a single valid JSON object.
            No markdown. No explanations. No code fences. No comments.

            Rules (MUST follow):
            - Replace all single and double quotation marks with backticks (`)
            - Response must follow the JSON schema (given at the end)
            - Output must be valid JSON (RFC 8259).
            - Use double quotes for ALL strings.
            - Do NOT include trailing commas.
            - Numbers must be JSON numbers (no quotes). If unknown, use null (not 0).
            - currency must be a 3-letter code like "USD" if known; otherwise null.
            - billing_period must be one of: "monthly", "yearly", "one-time", or null.
            - features must be an array of strings (can be empty).
            - limitations must be a string (can be empty) or null.
            - query must echo the user query string exactly.

            Input text is between <TEXT> and </TEXT>.

            <TEXT>
            {llm_response}
            </TEXT>
            
            Return JSON with this exact schema:
            {{
            "company_name": string,
            "plans": [
                {{
                "plan_name": string,
                "input_tokens": number|null,
                "output_tokens": number|null,
                "currency": string|null,
                "billing_period": "monthly"|"yearly"|"one-time"|null,
                "features": string[],
                "limitations": string|null,
                "query": string
                }}
            ]
            }}
            """
            
            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")
            pricing_data = json.loads(extraction_response)
            
            for plan in pricing_data.get("plans", []):
                # complete
                # Insert plan data into DB 
                await self.sqlite_server.execute_tool("write_query", {
                    "query": f"""
                    INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
                    VALUES (
                        '{pricing_data.get("company_name", "Unknown")}',
                        '{plan.get("plan_name", "Unknown Plan")}',
                        '{plan.get("input_tokens", 0)}',
                        '{plan.get("output_tokens", 0)}',
                        '{plan.get("currency", "USD")}',
                        '{plan.get("billing_period", "unknown")}',
                        '{json.dumps(plan.get("features", []))}',
                        '{plan.get("limitations", "")}',
                        '{user_query}')
                    """
                })

            # logger.info(f"Input string: {llm_response}")
            # logger.info(f"Extraction Response: {extraction_response}")
            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")
            
        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(
            # base_url="https://claude.vocareum.com", 
            api_key=api_key)
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")



    async def process_query(self, query: str) -> List[Dict]:
        """
        Process a user query and extract/store relevant data.
        Args:
            query: Query input to chatbot eg "Compare cloudrift ai and deepinfra's costs for deepseek v3"

        Returns: 
            llm_respose 
        """
        # If query is about comparison, tell LLM to use the tool to get data from local DB 
        if "compare" in query.lower(): 
            query = (
                "Firstly, before comparing, select all data from `pricing_plans` table "
                "with: `SELECT * FROM pricing_plans`. "
                "This will give relevant data for comparison. "
                + query
            )
        
        messages = [{"role": "user", "content": query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-sonnet-4-20250514', 
            tools=self.available_tools,
            messages=messages
        )

        # dump_response(response)
        full_response = ""

        while True:
            # Check for errors 
            if getattr(response, "type", None) == "error":
                err = getattr(response, "error", None)
                raise RuntimeError(f"Anthropic error response: {err}")
            content = getattr(response, "content", None)
            if not content:  
                if response.stop_reason == "end_turn":
                    break
                else: 
                    raise RuntimeError(f"Anthropic returned empty content. Response: {response!r}")                

            assistant_blocks = [] # assistant_content 
            tool_use_blocks = [] # There can be multiple tool use blocks returned 

            # print(f"Response content: {response}")

            for block in content:
                assistant_blocks.append(block)
                if block.type == "text":
                    full_response += block.text
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Always append assistant content so the conversation state is correct
            messages.append({"role": "assistant", "content": assistant_blocks})

            # If no tool call, break 
            if len(tool_use_blocks) == 0:
                break

            # Execute ALL requested tools and respond with tool_result blocks for EACH tool_use_id
            tool_results_payload = []

            for tool_use_block in tool_use_blocks: 

                # Extract tool details
                tool_use_id = tool_use_block.id
                tool_name = tool_use_block.name
                tool_args = tool_use_block.input

                # Execute tools at the server
                server_name = self.tool_to_server.get(tool_name)
                if not server_name:
                    tool_result_text = f"Tool '{tool_name}' not found in tool_to_server map."
                else:
                    server = next((s for s in self.servers if s.name == server_name), None)
                    if not server:
                        tool_result_text = f"Server '{server_name}' not found for tool '{tool_name}'."
                    else:
                        try:
                            # Proceed with tool server execution 
                            tool_result_text = await server.execute_tool(tool_name, tool_args)
                            logger.info(f"Successfully executed tool {tool_name}")
                            # logger.info(f"Tool result text: {tool_result_text}")
                        except Exception as e:
                            tool_result_text = f"Error executing tool '{tool_name}': {e}"

                # Append tool results 
                tool_results_payload.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(tool_result_text),
                })
                

            # Send tool result back to Claude
            messages.append({"role": "user", "content": tool_results_payload}) 

            # Ask Claude again
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model="claude-sonnet-4-20250514",
                tools=self.available_tools,
                messages=messages,
            )
            # dump_response(response)

        # Extract/store pricing from the final response 
        parser_text = anthropic_messages_to_string(messages)
        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip(), None)

        print("LLM Full Response: \n", full_response.strip())
        return full_response.strip()

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data()
                    continue
                    
                llm_resonse = await self.process_query(query)
                print("\n")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                traceback.print_exc()
                print(f"\nError in chat_loop: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return
            
        try:
            # complete
            pricing = await self.sqlite_server.execute_tool("read_query", {
                "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
            })

            print("\nRecently Stored Data:")
            print("=" * 50)
            print("\nPricing Plans:")
            
            if not pricing.content:
                print("  (no results returned)")
                print("=" * 50)
                return

            content0 = pricing.content[0]
            text = getattr(content0, "text", None)
            if text is None:
                raise TypeError(f"Unexpected content type: {type(content0)}")

            # Parse rows: try JSON first, else Python literal repr
            try:
                rows = json.loads(text)
            except json.JSONDecodeError:
                rows = ast.literal_eval(text)  

            if not isinstance(rows, list):
                raise TypeError(f"Expected list of rows, got {type(rows)}: {rows!r}")

            # Print header line 
            print(
                "\n=== Pricing Plan Table Data ===\n"
                "Showing unique company-plan combinations\n"
            )

            # Print only unique company name - plan name 
            seen = set()
            for plan in rows:
                key = (plan.get("company_name"), plan.get("plan_name"))
                if key in seen:
                    continue
                else: 
                    seen.add(key)
                company = plan.get("company_name", "")
                plan_name = plan.get("plan_name", "")
                input_token = plan.get("input_tokens", "")
                output_token = plan.get("output_tokens", "")
                currency = plan.get("currency","")
                billing_period = plan.get("billing_period","")
                features = plan.get("features", "")
                limitations = plan.get("limitations", "")

                print(
                    f"• {company}: {plan_name} - Input Token: ${currency} {input_token}, Output Tokens: ${currency} {output_token}\n"
                    # f"Billing period: {billing_period}. Features: {features}\n"
                    # f"Limitations: {limitations}\n"
                )

            print("=" * 50)
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")
            
            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    
    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"
    
    server_config = config.load_config(config_file)
    
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())