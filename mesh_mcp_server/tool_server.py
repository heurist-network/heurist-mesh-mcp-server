"""
Mesh Tool MCP Server - Connects to mesh API endpoints and provides tools for tool execution.
Supports fine-grained tool selection per agent via configuration file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import anyio
import click
import colorlog
import mcp.types as types
import uvicorn
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from starlette.applications import Starlette
from starlette.routing import Mount, Route

# Load environment variables
load_dotenv()


# ===== Configuration =====
class Config:
    """Server configuration settings."""

    # API endpoints and authentication
    HEURIST_API_KEY = os.environ.get("HEURIST_API_KEY")
    HEURIST_API_ENDPOINT = os.getenv(
        "MESH_API_ENDPOINT", "https://sequencer-v2.heurist.xyz"
    )
    HEURIST_METADATA_ENDPOINT = os.getenv(
        "MESH_METADATA_ENDPOINT", "https://mesh.heurist.ai/mesh_agents_metadata.json"
    )

    # Configuration file path
    CONFIG_FILE = "config.json"

    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(log_color)s%(levelname)s%(reset)s:     %(message)s"
    LOGGER_NAME = "mesh-mcp-tools"

    @classmethod
    def setup_logger(cls):
        """Configure and return a logger with colored output."""
        logger = colorlog.getLogger(cls.LOGGER_NAME)
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(cls.LOG_FORMAT))
        logger.handlers = []
        logger.addHandler(handler)
        logger.setLevel(cls.LOG_LEVEL)
        return logger

    @classmethod
    def load_agent_config(
        cls, config_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Load agent and tool configuration from JSON file.

        Args:
            config_path: Optional path to config file. If not provided, uses default.

        Returns:
            Dictionary mapping agent IDs to list of enabled tool names

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        if config_path is None:
            # Look for config.json in the same directory as the script
            script_dir = Path(__file__).parent
            config_path = script_dir / cls.CONFIG_FILE
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(
                f"Config file not found at {config_path}, using empty configuration"
            )
            return {}

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                return config_data.get("agents", {})
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise


# Configure logger
logger = Config.setup_logger()


# ===== Custom Exceptions =====
class MeshApiError(Exception):
    """Raised when there's an error with the Mesh API."""

    pass


class ToolExecutionError(Exception):
    """Raised when there's an error executing a tool."""

    pass


# ===== API Client =====
async def call_mesh_api(
    path: str, method: str = "GET", json: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Helper function to call the mesh API endpoint.

    Args:
        path: API path to call
        method: HTTP method to use
        json: Optional JSON payload

    Returns:
        API response as dictionary

    Raises:
        MeshApiError: If there's an error calling the API
    """
    async with aiohttp.ClientSession() as session:
        url = f"{Config.HEURIST_API_ENDPOINT}/{path}"
        try:
            headers = {}
            if Config.HEURIST_API_KEY:
                headers["X-HEURIST-API-KEY"] = Config.HEURIST_API_KEY

            async with session.request(
                method, url, json=json, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise MeshApiError(f"Mesh API error: {error_text}")
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error calling mesh API: {e}")
            raise MeshApiError(f"Failed to connect to mesh API: {str(e)}") from e


# ===== Tool MCP Server =====
class MeshToolServer:
    """Encapsulates the MCP server for mesh agent tools."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the server.

        Args:
            config_path: Optional path to configuration file
        """
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        self.agent_tool_config: Dict[str, List[str]] = {}
        self.config_path = config_path
        self.server = None

    async def fetch_agent_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Fetch agent metadata from the API.

        Returns:
            Dictionary mapping agent IDs to their metadata

        Raises:
            MeshApiError: If there's an error fetching metadata
        """
        logger.info(f"Fetching agent metadata from {Config.HEURIST_METADATA_ENDPOINT}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(Config.HEURIST_METADATA_ENDPOINT) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to fetch agent metadata: HTTP {response.status}"
                        )
                        return {}
                    data = await response.json()
                    return data.get("agents", {})
        except Exception as e:
            logger.error(f"Error fetching agent metadata: {e}")
            raise MeshApiError(f"Failed to fetch agent metadata: {str(e)}") from e

    def is_tool_enabled(self, agent_id: str, tool_name: str) -> bool:
        """Check if a specific tool is enabled for an agent based on configuration.

        Args:
            agent_id: The agent ID
            tool_name: The tool name

        Returns:
            True if the tool is enabled, False otherwise
        """
        if not self.agent_tool_config:
            # No configuration means all tools are disabled
            return False

        if agent_id not in self.agent_tool_config:
            # Agent not in config means it's disabled
            return False

        agent_tools = self.agent_tool_config[agent_id]

        # Empty list means all tools are enabled for this agent
        if not agent_tools:
            return True

        # Check if this specific tool is in the enabled list
        return tool_name in agent_tools

    async def process_tool_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Process agent metadata and extract tool information based on configuration.

        Returns:
            Dictionary mapping tool IDs to tool information
        """
        # Load configuration
        self.agent_tool_config = Config.load_agent_config(self.config_path)

        if not self.agent_tool_config:
            logger.warning("No agents configured in config file")
            return {}

        agents_metadata = await self.fetch_agent_metadata()
        tool_registry = {}

        # Log configuration status
        logger.info(
            f"Processing tools for {len(self.agent_tool_config)} configured agents"
        )

        # Track statistics
        agents_processed = set()
        tools_enabled = 0
        tools_skipped = 0

        for agent_id, agent_data in agents_metadata.items():
            # Skip agents not in our configuration
            if agent_id not in self.agent_tool_config:
                continue

            agents_processed.add(agent_id)
            configured_tools = self.agent_tool_config[agent_id]

            # Log agent processing
            if configured_tools:
                logger.info(
                    f"Processing agent {agent_id} with {len(configured_tools)} specific tools"  # noqa: E501
                )
            else:
                logger.info(f"Processing agent {agent_id} with all tools enabled")

            # Process tools for this agent
            for tool in agent_data.get("tools", []):
                if tool.get("type") == "function":
                    function_data = tool.get("function", {})
                    tool_name = function_data.get("name")

                    if not tool_name:
                        continue

                    # Check if this tool is enabled based on configuration
                    if not self.is_tool_enabled(agent_id, tool_name):
                        logger.debug(
                            f"Skipping tool {tool_name} for agent {agent_id} (not in config)"  # noqa: E501
                        )
                        tools_skipped += 1
                        continue

                    # Create a unique tool ID
                    tool_id = f"{agent_id.lower()}_{tool_name}"

                    # Get parameters or create default schema
                    parameters = function_data.get("parameters", {})
                    if not parameters:
                        parameters = {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }

                    # Store tool info
                    tool_registry[tool_id] = {
                        "agent_id": agent_id,
                        "tool_name": tool_name,
                        "description": function_data.get("description", ""),
                        "parameters": parameters,
                    }
                    tools_enabled += 1
                    logger.debug(f"Enabled tool: {tool_id}")

        # Log summary
        logger.info("Configuration summary:")
        logger.info(
            f"  - Agents processed: {len(agents_processed)} / {len(self.agent_tool_config)} configured"  # noqa: E501
        )
        logger.info(f"  - Tools enabled: {tools_enabled}")
        logger.info(f"  - Tools skipped: {tools_skipped}")

        if agents_processed:
            logger.info(f"  - Active agents: {', '.join(sorted(agents_processed))}")

        return tool_registry

    async def execute_tool(
        self, agent_id: str, tool_name: str, tool_arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on a mesh agent.

        Args:
            agent_id: ID of the agent to execute the tool on
            tool_name: Name of the tool to execute
            tool_arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If there's an error executing the tool
        """
        request_data = {
            "agent_id": agent_id,
            "input": {"tool": tool_name, "tool_arguments": tool_arguments},
        }

        # Add API key if available
        if Config.HEURIST_API_KEY:
            request_data["api_key"] = Config.HEURIST_API_KEY

        try:
            result = await call_mesh_api(
                "mesh_request", method="POST", json=request_data
            )
            return result.get("data", result)  # Prefer the 'data' field if it exists
        except MeshApiError as e:
            # Re-raise API errors with clearer context
            raise ToolExecutionError(str(e)) from e
        except Exception as e:
            logger.error(f"Error calling {agent_id} tool {tool_name}: {e}")
            raise ToolExecutionError(
                f"Failed to call {agent_id} tool {tool_name}: {str(e)}"
            ) from e

    async def initialize(self) -> Server:
        """Initialize by loading tools from metadata.

        Returns:
            The configured MCP server instance

        Raises:
            ValueError: If no tools could be loaded from metadata
        """
        self.tool_registry = await self.process_tool_metadata()
        if not self.tool_registry:
            logger.warning(
                "No tools loaded. Check your config.json file and ensure agents/tools are properly configured."  # noqa: E501
            )
        self.server = self._create_server()
        return self.server

    def _create_server(self) -> Server:
        """Create and configure the MCP server with all tools.

        Returns:
            Configured MCP server instance
        """
        app = Server("mesh-agent-tools-mcp-server")

        @app.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List all available tools."""
            return [
                types.Tool(
                    name=tool_id,
                    description=tool_info["description"],
                    inputSchema=tool_info["parameters"],
                )
                for tool_id, tool_info in self.tool_registry.items()
            ]

        @app.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[types.TextContent]:
            """Call the specified tool with the given arguments."""
            try:
                if name not in self.tool_registry:
                    raise ValueError(f"Unknown tool: {name}")

                tool_info = self.tool_registry[name]
                result = await self.execute_tool(
                    agent_id=tool_info["agent_id"],
                    tool_name=tool_info["tool_name"],
                    tool_arguments=arguments,
                )

                # Convert result to TextContent
                return [types.TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                raise ValueError(f"Failed to call tool {name}: {str(e)}") from e

        return app

    async def run_stdio(self):
        """Run the server using stdio transport."""
        if not self.server:
            await self.initialize()

        logger.info("Starting stdio server")
        async with stdio_server() as streams:
            await self.server.run(
                streams[0], streams[1], self.server.create_initialization_options()
            )

    def run_sse(self, port: int, base_path: str = ""):
        """Run the server using SSE transport.

        Args:
            port: Port to listen on
            base_path: Optional base path for URL construction
        """
        if not self.server:
            anyio.run(self.initialize)

        # Use the base_path for messages endpoint
        messages_path = "/messages/"
        messages_endpoint = (
            f"{base_path}{messages_path}" if base_path else messages_path
        )

        sse = SseServerTransport(messages_endpoint)

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.server.run(
                    streams[0], streams[1], self.server.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        logger.info(f"Starting SSE server on port {port}")
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)


# ===== CLI Entry Point =====
@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option(
    "--base-path",
    default="",
    help="Base path for URL construction (e.g. /mcp)",
    is_flag=False,
    flag_value="",
    required=False,
)
@click.option(
    "--config",
    type=click.Path(exists=False),
    help="Path to configuration file (default: config.json in script directory)",
)
def main(port: int, transport: str, base_path: str, config: Optional[str]) -> int:
    """Run the Mesh Agent Tools MCP Server.

    Connects to mesh API endpoints and provides tools for tool execution
    with fine-grained control over which tools are enabled per agent.
    """
    # Create server instance with configuration
    server = MeshToolServer(config_path=config)

    # Run with appropriate transport
    if transport == "sse":
        server.run_sse(port, base_path)
    else:
        anyio.run(server.run_stdio)

    return 0


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
