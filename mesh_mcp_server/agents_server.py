import contextlib
import inspect
import logging
import os
from typing import Any, Optional

import aiohttp
import colorlog
import httpx
import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

load_dotenv()

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s%(reset)s:     %(message)s")
)
logger = colorlog.getLogger("mesh-mcp")
logger.handlers = []
logger.addHandler(handler)
logger.setLevel(logging.INFO)

MESH_API_ENDPOINT = os.getenv("MESH_API_ENDPOINT", "https://sequencer-v2.heurist.xyz")
MESH_METADATA_ENDPOINT = os.getenv(
    "MESH_METADATA_ENDPOINT", "https://mesh.heurist.ai/mesh_agents_metadata.json"
)

CURATED_AGENTS = [
    "CoinGeckoTokenInfoAgent",
    "ElfaTwitterIntelligenceAgent",
    "ExaSearchAgent",
    "FirecrawlSearchAgent",
    "ZerionWalletAnalysisAgent",
    "DexScreenerTokenInfoAgent",
    "TrendingTokenAgent",
    "AIXBTProjectInfoAgent",
]


def fetch_agent_metadata_sync() -> dict[str, dict[str, Any]]:
    """Fetch agent metadata synchronously at startup."""
    logger.info(f"Fetching agent metadata from {MESH_METADATA_ENDPOINT}")
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(MESH_METADATA_ENDPOINT)
            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch agent metadata: HTTP {response.status_code}"
                )
                return {}
            data = response.json()
            return data.get("agents", {})
    except Exception as e:
        logger.error(f"Error fetching agent metadata: {e}")
        return {}


async def call_mesh_api(agent_id: str, tool_name: str, tool_params: dict[str, Any]) -> dict[str, Any]:
    """Execute an agent tool via the mesh API."""
    async with aiohttp.ClientSession() as session:
        url = f"{MESH_API_ENDPOINT}/mesh_request"
        request_data = {
            "agent_id": agent_id,
            "input": {
                "tool_name": tool_name,
                **tool_params
            }
        }
        if "HEURIST_API_KEY" in os.environ:
            request_data["api_key"] = os.environ["HEURIST_API_KEY"]

        headers = {}
        if "HEURIST_API_KEY" in os.environ:
            headers["X-HEURIST-API-KEY"] = os.environ["HEURIST_API_KEY"]

        async with session.post(url, json=request_data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Mesh API error: {error_text}")
            return await response.json()


def sanitize_tool_name(name: str) -> str:
    """Convert a tool name to a valid Python function name."""
    name = name.lower()
    name = "".join(c if c.isalnum() else "_" for c in name)
    name = "_".join(filter(None, name.split("_")))
    if name and name[0].isdigit():
        name = f"tool_{name}"
    return name


def make_agent_tool(agent_id: str, tool_name: str, parameters: dict[str, Any]):
    """Factory function to create an agent tool with proper closure and dynamic parameters."""
    properties = parameters.get("properties", {})
    required_params = parameters.get("required", [])
    param_names = list(properties.keys())

    async def tool_fn(**kwargs) -> str:
        result = await call_mesh_api(agent_id, tool_name, kwargs)
        return str(result)
    params = []
    annotations = {"return": str}

    for param_name, param_def in properties.items():
        param_type = param_def.get("type", "string")
        is_required = param_name in required_params
        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(param_type, str)

        if is_required:
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=inspect.Parameter.empty,
                )
            )
            annotations[param_name] = python_type
        else:
            default_val = param_def.get("default", None)
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default_val,
                )
            )
            annotations[param_name] = Optional[python_type]
    new_sig = inspect.Signature(params)

    async def typed_tool_fn(**kwargs) -> str:
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = await call_mesh_api(agent_id, tool_name, filtered_kwargs)
        return str(result)

    typed_tool_fn.__signature__ = new_sig
    typed_tool_fn.__annotations__ = annotations
    typed_tool_fn.__name__ = sanitize_tool_name(tool_name)

    return typed_tool_fn


def register_agent_tools(mcp: FastMCP, agent_id: str, metadata: dict[str, Any], prefix: str = "") -> int:
    """Register all tools from an agent's metadata to an MCP server.

    Returns the number of tools registered.
    """
    tools = metadata.get("tools", [])
    registered = 0

    for tool_def in tools:
        if tool_def.get("type") != "function":
            continue

        func_def = tool_def.get("function", {})
        tool_name = func_def.get("name")
        if not tool_name:
            continue

        description = func_def.get("description", f"Execute {tool_name}")
        parameters = func_def.get("parameters", {"type": "object", "properties":
        mcp_tool_name = f"{prefix}{sanitize_tool_name(tool_name)}" if prefix else sanitize_tool_name(tool_name)

        tool_fn = make_agent_tool(agent_id, tool_name, parameters)
        mcp.tool(name=mcp_tool_name, description=description)(tool_fn)
        registered += 1

    return registered


def create_single_agent_mcp(agent_id: str, metadata: dict[str, Any]) -> FastMCP:
    """Create an MCP server for a single agent with all its tools."""
    mcp = FastMCP(
        name=f"mesh-{agent_id}",
        stateless_http=True,
        json_response=True,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    tool_count = register_agent_tools(mcp, agent_id, metadata)
    logger.info(f"Registered {tool_count} tools for agent {agent_id}")

    return mcp


def create_curated_mcp(
    agent_ids: list[str], all_metadata: dict[str, dict[str, Any]]
) -> FastMCP:
    """Create an MCP server with all tools from curated agents."""
    mcp = FastMCP(
        name="mesh-curated-agents",
        stateless_http=True,
        json_response=True,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    total_tools = 0
    for agent_id in agent_ids:
        if agent_id not in all_metadata:
            logger.warning(f"Curated agent {agent_id} not found in metadata")
            continue

        metadata = all_metadata[agent_id]
        prefix = f"{sanitize_tool_name(agent_id)}_"
        tool_count = register_agent_tools(mcp, agent_id, metadata, prefix=prefix)
        total_tools += tool_count
        logger.info(f"Registered {tool_count} tools from {agent_id} to curated MCP")

    logger.info(f"Total tools in curated MCP: {total_tools}")
    return mcp


logger.info("Loading agent metadata at startup...")
ALL_METADATA = fetch_agent_metadata_sync()
logger.info(f"Loaded {len(ALL_METADATA)} agents")

CURATED_MCP = create_curated_mcp(CURATED_AGENTS, ALL_METADATA)
logger.info(f"Created curated MCP server with {len(CURATED_AGENTS)} agents")

AGENT_MCPS: dict[str, FastMCP] = {}
for agent_id, metadata in ALL_METADATA.items():
    agent_meta = metadata.get("metadata", {})
    if agent_meta.get("hidden", False):
        continue
    AGENT_MCPS[agent_id] = create_single_agent_mcp(agent_id, metadata)

logger.info(f"Created {len(AGENT_MCPS)} individual agent MCP servers")


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
            "curated_agents": len(CURATED_AGENTS),
            "total_agents": len(AGENT_MCPS),
        }
    )


def get_agent_tools(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool definitions from agent metadata."""
    tools = []
    for tool_def in metadata.get("tools", []):
        if tool_def.get("type") != "function":
            continue
        func_def = tool_def.get("function", {})
        tool_name = func_def.get("name")
        if not tool_name:
            continue
        tools.append({
            "name": tool_name,
            "description": func_def.get("description", ""),
            "parameters": func_def.get("parameters", {}).get("properties", {}),
            "required": func_def.get("parameters", {}).get("required", []),
        })
    return tools


async def list_agents(request):
    """List all available agents and their endpoints."""
    agents = []
    for agent_id, metadata in ALL_METADATA.items():
        agent_meta = metadata.get("metadata", {})
        if agent_meta.get("hidden", False):
            continue
        agents.append(
            {
                "id": agent_id,
                "name": agent_meta.get("name", agent_id),
                "description": agent_meta.get("description", ""),
                "tools": get_agent_tools(metadata),
                "endpoints": {
                    "streamable_http": f"/mcp/agents/{agent_id}",
                    "sse": f"/mcp/agents/{agent_id}/sse",
                },
            }
        )

    return JSONResponse(
        {
            "curated": {
                "agents": CURATED_AGENTS,
                "endpoints": {
                    "streamable_http": "/mcp",
                    "sse": "/mcp/sse",
                },
            },
            "all_agents": agents,
        }
    )


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Manage lifespan of all MCP servers."""
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(CURATED_MCP.session_manager.run())
        for agent_mcp in AGENT_MCPS.values():
            await stack.enter_async_context(agent_mcp.session_manager.run())
        logger.info("All MCP session managers started")
        yield
    logger.info("All MCP session managers stopped")


class TrailingSlashMiddleware:
    """Add trailing slash to MCP routes if missing, except for health and agents endpoints."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if path.startswith("/mcp") and not path.endswith("/"):
                if path not in ["/mcp/health", "/mcp/agents"]:
                    scope["path"] = path + "/"
        await self.app(scope, receive, send)


routes = [
    Route("/mcp/health", endpoint=health_check),
    Route("/mcp/agents", endpoint=list_agents),
]

for agent_id, agent_mcp in AGENT_MCPS.items():
    routes.append(Mount(f"/mcp/agents/{agent_id}/sse", app=agent_mcp.sse_app()))
    routes.append(Mount(f"/mcp/agents/{agent_id}", app=agent_mcp.streamable_http_app()))

routes.append(Mount("/mcp/sse", app=CURATED_MCP.sse_app()))
routes.append(Mount("/mcp", app=CURATED_MCP.streamable_http_app()))

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(TrailingSlashMiddleware)],
)

logger.info(
    "Routes configured - Curated: /mcp, /mcp/sse | Individual: /mcp/agents/<agent_id>"
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Mesh MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
