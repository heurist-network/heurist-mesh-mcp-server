import contextlib
import logging
import os
from typing import Any

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


async def call_mesh_api(agent_id: str, query: str) -> dict[str, Any]:
    """Execute an agent query via the mesh API."""
    async with aiohttp.ClientSession() as session:
        url = f"{MESH_API_ENDPOINT}/mesh_request"
        request_data = {"agent_id": agent_id, "input": {"query": query}}
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


def create_tool_name(agent_id: str) -> str:
    """Convert agent ID to a valid tool function name."""
    name = agent_id.lower()
    name = "".join(c if c.isalnum() else "_" for c in name)
    name = "_".join(filter(None, name.split("_")))
    if name and name[0].isdigit():
        name = f"agent_{name}"
    return name


def make_agent_tool(agent_id: str):
    """Factory function to create an agent tool with proper closure."""

    async def tool_fn(query: str) -> str:
        result = await call_mesh_api(agent_id, query)
        return str(result)

    return tool_fn


def create_single_agent_mcp(agent_id: str, metadata: dict[str, Any]) -> FastMCP:
    """Create an MCP server for a single agent."""
    agent_meta = metadata.get("metadata", {})
    description = agent_meta.get("description", f"Query the {agent_id} agent")

    mcp = FastMCP(
        name=f"mesh-{agent_id}",
        stateless_http=True,
        json_response=True,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    tool_fn = make_agent_tool(agent_id)
    mcp.tool(name=create_tool_name(agent_id), description=description)(tool_fn)

    return mcp


def create_curated_mcp(
    agent_ids: list[str], all_metadata: dict[str, dict[str, Any]]
) -> FastMCP:
    """Create an MCP server with multiple curated agents as tools."""
    mcp = FastMCP(
        name="mesh-curated-agents",
        stateless_http=True,
        json_response=True,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    for agent_id in agent_ids:
        if agent_id not in all_metadata:
            logger.warning(f"Curated agent {agent_id} not found in metadata")
            continue

        metadata = all_metadata[agent_id]
        agent_meta = metadata.get("metadata", {})
        description = agent_meta.get("description", f"Query the {agent_id} agent")

        tool_fn = make_agent_tool(agent_id)
        mcp.tool(name=create_tool_name(agent_id), description=description)(tool_fn)

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
