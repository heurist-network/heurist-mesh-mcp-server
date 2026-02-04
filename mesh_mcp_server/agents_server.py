import asyncio
import contextlib
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp
import colorlog
import httpx
import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from .auth import (
    AuthContext,
    AuthValidator,
    get_current_auth_context,
    get_validator,
    set_current_auth_context,
)
from .usage_tracker import record_usage

load_dotenv()

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s%(reset)s:     %(message)s")
)
logger = colorlog.getLogger("mesh-mcp")
logger.handlers = []
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Suppress noisy MCP library logs (ClosedResourceError is expected in stateless mode)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("mcp.server.sse").setLevel(logging.CRITICAL)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.CRITICAL)
logging.getLogger("mcp").setLevel(logging.CRITICAL)
logging.getLogger("anyio").setLevel(logging.CRITICAL)

MESH_API_ENDPOINT = os.getenv("MESH_API_ENDPOINT", "https://mesh.heurist.xyz")
MESH_METADATA_ENDPOINT = os.getenv(
    "MESH_METADATA_ENDPOINT", "https://mesh.heurist.ai/metadata.json"
)

CURATED_AGENTS = [
    "TokenResolverAgent",
    "TrendingTokenAgent",
    "TwitterIntelligenceAgent",
    "ExaSearchDigestAgent",
    "FundingRateAgent",
    "AIXBTProjectInfoAgent",
    "ZerionWalletAnalysisAgent",
]

METADATA_REFRESH_INTERVAL = int(os.getenv("METADATA_REFRESH_INTERVAL", "600"))

# Cache for agent credits (populated from metadata)
AGENT_CREDITS: dict[str, int] = {}

# Transport security settings - disable DNS rebinding protection since we're behind nginx
TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)


@dataclass
class MetadataManager:
    """Manages agent metadata with periodic refresh capability."""

    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_refresh: float = 0
    refresh_interval: int = METADATA_REFRESH_INTERVAL
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def fetch_async(self) -> dict[str, dict[str, Any]]:
        """Fetch metadata asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    MESH_METADATA_ENDPOINT, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to fetch metadata: HTTP {response.status}"
                        )
                        return {}
                    data = await response.json()
                    return data.get("agents", {})
        except Exception as e:
            logger.error(f"Error fetching metadata async: {e}")
            return {}

    async def refresh(self) -> bool:
        """Refresh metadata if changed. Returns True if updated."""
        async with self._lock:
            new_metadata = await self.fetch_async()
            if not new_metadata:
                logger.warning("Failed to fetch new metadata, keeping existing")
                return False

            if new_metadata == self.metadata:
                logger.info("Metadata unchanged, skipping rebuild")
                self.last_refresh = time.time()
                return False

            self.metadata = new_metadata
            self.last_refresh = time.time()
            logger.info(f"Metadata refreshed: {len(self.metadata)} agents")
            return True

    async def start_refresh_loop(self):
        """Background task for periodic refresh."""
        logger.info(
            f"Starting metadata refresh loop (interval: {self.refresh_interval}s)"
        )
        while True:
            await asyncio.sleep(self.refresh_interval)
            try:
                logger.info("Running scheduled metadata refresh...")
                changed = await self.refresh()
                if changed:
                    await rebuild_mcp_servers()
            except Exception as e:
                logger.error(f"Refresh loop error: {e}")


METADATA_MANAGER = MetadataManager()


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


def get_agent_credits(agent_id: str) -> int:
    """Get the credit cost for an agent from cached metadata."""
    return AGENT_CREDITS.get(agent_id, 0)


def update_agent_credits_cache(metadata: dict[str, dict[str, Any]]) -> None:
    """Update the agent credits cache from metadata."""
    global AGENT_CREDITS
    new_credits = {}
    for agent_id, agent_data in metadata.items():
        agent_meta = agent_data.get("metadata", {})
        credits = agent_meta.get("credits", 0)
        if isinstance(credits, (int, float)):
            new_credits[agent_id] = int(credits)
        else:
            new_credits[agent_id] = 0
    AGENT_CREDITS = new_credits
    logger.info(f"Updated agent credits cache: {len(new_credits)} agents")


async def call_mesh_api(
    agent_id: str, tool_name: str, tool_params: dict[str, Any]
) -> dict[str, Any]:
    """Execute an agent tool via the mesh API.

    This function:
    1. Gets the current auth context (API key and user info)
    2. Deducts credits based on agent's credit cost
    3. Calls the mesh API with the user's API key
    """
    auth_ctx = get_current_auth_context()

    if auth_ctx and auth_ctx.heurist_key:
        api_key = auth_ctx.heurist_key
    elif "HEURIST_API_KEY" in os.environ:
        api_key = os.environ["HEURIST_API_KEY"]
    else:
        api_key = None
    credit_cost = get_agent_credits(agent_id)
    if auth_ctx and auth_ctx.user_id != "anonymous" and credit_cost > 0:
        validator = get_validator()
        success = validator.deduct_credits(auth_ctx.user_id, credit_cost)
        if not success:
            raise ValueError(
                f"Insufficient credits. This agent requires {credit_cost} credits."
            )
        logger.info(
            f"Deducted {credit_cost} credits from {auth_ctx.user_id} for {agent_id}"
        )
        asyncio.create_task(record_usage(auth_ctx.user_id, agent_id, credit_cost))

    async with aiohttp.ClientSession() as session:
        url = f"{MESH_API_ENDPOINT}/mesh_request"
        request_data = {
            "agent_id": agent_id,
            "input": {
                "tool": tool_name,
                "tool_arguments": tool_params,
                "raw_data_only": True,
            },
        }
        if api_key:
            request_data["api_key"] = api_key

        headers = {}
        if api_key:
            headers["X-HEURIST-API-KEY"] = api_key

        async with session.post(url, json=request_data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Mesh API error: {error_text}")
            return await response.json()


def sanitize_name(name: str) -> str:
    """Convert a name to a valid Python identifier."""
    name = name.lower()
    name = "".join(c if c.isalnum() else "_" for c in name)
    name = "_".join(filter(None, name.split("_")))
    if name and name[0].isdigit():
        name = f"tool_{name}"
    return name


def create_tool_id(agent_id: str, tool_name: str, max_length: int = 50) -> str:
    """Create a tool ID by combining agent ID and tool name.

    Truncates the agent ID if necessary to keep the total length within limits.
    This is needed because some MCP clients (like Cursor) have length limits.
    """
    agent_id_lower = sanitize_name(agent_id)
    tool_name_sanitized = sanitize_name(tool_name)
    separator = "_"

    max_agent_id_length = max_length - len(separator) - len(tool_name_sanitized)
    if max_agent_id_length > 0 and len(agent_id_lower) > max_agent_id_length:
        agent_id_lower = agent_id_lower[:max_agent_id_length]
        logger.info(
            f"Truncated agent ID for tool {tool_name}: {agent_id} -> {agent_id_lower}"
        )

    return f"{agent_id_lower}{separator}{tool_name_sanitized}"


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
    typed_tool_fn.__name__ = sanitize_name(tool_name)

    return typed_tool_fn


def register_agent_tools(
    mcp: FastMCP, agent_id: str, metadata: dict[str, Any], prefix: str = ""
) -> int:
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
        parameters = func_def.get("parameters", {"type": "object", "properties": {}})

        # Use create_tool_id for proper length limiting (Cursor compatibility)
        if prefix:
            # For curated MCP, include agent prefix in the tool ID
            mcp_tool_name = create_tool_id(prefix.rstrip("_"), tool_name)
        else:
            # For individual agent MCP, use agent_id + tool_name
            mcp_tool_name = create_tool_id(agent_id, tool_name)

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
        transport_security=TRANSPORT_SECURITY,
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
        transport_security=TRANSPORT_SECURITY,
    )
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    total_tools = 0
    for agent_id in agent_ids:
        if agent_id not in all_metadata:
            logger.warning(f"Curated agent {agent_id} not found in metadata")
            continue

        metadata = all_metadata[agent_id]
        prefix = f"{sanitize_name(agent_id)}_"
        tool_count = register_agent_tools(mcp, agent_id, metadata, prefix=prefix)
        total_tools += tool_count
        logger.info(f"Registered {tool_count} tools from {agent_id} to curated MCP")

    logger.info(f"Total tools in curated MCP: {total_tools}")
    return mcp


CURATED_MCP: FastMCP = None
AGENT_MCPS: dict[str, FastMCP] = {}
# Store the exit stack so we it start new session managers when rebuilding
_SESSION_STACK: Optional[contextlib.AsyncExitStack] = None


async def rebuild_mcp_servers():
    """Rebuild all MCP servers with current metadata."""
    global CURATED_MCP, AGENT_MCPS, _SESSION_STACK

    metadata = METADATA_MANAGER.metadata
    if not metadata:
        logger.warning("No metadata available, skipping rebuild")
        return

    logger.info("Rebuilding MCP servers with updated metadata...")

    update_agent_credits_cache(metadata)

    new_curated = create_curated_mcp(CURATED_AGENTS, metadata)
    new_agent_mcps = {}
    for agent_id, agent_metadata in metadata.items():
        agent_meta = agent_metadata.get("metadata", {})
        if agent_meta.get("hidden", False):
            continue
        new_agent_mcps[agent_id] = create_single_agent_mcp(agent_id, agent_metadata)

    # Initialize streamable_http_app for new MCPs (creates session managers)
    new_curated.streamable_http_app()
    for agent_mcp in new_agent_mcps.values():
        agent_mcp.streamable_http_app()

    if _SESSION_STACK is not None:
        await _SESSION_STACK.enter_async_context(new_curated.session_manager.run())
        for agent_mcp in new_agent_mcps.values():
            await _SESSION_STACK.enter_async_context(agent_mcp.session_manager.run())
        logger.info("Started session managers for rebuilt MCP servers")

    CURATED_MCP = new_curated
    AGENT_MCPS = new_agent_mcps

    logger.info(
        f"Rebuilt MCP servers: curated={len(CURATED_AGENTS)} agents, "
        f"individual={len(AGENT_MCPS)} agents"
    )


def initialize_servers():
    """Initialize MCP servers at startup (synchronous)."""
    global CURATED_MCP, AGENT_MCPS

    logger.info("Loading agent metadata at startup...")
    initial_metadata = fetch_agent_metadata_sync()
    METADATA_MANAGER.metadata = initial_metadata
    METADATA_MANAGER.last_refresh = time.time()
    logger.info(f"Loaded {len(initial_metadata)} agents")

    # Initialize agent credits cache
    update_agent_credits_cache(initial_metadata)
    try:
        get_validator()
    except Exception as e:
        logger.error(f"Failed to initialize auth validator: {e}")
        logger.warning(
            "Continuing without authentication - set AUTH_ENABLED=false to suppress"
        )

    CURATED_MCP = create_curated_mcp(CURATED_AGENTS, initial_metadata)
    logger.info(f"Created curated MCP server with {len(CURATED_AGENTS)} agents")

    AGENT_MCPS = {}
    for agent_id, metadata in initial_metadata.items():
        agent_meta = metadata.get("metadata", {})
        if agent_meta.get("hidden", False):
            continue
        AGENT_MCPS[agent_id] = create_single_agent_mcp(agent_id, metadata)

    logger.info(f"Created {len(AGENT_MCPS)} individual agent MCP servers")


initialize_servers()


async def health_check(request):
    """Health check endpoint with refresh status."""
    return JSONResponse(
        {
            "status": "healthy",
            "curated_agents": len(CURATED_AGENTS),
            "total_agents": len(AGENT_MCPS),
            "last_refresh": METADATA_MANAGER.last_refresh,
            "refresh_interval_seconds": METADATA_MANAGER.refresh_interval,
            "seconds_since_refresh": int(time.time() - METADATA_MANAGER.last_refresh),
        }
    )


async def refresh_metadata(request):
    """POST /mcp/refresh - Manually trigger metadata refresh."""
    try:
        changed = await METADATA_MANAGER.refresh()
        if changed:
            await rebuild_mcp_servers()
            return JSONResponse(
                {
                    "status": "refreshed",
                    "agents_count": len(METADATA_MANAGER.metadata),
                    "last_refresh": METADATA_MANAGER.last_refresh,
                }
            )
        return JSONResponse(
            {
                "status": "unchanged",
                "agents_count": len(METADATA_MANAGER.metadata),
                "last_refresh": METADATA_MANAGER.last_refresh,
            }
        )
    except Exception as e:
        logger.error(f"Manual refresh failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        tools.append(
            {
                "name": tool_name,
                "description": func_def.get("description", ""),
                "parameters": func_def.get("parameters", {}).get("properties", {}),
                "required": func_def.get("parameters", {}).get("required", []),
            }
        )
    return tools


async def list_agents(request):
    """List all available agents and their endpoints."""
    base_url = "https://mesh.heurist.xyz"
    agents = []
    for agent_id, metadata in METADATA_MANAGER.metadata.items():
        agent_meta = metadata.get("metadata", {})
        if agent_meta.get("hidden", False):
            continue
        agents.append(
            {
                "id": agent_id,
                "name": agent_meta.get("name", agent_id),
                "description": agent_meta.get("description", ""),
                "endpoints": {
                    "streamable_http": f"{base_url}/mcp/agents/{agent_id}/",
                    "sse": f"{base_url}/mcp/agents/{agent_id}/sse",
                },
            }
        )

    return JSONResponse(
        {
            "curated": {
                "agents": CURATED_AGENTS,
                "endpoints": {
                    "streamable_http": f"{base_url}/mcp/",
                    "sse": f"{base_url}/mcp/sse",
                },
            },
            "all_agents": agents,
        }
    )


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Manage lifespan of all MCP servers and background tasks."""
    global _SESSION_STACK

    refresh_task = asyncio.create_task(METADATA_MANAGER.start_refresh_loop())
    logger.info(
        f"Started metadata refresh background task "
        f"(interval: {METADATA_MANAGER.refresh_interval}s)"
    )

    CURATED_MCP.streamable_http_app()
    for agent_mcp in AGENT_MCPS.values():
        agent_mcp.streamable_http_app()

    async with contextlib.AsyncExitStack() as stack:
        _SESSION_STACK = stack

        await stack.enter_async_context(CURATED_MCP.session_manager.run())
        for agent_mcp in AGENT_MCPS.values():
            await stack.enter_async_context(agent_mcp.session_manager.run())
        logger.info("All MCP session managers started")
        yield

    # Cleanup
    _SESSION_STACK = None
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    logger.info("All MCP session managers and refresh task stopped")


class DynamicMCPMiddleware:
    """Middleware that dynamically routes MCP protocol requests to current instances.

    This is the KEY to making tool definitions update without server restart.
    Instead of static Mount objects that capture MCP instances at startup,
    this middleware reads from the CURRENT global CURATED_MCP and AGENT_MCPS
    on every request, so updates from rebuild_mcp_servers() take effect immediately.

    Additionally, this middleware handles API key authentication:
    1. Extracts API key from Authorization header
    2. Validates against DynamoDB
    3. Checks credits
    4. Sets auth context for downstream use
    """

    # JSON API endpoints that should pass through to regular Starlette routes
    API_ENDPOINTS = {"/mcp/health", "/mcp/agents", "/mcp/refresh"}

    def __init__(self, app):
        self.app = app

    def _get_header(self, scope: dict, name: str) -> Optional[str]:
        """Extract a header value from the ASGI scope."""
        name_lower = name.lower().encode()
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == name_lower:
                return header_value.decode()
        return None

    def _get_query_param(self, scope: dict, name: str) -> Optional[str]:
        """Extract a query parameter from the ASGI scope."""
        query_string = scope.get("query_string", b"").decode()
        if not query_string:
            return None
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                if key == name:
                    return value
        return None

    async def _send_error_response(
        self, scope, receive, send, status_code: int, message: str
    ):
        """Send an error JSON response."""
        response = JSONResponse({"error": message}, status_code=status_code)
        await response(scope, receive, send)

    async def _authenticate(
        self, scope
    ) -> tuple[bool, Optional[str], Optional[AuthContext]]:
        """Authenticate the request and return (success, error_message, auth_context).

        Returns:
            (True, None, AuthContext) on success
            (False, error_message, None) on failure
        """
        try:
            validator = get_validator()
        except Exception as e:
            logger.error(f"Auth validator not available: {e}")
            return True, None, None

        if not validator.enabled:
            return True, None, None

        # Try to get API key from headers (X-HEURIST-API-KEY preferred, then Authorization)
        x_heurist_key = self._get_header(scope, "x-heurist-api-key")
        auth_header = self._get_header(scope, "authorization")
        api_key = AuthValidator.extract_api_key_from_headers(auth_header, x_heurist_key)

        if not api_key:
            api_key = self._get_query_param(scope, "api_key")

        if not api_key:
            return (
                False,
                "API key required. Provide via X-HEURIST-API-KEY header or api_key query parameter.",
                None,
            )
        try:
            auth_ctx = validator.validate_api_key(api_key)
            return True, None, auth_ctx
        except ValueError as e:
            return False, str(e), None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        if path.startswith("/mcp") and not path.endswith("/"):
            if path not in self.API_ENDPOINTS:
                path = path + "/"
                scope = dict(scope)
                scope["path"] = path

        if path.rstrip("/") in self.API_ENDPOINTS:
            await self.app(scope, receive, send)
            return

        success, error_msg, auth_ctx = await self._authenticate(scope)
        if not success:
            await self._send_error_response(scope, receive, send, 401, error_msg)
            return

        token = set_current_auth_context(auth_ctx)

        try:
            match = re.match(r"^/mcp/agents/([^/]+)(/sse)?(/.*)?$", path)
            if match:
                agent_id = match.group(1)
                use_sse = match.group(2) == "/sse"
                remaining = match.group(3) or "/"

                if agent_id not in AGENT_MCPS:
                    response = JSONResponse(
                        {"error": f"Agent '{agent_id}' not found"}, status_code=404
                    )
                    await response(scope, receive, send)
                    return

                mcp = AGENT_MCPS[agent_id]
                new_scope = dict(scope)
                new_scope["path"] = remaining

                mcp_app = mcp.sse_app() if use_sse else mcp.streamable_http_app()
                await mcp_app(new_scope, receive, send)
                return

            if path.startswith("/mcp/sse"):
                new_scope = dict(scope)
                new_scope["path"] = path[8:] or "/"
                await CURATED_MCP.sse_app()(new_scope, receive, send)
                return

            if path.startswith("/mcp"):
                new_scope = dict(scope)
                new_scope["path"] = path[4:] or "/"
                await CURATED_MCP.streamable_http_app()(new_scope, receive, send)
                return

            await self.app(scope, receive, send)
        finally:
            set_current_auth_context(None)


# Build routes - only JSON API endpoints
# MCP protocol endpoints are handled by DynamicMCPMiddleware for true dynamic routing
routes = [
    Route("/mcp/health", endpoint=health_check),
    Route("/mcp/agents", endpoint=list_agents),
    Route("/mcp/refresh", endpoint=refresh_metadata, methods=["POST"]),
]
app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(DynamicMCPMiddleware)],
)

logger.info(
    "Routes configured with dynamic MCP routing - "
    "Curated: /mcp, /mcp/sse | Individual: /mcp/agents/<agent_id> | "
    "API: /mcp/health, /mcp/agents, /mcp/refresh"
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Mesh MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
