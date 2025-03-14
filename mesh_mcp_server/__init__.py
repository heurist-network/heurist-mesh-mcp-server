"""
Mesh MCP Server - A Model Context Protocol server for accessing mesh agents.
"""

# Import these conditionally to avoid circular imports during testing
try:
    from .server import app
    __all__ = ["app"]
except ImportError:
    # During testing or when server.py doesn't exist yet
    __all__ = []

__version__ = "0.1.0" 