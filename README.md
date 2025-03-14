# Mesh Agent MCP Server
A Model Context Protocol (MCP) server that connects to Heurist's mesh agent APIs, providing Claude with access to various blockchain and web3 research tools.

## Features
- Connects to the Heurist mesh agent API
- Dynamically loads tools from multiple agents
- Supports both SSE and stdio transports
- Works with Claude in Cursor, Claude Desktop, and other MCP-compatible interfaces

## Prerequisites

- Python 3.10 or higher
- UV package manager (recommended)
- OR Docker

## Installation
### Using UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/mesh-agent-mcp.git
cd mesh-agent-mcp

# Install the package
uv pip install -e .
```

### Using Docker
```bash
# Clone the repository
git clone https://github.com/yourusername/mesh-agent-mcp.git
cd mesh-agent-mcp

# Build the Docker image
docker build -t mesh-tool-server .
```
## Usage
### Option 1: Run with stdio Transport (for Claude Desktop)
To use this with Claude Desktop, add the following to your claude_desktop_config.json:
```bash
{
  "mcpServers": {
    "mesh-agent": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mesh-agent-mcp",  // Update this path
        "run",
        "mesh-tool-server"
      ],
      "env": {
        "HEURIST_API_KEY": "your-api-key-here"  // Update this key
      }
    }
  }
}
```
Replace /path/to/mesh-agent-mcp with the actual path to the repository and your-api-key-here with your Heurist API key.

### Option 2: Run with SSE Transport (for Cursor)
#### Using UV:
```bash
uv run mesh-tool-server --transport sse --port 8000
```
#### Using Docker:
```bash
docker run -p 8000:8000 -e PORT=8000 mesh-tool-server
```
Then, in Cursor, add the MCP Server URL: http://0.0.0.0:8000/sse

## License
This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License.