# Heurist Mesh MCP Server

![mesh-2](https://github.com/user-attachments/assets/da67c0e6-0709-4f28-ab82-8abfc0c0734c)

A Model Context Protocol (MCP) server that connects to [Heurist Mesh](https://github.com/heurist-network/heurist-agent-framework/tree/main/mesh) APIs - your gateway to Web3 intelligence.

**[Heurist Mesh](https://mesh.heurist.ai)** is the skills marketplace for AI agents. General-purpose AI models lack specialized knowledge about Web3 and often fail to deliver accurate results. Heurist Mesh provides 30+ specialized AI agents that are experts in crypto analytics, ready to give your AI applications the Web3 expertise they need.

<a href="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server/badge" alt="Mesh Agent Server MCP server" />
</a>

## Features
- **Gateway to Web3 Intelligence**: Access 30+ specialized crypto analytics agents via MCP
- **Optimized for AI**: Input/output formats optimized for AI agents - fewer tool calls, less token usage
- Supports both SSE and stdio transports
- Works with Claude, Cursor, Claude Desktop, and other MCP-compatible interfaces
- Use one API key to access multiple services (CoinGecko, DexScreener, Twitter analytics, and more)

## 🔥 Just In: Customize Your Agents and Create Managed MCP Servers On-Demand
You can use [Heurist Mesh Console](https://mesh.heurist.ai/console) to create SSE MCP Servers. Select your agents and compose a personalized swarm for your tasks!

## Hosted SSE Endpoint
We provide a hosted SSE endpoint at https://mesh.heurist.xyz/mcp/sse. **Authentication is required** - you need a [Heurist API key](https://dev-api-form.heurist.ai/) (use invite code "claude" for free credits).

Provide your API key via:
- `X-HEURIST-API-KEY` header (recommended)
- `Authorization: Bearer <your-api-key>` header
- `api_key` query parameter

This endpoint includes tools from recommended agents for comprehensive Web3 intelligence:

| Agent | Description |
|-------|-------------|
| `TokenResolverAgent` | Find tokens by address/symbol/name, return normalized profiles and top DEX pools |
| `TrendingTokenAgent` | Aggregates trending tokens from GMGN, CoinGecko, Pump.fun, Dexscreener, Zora and Twitter |
| `TwitterIntelligenceAgent` | Twitter/X timeline, tweet detail, and smart search |
| `ExaSearchDigestAgent` | Web search with concise LLM summarization |
| `FundingRateAgent` | Binance funding rates, open interest, and spot-futures opportunities |
| `AIXBTProjectInfoAgent` | Trending project info, fundamental analysis, and market summary |
| `ZerionWalletAnalysisAgent` | EVM wallet token and NFT holdings analysis |

This is a shared server and the performance may be unstable. For production use, we recommend self-hosting or using [Heurist Mesh Console](https://mesh.heurist.ai/console) to create dedicated servers.

Cursor can directly access SSE servers. For Claude Desktop users, we recommend installing [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) to connect to the SSE server.

## Prerequisites

- Python 3.10 or higher
- UV package manager (recommended)
- OR Docker
- [Get a Heurist API key](https://docs.heurist.ai/protocol-overview/credits) or you can [claim free API credits](https://dev-api-form.heurist.ai/) with invite code "claude"

## Installation
### Using UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/heurist-network/heurist-mesh-mcp-server.git
cd heurist-mesh-mcp-server

# Install the package
uv pip install -e .
```

### Using Docker
```bash
# Clone the repository
git clone https://github.com/heurist-network/heurist-mesh-mcp-server.git
cd heurist-mesh-mcp-server

# Build the Docker image
docker build -t mesh-tool-server .
```
## Usage
### Option 1: Run with stdio Transport (for Claude Desktop)
#### Using UV
To use this with Claude Desktop, add the following to your `claude_desktop_config.json`:
```bash
{
  "mcpServers": {
    "heurist-mesh-agent": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/heurist-mesh-mcp-server/mesh_mcp_server",  // Update this path
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
#### Using Docker
Alternatively, you can use Docker with Claude Desktop by adding this to your `claude_desktop_config.json`:
```bash
{
  "mcpServers": {
    "mesh-agent": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e", "TRANSPORT=stdio",
        "-e", "HEURIST_API_KEY=your-api-key-here",  // Update this key
        "mesh-tool-server"
      ]
    }
  }
}
```

Replace `/path/to/heurist-mesh-mcp-server` with the actual path to the repository and `your-api-key-here` with your Heurist API key.

### Option 2: Run with SSE Transport (for Cursor)
#### Setting up Environment Variables in `.env`
```bash
cp .env.example .env
HEURIST_API_KEY=your-api-key-here
```
#### Using UV:
```bash
uv run mesh-tool-server --transport sse --port 8000
```
#### Using Docker:
```bash
docker run -p 8000:8000 -e PORT=8000 mesh-tool-server
```
Then, in Cursor, add the MCP Server URL: **http://0.0.0.0:8000/sse**

## Available Tools
Visit https://mesh.heurist.ai/metadata.json or https://mesh.heurist.ai/console to view all 30+ available agents and their tools.

## Recommended Tools
These tools from our recommended agents cover most Web3 intelligence use cases:

| Tool Name | Description | Agent |
|-----------|-------------|-------|
| token_search | Find tokens by address, symbol, name, or CoinGecko ID | TokenResolverAgent |
| token_profile | Get comprehensive token profile with market data, socials, and top pools | TokenResolverAgent |
| get_trending_tokens | Aggregated trending tokens from multiple sources | TrendingTokenAgent |
| get_market_summary | AI-generated market summary across all trending sources | TrendingTokenAgent |
| twitter_search | Smart Twitter search for crypto topics | TwitterIntelligenceAgent |
| user_timeline | Get recent tweets from a Twitter user | TwitterIntelligenceAgent |
| tweet_detail | Get detailed info about a specific tweet | TwitterIntelligenceAgent |
| exa_web_search | Web search with AI summarization | ExaSearchDigestAgent |
| exa_scrape_url | Scrape and summarize webpage content | ExaSearchDigestAgent |
| get_all_funding_rates | Get funding rates for all Binance perpetual contracts | FundingRateAgent |
| get_symbol_oi_and_funding | Get open interest and funding for a specific symbol | FundingRateAgent |
| find_spot_futures_opportunities | Find arbitrage opportunities between spot and futures | FundingRateAgent |
| search_projects | Search trending projects with fundamental analysis | AIXBTProjectInfoAgent |
| get_market_summary | Get AI market summary from AIXBT | AIXBTProjectInfoAgent |
| fetch_wallet_tokens | Get EVM wallet token holdings | ZerionWalletAnalysisAgent |
| fetch_wallet_nfts | Get EVM wallet NFT holdings | ZerionWalletAnalysisAgent |

## Customizing Supported Agents
The server comes with a default set of recommended agents. To modify which agents are available:
1. Open the `server.py` file and locate the `Config` class.
2. Edit the `DEFAULT_AGENTS` list to add or remove agents listed [at Heurist Metadata](https://mesh.heurist.ai/metadata.json)
```python
DEFAULT_AGENTS = [
    "TokenResolverAgent",        # Comprehensive token lookup
    "TrendingTokenAgent",        # Aggregated trending tokens
    "TwitterIntelligenceAgent",  # Twitter/X analytics
    "ExaSearchDigestAgent",      # Web search with summarization
    "FundingRateAgent",          # Binance funding & OI data
    "AIXBTProjectInfoAgent",     # Project analysis
    "ZerionWalletAnalysisAgent", # Wallet holdings
    # Add more agents as needed
]
```

## License
This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License.