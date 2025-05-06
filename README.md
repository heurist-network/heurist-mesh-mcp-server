# Mesh Agent MCP Server

![mesh-2](https://github.com/user-attachments/assets/da67c0e6-0709-4f28-ab82-8abfc0c0734c)

A Model Context Protocol (MCP) server that connects to [Heurist Mesh](https://github.com/heurist-network/heurist-agent-framework/tree/main/mesh) APIs, providing Claude with access to various blockchain and web3 tools.

Heurist Mesh is an open network of purpose-built AI agents and tools, each specialized in particular web3 domains such as blockchain data analysis, smart contract security, token metrics, and blockchain interaction. We are actively growing the Heurist Mesh ecosystem, continuously integrating more tools to expand its capabilities.

<a href="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server/badge" alt="Mesh Agent Server MCP server" />
</a>

## Features
- Connects to the Heurist Mesh API 
- Loads tools for cryptocurrency data and Web3 use cases
- Supports both SSE and stdio transports
- Works with Claude in Cursor, Claude Desktop, and other MCP-compatible interfaces
- Use one API key to access multiple services (e.g. CoinGecko crypto market data, GoPlus token security review)

## 🔥 Just In: Customize Your Agents and Create Managed MCP Servers On-Demand
You can use [Heurist Mesh MCP Portal](https://mcp.heurist.ai/) to create SSE MCP Servers. Select your agents and compose a personalized swarm for your tasks!

## Hosted SSE Endpoint
We provide a hosted SSE endpoint at https://sequencer-v2.heurist.xyz/mcp/sse. This includes all the tools from the following commonly used agents: `CoingeckoTokenInfoAgent`, `ElfaTwitterIntelligenceAgent`, `ExaSearchAgent`, `DexScreenerTokenInfoAgent`, `ZerionWalletAnalysisAgent`. This is a shared server and the performance may be unstable.

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
Visit https://mesh.heurist.ai/metadata.json or https://mcp.heurist.ai/ to view all available tools.

## Featured tools
| Tool Name | Description | Agent | Parameters | Required Params |
|-----------|-------------|-------|------------|----------------|
| get_coingecko_id | Search for a token by name to get its CoinGecko ID | CoinGeckoTokenInfoAgent | **token_name** (string): The token name to search for | token_name |
| get_token_info | Get detailed token information and market data using CoinGecko ID (you can't use the token address or name or symbol) | CoinGeckoTokenInfoAgent | **coingecko_id** (string): The CoinGecko ID of the token | coingecko_id |
| get_trending_coins | Get the current top trending cryptocurrencies on CoinGecko | CoinGeckoTokenInfoAgent | - | None |
| get_specific_pair_info | Get trading pair info by chain and pair address on DexScreener | DexScreenerTokenInfoAgent | **chain** (string): Chain identifier (e.g., solana, bsc, ethereum, base)<br>**pair_address** (string): The pair contract address to look up | chain, pair_address |
| get_token_pairs | Get the trading pairs by chain and token address on DexScreener | DexScreenerTokenInfoAgent | **chain** (string): Chain identifier (e.g., solana, bsc, ethereum, base)<br>**token_address** (string): The token contract address to look up all pairs for | chain, token_address |
| get_token_profiles | Get the basic info of the latest tokens from DexScreener | DexScreenerTokenInfoAgent | - | None |
| search_pairs | Search for trading pairs on DexScreener by token name, symbol, or address | DexScreenerTokenInfoAgent | **search_term** (string): Search term (token name, symbol, or address) | search_term |
| get_trending_tokens | Get current trending tokens on Twitter | ElfaTwitterIntelligenceAgent | **time_window** (string): Time window to analyze | None |
| search_account | Analyze a Twitter account with both mention search and account stats | ElfaTwitterIntelligenceAgent | **username** (string): Twitter username to analyze (without @)<br>**days_ago** (integer): Number of days to look back for mentions<br>**limit** (integer): Maximum number of mention results | username |
| search_mentions | Search for mentions of specific tokens or topics on Twitter | ElfaTwitterIntelligenceAgent | **keywords** (array): List of keywords to search for<br>**days_ago** (integer): Number of days to look back<br>**limit** (integer): Maximum number of results (minimum: 20) | keywords |
| answer | Get a direct answer to a question using Exa's answer API | ExaSearchAgent | **question** (string): The question to answer | question |
| search | Search for webpages related to a query | ExaSearchAgent | **search_term** (string): The search term<br>**limit** (integer): Maximum number of results to return (default: 10) | search_term |
| search_and_answer | Perform both search and answer operations for a query | ExaSearchAgent | **topic** (string): The topic to search for and answer | topic |
| execute_search | Execute a web search query by reading the web pages | FirecrawlSearchAgent | **search_term** (string): The search term to execute | search_term |
| generate_queries | Generate related search queries for a topic that can expand the research | FirecrawlSearchAgent | **topic** (string): The main topic to research<br>**num_queries** (integer): Number of queries to generate | topic |
| fetch_security_details | Fetch security details of a blockchain token contract | GoplusAnalysisAgent | **contract_address** (string): The token contract address<br>**chain_id** (['integer', 'string']): The blockchain chain ID or 'solana' for Solana tokens. Supported chains: Ethereum (1), Optimism (10), Cronos (25), BSC (56), Gnosis (100), HECO (128), Polygon (137), Fantom (250), KCC (321), zkSync Era (324), ETHW (10001), FON (201022), Arbitrum (42161), Avalanche (43114), Linea Mainnet (59144), Base (8453), Tron (tron), Scroll (534352), opBNB (204), Mantle (5000), ZKFair (42766), Blast (81457), Manta Pacific (169), Berachain Artio Testnet (80085), Merlin (4200), Bitlayer Mainnet (200901), zkLink Nova (810180), X Layer Mainnet (196), Solana (solana) | contract_address |

## Customizing Supported Agents
The server comes with a default set of agents. To modify which agents are available:
1. Open the `server.py` file and locate the `Config` class.
2. Edit the `DEFAULT_AGENTS` list to add or remove agents listed [at Heurist Metadata](https://mesh.heurist.ai/metadata.json)
```python
DEFAULT_AGENTS = [
    "CoinGeckoTokenInfoAgent",
    "DexScreenerTokenInfoAgent",
    "ElfaTwitterIntelligenceAgent",
    "ExaSearchAgent",
    "FirecrawlSearchAgent",
    "GoplusAnalysisAgent",
    # Add agents here
    "NewAgent"
]
```

## License
This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License.