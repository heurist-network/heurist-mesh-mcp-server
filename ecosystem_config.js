module.exports = {
    apps: [
        {
            name: "mcp-server",
            script: ".venv/bin/uvicorn",
            args: "mesh_mcp_server.agents_server:app --host 0.0.0.0 --port 8001",
            cwd: "/home/appuser/heurist-mesh-mcp-server",
            interpreter: "none",
            env: {
                MESH_API_ENDPOINT: "https://mesh.heurist.xyz",
                MESH_METADATA_ENDPOINT: "https://mesh.heurist.ai/mesh_agents_metadata.json",
                PORT: "8001"
            },
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: "500M",
            log_date_format: "YYYY-MM-DD HH:mm:ss Z",
            error_file: "/home/appuser/.pm2/logs/mcp-server-error.log",
            out_file: "/home/appuser/.pm2/logs/mcp-server-out.log"
        }
    ]
};
