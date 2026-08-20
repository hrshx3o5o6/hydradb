"""
hydradna console entry point.

Runs the HydraDNA causal-memory MCP server over stdio, ready for any MCP
client (Claude Desktop, opencode, etc.):

    HYDRADB_URL=http://localhost:18444 hydradna

Register in an MCP client as:
    { "hydradna": { "type": "local",
                    "command": ["hydradna"],
                    "env": { "HYDRADB_URL": "http://localhost:18444",
                             "HYDRADB_TOKEN": "local-dev-auth-token-32-characters-long" } } }
"""

import argparse
import asyncio

__version__ = "0.1.0"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="hydradna",
        description="HydraDNA causal-memory MCP server (stdio transport).",
    )
    ap.add_argument(
        "--version",
        action="version",
        version=f"hydradna {__version__}",
    )
    ap.parse_args(argv)

    from .mcp_server import main as server_main

    asyncio.run(server_main())


if __name__ == "__main__":
    main()