"""Running the MCP server."""

import logging

from mcp_server.core.server import mcp


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)












