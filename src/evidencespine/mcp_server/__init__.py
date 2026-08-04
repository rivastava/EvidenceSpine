"""EvidenceSpine MCP server package.

Requires the optional ``[mcp]`` extra (the official ``mcp`` SDK). The SDK is
imported lazily so the core package stays dependency-free.
"""

from evidencespine.mcp_server.server import create_server, run_server

__all__ = ["create_server", "run_server"]
