"""Standardised response wrappers for every MCP tool."""

from typing import Any, Dict


def success(data: Any) -> Dict[str, Any]:

    """success!!!!!!!!!!"""

    return {"success": True, "data": data, "error": None}


def failure(message: str) -> Dict[str, Any]:
    """error *boooooooo*"""
    return {"success": False, "data": None, "error": message}