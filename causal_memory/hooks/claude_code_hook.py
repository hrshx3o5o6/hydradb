#!/usr/bin/env python3
"""
Claude Code hook entry point: reads a hook event JSON on stdin, ingests it
into HydraDNA via the reference adapter. Wire into .claude/settings.json:

  "hooks": {
    "PreToolUse": [{"hooks": [{"type": "command",
        "command": "<venv>/bin/python -m causal_memory.hooks.claude_code_hook"}]}],
    "PostToolUse": [...same...],
    "UserPromptSubmit": [...same...],
    "Stop": [...same...]
  }

Requires HYDRADB_URL / HYDRADB_TOKEN in the environment (same as the MCP
server). session_id is taken from the hook payload's own session_id field
so events from concurrent sessions land in separate Session anchors.
"""
import json
import os
import sys

from causal_memory.adapters.claude_code_adapter import ingest_hook_event
from causal_memory.memory import CausalMemory


def main() -> None:
    payload = json.load(sys.stdin)
    hook_event_name = payload.get("hook_event_name", "Unknown")
    session_id = payload.get("session_id", "unknown-session")

    mem = CausalMemory(
        url=os.environ.get("HYDRADB_URL", "http://localhost:18443"),
        auth_token=os.environ.get("HYDRADB_TOKEN", "local-dev-auth-token-32-characters-long"),
    )
    try:
        ingest_hook_event(mem, hook_event_name, payload, session_id)
    except Exception as e:
        # Never block the harness on a logging failure.
        print(f"[hydradna-hook] ingest failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
