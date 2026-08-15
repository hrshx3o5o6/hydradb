"""
HydraDB causal-memory MCP server.

Pure graph layer — NO LLM calls inside. The host harness's own model does all
causal reasoning: it asks for a candidate window (`propose_edges_window` or
`recent_events`), decides which events caused which (its own inference), then
persists the result via `link`. `why` returns raw paths; the host model
narrates them.

Run (stdio):
    HYDRADB_URL=http://localhost:18443 \
    HYDRADB_TOKEN=... \
    python -m causal_memory.mcp_server

Register in opencode.json:
    "mcp": { "hydradb-memory": {
        "type": "local",
        "command": ["<venv>/bin/python", "-m", "causal_memory.mcp_server"],
        "env": { "HYDRADB_URL": "http://localhost:18443", ... }
    }}
"""

import os
from dataclasses import asdict
from typing import Any

import json
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.context import ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from .memory import CausalMemory


async def _on_list_tools(
    _ctx: ServerRequestContext,
    _params: PaginatedRequestParams | None,
) -> ListToolsResult:
    return ListToolsResult(tools=await list_tools())


async def _on_call_tool(
    _ctx: ServerRequestContext,
    params: CallToolRequestParams,
) -> CallToolResult:
    results = await call_tool(params.name, params.arguments or {})
    return CallToolResult(content=results)


server = Server(
    "hydradb-causal-memory",
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


def _mem() -> CausalMemory:
    return CausalMemory(
        url=os.environ.get("HYDRADB_URL", "http://localhost:18443"),
        auth_token=os.environ.get("HYDRADB_TOKEN", "local-dev-auth-token-32-characters-long"),
        admin_url=os.environ.get("HYDRADB_ADMIN_URL", "http://localhost:9090"),
    )


def _event_json(e) -> dict[str, Any]:
    if e is None:
        return {"id": None, "text": None, "timestamp": None, "session_id": None,
                "type": None, "topic": None}
    return {"id": e.id, "text": e.text, "timestamp": e.timestamp,
            "session_id": e.session_id, "type": e.event_type, "topic": e.topic}


def _relation_json(r) -> dict[str, Any]:
    if r is None:
        return {"source_id": None, "target_id": None, "type": None,
                "confidence": None, "mechanism": None}
    return {"source_id": r.source_id, "target_id": r.target_id,
            "type": r.relation_type, "confidence": r.confidence,
            "mechanism": r.mechanism}


def _path_json(p) -> dict[str, Any]:
    return {
        "events": [_event_json(e) for e in p.events],
        "relations": [_relation_json(r) for r in p.relations],
        "weight": p.path_weight,
    }


async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="observe",
            description=(
                "Durable-cally remember an event/action with dedup. Call after "
                "every material action in the session (wrote code, ran test, "
                "made a decision, hit an error). Same (session, text, time) "
                "resolves to the same vertex."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "session_id": {"type": "string"},
                    "topic": {"type": "string"},
                    "event_type": {"type": "string"},
                    "timestamp": {"type": "integer"},
                },
                "required": ["text", "session_id"],
            },
        ),
        types.Tool(
            name="recent_events",
            description=(
                "Last N remembered events in reverse-chronological order. Feed "
                "this to yourself as the candidate window for causal reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        ),
        types.Tool(
            name="propose_edges_window",
            description=(
                "Candidate window for edge discovery: temporally ordered event "
                "pairs from recent history. Use YOUR OWN model to decide which "
                "pairs are causally related, then persist with `link`."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "window_events": {"type": "integer"},
                    "max_candidates": {"type": "integer"},
                },
            },
        ),
        types.Tool(
            name="link",
            description=(
                "Persist a causal edge (source CAUSES target) with confidence "
                "and a mechanism sentence. Idempotent (MERGE). This is the "
                "'you reasoned it, we remember it' write."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "target_id": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "mechanism": {"type": "string"},
                },
                "required": ["source_id", "target_id"],
            },
        ),
        types.Tool(
            name="search",
            description="Find remembered events whose text contains the keyword.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["text"],
            },
        ),
        types.Tool(
            name="get_event",
            description="Fetch one event by integer id.",
            inputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "integer"}},
                "required": ["event_id"],
            },
        ),
        types.Tool(
            name="find_causes",
            description="Causal ancestors of an event: paths leading INTO it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "max_hops": {"type": "integer"},
                },
                "required": ["event_id"],
            },
        ),
        types.Tool(
            name="find_effects",
            description="Causal descendants of an event: paths leading OUT of it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "max_hops": {"type": "integer"},
                },
                "required": ["event_id"],
            },
        ),
        types.Tool(
            name="causal_path",
            description="All causal paths between two events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer"},
                    "target_id": {"type": "integer"},
                    "max_hops": {"type": "integer"},
                },
                "required": ["source_id", "target_id"],
            },
        ),
        types.Tool(
            name="why",
            description=(
                "Resolve a target question to its event, return its causal "
                "ancestry as raw paths. YOU narrate the answer from the paths "
                "returned — the graph hands you the structure, you write the "
                "words."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "max_hops": {"type": "integer"},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="graph_dump",
            description=(
                "Full causal graph as nodes+edges for visualization (cytoscape "
                "friendly). Use to render the live graph."
            ),
            inputSchema={"type": "object", "properties": {"max_hops": {"type": "integer"}}},
        ),
        types.Tool(
            name="reset",
            description="Wipe the causal memory store (dangerous).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    mem = _mem()

    if name == "observe":
        t = arguments.get("timestamp")
        e = mem.observe(
            arguments["text"],
            session_id=arguments["session_id"],
            topic=arguments.get("topic"),
            timestamp=t,
        )
        return [types.TextContent(type="text", text=f"event id={e.id} stored")]
    if name == "recent_events":
        events = mem._recent_events(limit=arguments.get("limit", 20))
        return [types.TextContent(type="text", text=json.dumps([_event_json(e) for e in events]))]
    if name == "propose_edges_window":
        pairs, _offset = _candidate_pairs(mem, arguments.get("window_events", 30),
                                          arguments.get("max_candidates", 60))
        return [types.TextContent(type="text", text=json.dumps(pairs))]
    if name == "link":
        r = mem.add_causal_relation(
            source_id=arguments["source_id"],
            target_id=arguments["target_id"],
            relation_type="CAUSES",
            confidence=float(arguments.get("confidence", 1.0)),
            mechanism=arguments.get("mechanism"),
        )
        return [types.TextContent(type="text",
                                  text=f"linked {r.source_id} CAUSES {r.target_id} (conf {r.confidence})")]
    if name == "search":
        events = mem.search(arguments["text"], limit=arguments.get("limit", 10))
        return [types.TextContent(type="text", text=json.dumps([_event_json(e) for e in events]))]
    if name == "get_event":
        return [types.TextContent(type="text",
                                  text=json.dumps(_event_json(mem.get_event(arguments["event_id"]))))]
    if name == "find_causes":
        paths = mem.find_causes(arguments["event_id"], max_hops=arguments.get("max_hops", 3))
        return [types.TextContent(type="text", text=json.dumps([_path_json(p) for p in paths]))]
    if name == "find_effects":
        paths = mem.find_all_effects(arguments["event_id"], max_hops=arguments.get("max_hops", 3))
        return [types.TextContent(type="text", text=json.dumps([_path_json(p) for p in paths]))]
    if name == "causal_path":
        paths = mem.find_causal_path(
            source_id=arguments["source_id"],
            target_id=arguments["target_id"],
            max_hops=arguments.get("max_hops", 5),
        )
        return [types.TextContent(type="text", text=json.dumps([_path_json(p) for p in paths]))]
    if name == "why":
        target_id = mem._match_event_by_keywords(arguments["question"])
        if target_id is None:
            return [types.TextContent(type="text", text="Could not resolve question to an event.")]
        paths = mem.find_causes(target_id, max_hops=arguments.get("max_hops", 8))
        paths.sort(key=lambda p: -len(p.events))
        return [types.TextContent(type="text",
                                  text=json.dumps({"target_id": target_id,
                                            "paths": [_path_json(p) for p in paths]}))]
    if name == "graph_dump":
        return [types.TextContent(type="text", text=json.dumps(_dump_json(mem)))]
    if name == "reset":
        mem.reset()
        return [types.TextContent(type="text", text="store reset")]
    raise ValueError(f"Unknown tool: {name}")


def _candidate_pairs(mem, window_events: int, max_candidates: int):
    """Temporally ordered (cause?, effect?, texts) pairs for the host LLM."""
    events = mem._recent_events(limit=window_events)
    events.reverse()  # oldest -> newest
    pairs = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            ready = {"cause_id": events[i].id, "effect_id": events[j].id,
                     "cause_text": events[i].text, "effect_text": events[j].text}
            pairs.append(ready)
            if len(pairs) >= max_candidates:
                return pairs, window_events
    return pairs, window_events


def _dump_json(mem):
    """nodes + edges for a normal directed-graph renderer."""
    query = (
        "MATCH (e:Event) "
        "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic"
    )
    rows = mem.client.execute(query)
    nodes = [{"id": r["e.id"], "text": r["e.text"], "session_id": r["e.session_id"],
              "type": r["e.type"], "topic": r["e.topic"]} for r in rows]
    edges = []
    for rel_type in ("CAUSES", "ENABLES", "OVERWRITES", "CONFLICTS"):
        try:
            rows = mem.client.execute(
                f"MATCH (a:Event)-[r:{rel_type}]->(b:Event) "
                "RETURN a.id AS source, b.id AS target, r.confidence AS confidence"
            )
        except Exception:
            continue
        for r in rows:
            if r.get("source") is None or r.get("target") is None:
                continue
            edges.append({
                "source": r["source"], "target": r["target"],
                "type": rel_type,
                "confidence": float(r.get("confidence", 1.0) or 1.0),
            })
    return {"nodes": nodes, "edges": edges}


async def main() -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="hydradb-causal-memory",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())