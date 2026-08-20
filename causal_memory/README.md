# Causal Memory Layer

A causal graph-based memory system for AI agents, built on [HydraDB](https://github.com/hydra-db/hydradb).

## The Problem

Vector databases find **similar** memories but can't answer **WHY**.

- "My agent remembers I moved to SF" ✓
- "But can't trace WHY I love my neighborhood" ✗

## The Solution

Store **cause→effect** relationships, not just facts.

```
moved to SF → found apartment → loves neighborhood
```

Now the agent can answer:
- "Why do you love your neighborhood?" → traces causal chain
- "What did moving cause?" → finds all effects
- "Where do you live?" → returns current fact (handles overwrites)

## Quick Start

### 1. Start HydraDB

```bash
# Using Docker (recommended)
docker pull ghcr.io/hydra-db/hydradb:latest
# Follow https://github.com/hydra-db/hydradb README for setup

# Or build from source
git clone https://github.com/hydra-db/hydradb.git
cd hydradb
just native-check
cargo build --locked --features server-runtime --bin graph-node
```

### 2. Install Causal Memory

```bash
pip install -r requirements.txt
```

### 3. Use It

```python
from causal_memory import CausalMemory

memory = CausalMemory(
    url="http://localhost:18443",
    auth_token="your-auth-token"
)

# Add events
e1 = memory.add_event(
    text="User moved to San Francisco",
    session_id="session-1",
    topic="location"
)

e2 = memory.add_event(
    text="Found apartment in Mission",
    session_id="session-2",
    topic="housing"
)

# Create causal relationship
memory.add_causal_relation(
    source_id=e1.id,
    target_id=e2.id,
    relation_type="CAUSES",
    confidence=0.9,
    mechanism="relocation led to housing search"
)

# Query causal paths
paths = memory.find_causal_path(e1.id, e2.id)

# Search
results = memory.search("San Francisco")

# Get current fact (handles overwrites)
current = memory.find_current_fact("location")
```

## Architecture

```
┌─────────────────────────────────────┐
│   Demo App (Streamlit)              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Causal Memory Layer (Python SDK)  │
│   - add_event()                     │
│   - add_causal_relation()           │
│   - find_causal_path()              │
│   - find_all_effects()              │
│   - find_current_fact()             │
│   - why() (coming soon)             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   HydraDB Client (HTTP/Bolt)        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   HydraDB                           │
│   - Graph storage                   │
│   - Path traversal (GraphBLAS)      │
│   - Causal consistency              │
└─────────────────────────────────────┘
```

## Graph Schema

### Events (Vertices)
```cypher
(:Event {
  id: "e1",
  text: "User moved to SF",
  timestamp: 1700000000,
  session_id: "session-15",
  type: "fact",
  topic: "location"
})
```

### Causal Relationships (Edges)
```cypher
(:Event)-[:CAUSES {
  confidence: 0.85,
  mechanism: "relocation",
  timestamp: 1700000000,
  evidence: ["session-15"]
}]->(:Event)
```

### Overwrites (When facts change)
```cypher
(:Event)-[:OVERWRITES {
  reason: "moved again",
  timestamp: 1700100000
}]->(:Event)
```

## Key Queries

### Find causal chain
```cypher
CALL algo.SPpaths({
  sourceNode: 'e1',
  targetNode: 'e3',
  relTypes: ['CAUSES'],
  maxLen: 5,
  pathCount: 3
}) YIELD path RETURN path
```

### Find all effects
```cypher
CALL algo.SSpaths({
  sourceNode: 'e1',
  relTypes: ['CAUSES'],
  maxLen: 3,
  pathCount: 20
}) YIELD path RETURN path
```

### Get current fact (not overwritten)
```cypher
MATCH (e:Event {topic: "location"})
WHERE NOT (e)<-[:OVERWRITES]-()
RETURN e
```

## Features

- ✅ Event storage with temporal metadata
- ✅ Causal relationship tracking (CAUSES, OVERWRITES, CONFLICTS)
- ✅ Causal path queries (using HydraDB's `algo.SPpaths`)
- ✅ Effect discovery (using `algo.SSpaths`)
- ✅ Temporal reasoning (handles overwrites)
- ✅ Current fact retrieval
- 🔄 LLM integration for natural language queries
- 🔄 "Why" query interface
- 🔄 Counterfactual reasoning

## Built For

[Hack Hydra](https://hackhydra.hydradb.com) - Track 03: Memory + Context Retrieval

## Install

```bash
pip install hydradna          # or: uvx hydradna --version
```

Requires a running HydraDB node (see Docker Compose below).

## Run the MCP server (stdio)

```bash
HYDRADB_URL=http://localhost:18444 \
HYDRADB_TOKEN=local-dev-auth-token-32-characters-long \
hydradna
```

Register in an MCP client (Claude Desktop, opencode, ...):

```json
{ "hydradna": { "type": "local",
    "command": ["hydradna"],
    "env": { "HYDRADB_URL": "http://localhost:18444",
             "HYDRADB_TOKEN": "local-dev-auth-token-32-characters-long" } } }
```

Exposes 12 tools: `observe`, `recent_events`, `propose_edges_window`,
`link`, `search`, `get_event`, `find_causes`, `find_effects`, `causal_path`,
`why`, `graph_dump`, `reset`.

## Docker Compose (zero-config HydraDB)

```bash
cd deploy
mkdir -p data/store data/cache
printf 'local-dev-auth-token-32-characters-long' > data/auth-token
docker compose up -d hydradb
# node up on 18443 (bolt) / 18444 (query API) / 19090 (admin)
```

Then run `hydradna` on the host pointed at `http://localhost:18444`, or start
the packaged `hydradna` container with `docker compose up -d hydradna`.

## License

MIT
