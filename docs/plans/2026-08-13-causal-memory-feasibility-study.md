---
title: "Causal Memory Layer — Feasibility Study & Architecture"
status: draft-for-review
date: 2026-08-13
tags:
  - hackathon
  - track-03
  - agent-memory
  - causal-discovery
  - architecture
---

# Causal Memory Layer — Feasibility Study

## The Problem (Distilled)

You have Obsidian. You build knowledge graphs. But you never **use** them.

Why? Because Obsidian gives you **connections** (links between notes), not **reasoning** (why things are connected). The graph is pretty but inert. It stores relationships but doesn't think with them.

The same problem exists in AI agent memory:

| What exists today | What it does | What it can't do |
|---|---|---|
| **Vector DB** (Mem0, LangChain) | Find similar memories | Know why they're related |
| **Entity graph** (Zep, Mem0 Platform) | Find related entities | Reason about cause/effect |
| **Chat buffer** (MemGPT) | Remember recent context | Remember across 40 sessions |
| **Obsidian graph** | Show note connections | Answer "what depends on what?" |

**The gap:** No system stores **why** things are connected. No system reasons about causality, temporal ordering, or contradictions.

---

## The Opportunity

**Causal memory** — store not just "what happened" but "why it happened" and "what would happen if..."

### What This Enables

```
User (Session 3):  "I'm moving to San Francisco"
User (Session 15): "I found a great apartment in the Mission"
User (Session 28): "I love my new neighborhood"

Vector DB sees:     3 memories about "San Francisco" / "apartment" / "neighborhood"
                    → Returns all 3, can't distinguish relevance

Causal Graph sees:  moving → found apartment → loves neighborhood
                    → Can answer: "Why do you love your neighborhood?"
                    → Can answer: "What caused you to move?"
                    → Can answer: "If you hadn't moved, would you love your neighborhood?"
```

This is impossible with vector search. This is causal memory's superpower.

---

## Competitive Landscape

### Who Exists

| System | Architecture | Temporal | Contradictions | Causal | Open Source |
|--------|-------------|----------|---------------|--------|-------------|
| **Mem0** (63k stars) | Vector + SQL + Entity | v3 scoring | ADD-only, no resolution | No | Yes |
| **MemGPT/Letta** (24k stars) | OS-inspired hierarchy | No | Manual core memory edits | No | Yes |
| **LangChain Memory** | Pluggable types | No | No | No | Yes |
| **Zep** (enterprise) | Context Graph | Validity timestamps | Auto-invalidate | No (co-occurrence) | No |

### Who Doesn't Exist

**Nobody** combines causal discovery + graph storage + temporal reasoning for agent memory.

Closest academic work:
- **REMI** (Sep 2025): Causal schema memory for personalized agents
- **GRAVITY** (May 2026): Causal traces for temporal reasoning (+7.5-10% on LongMemEval)
- **Causal-AgentIR** (Jul 2026): Causal memory graph for image restoration agents

These are research prototypes, not production tools. The space is wide open.

---

## What HydraDB Actually Gives You

Based on codebase deep dive (not marketing):

### APIs Available

| Capability | How | Limitation |
|---|---|---|
| **Graph storage** | Vertices + edges with properties | Flat properties only (no nested objects) |
| **Cypher queries** | MATCH, WHERE, RETURN, CREATE, MERGE | Subset of OpenCypher (no WITH, no unbounded traversals) |
| **Path procedures** | `algo.SPpaths`, `algo.SSpaths`, `algo.MSpaths` | Bounded maxLen required, max ~32 hops |
| **Temporal reads** | Bookmark-based causal consistency | Read at a specific point in graph history |
| **GraphBLAS traversal** | Matrix-accelerated BFS | Via native path procedures |
| **Python client** | Neo4j driver (Bolt) or HTTP wrapper | No dedicated SDK |

### What You Build Yourself

| Component | Effort | Notes |
|---|---|---|
| **Causal reasoning logic** | Medium | Python layer on top of HydraDB primitives |
| **Causal discovery** | Medium | Use causal-learn / DoWhy / LLM extraction |
| **Python SDK** | Low | ~200 LOC wrapping HTTP API |
| **Demo app** | Low | Streamlit or Gradio |

### Key Insight

HydraDB is **storage + traversal**, not reasoning. You build the causal logic in Python. This is actually good — iterate on causal reasoning without touching Rust.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Demo Application                        │
│              (Streamlit / Gradio / CLI)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               Causal Memory Layer (Python SDK)               │
│                                                               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Causal        │  │ Temporal     │  │ Explanation      │  │
│  │ Reasoning     │  │ Queries      │  │ Generation       │  │
│  │               │  │              │  │                  │  │
│  │ • adjustSet() │  │ • bookmarks  │  │ • why()          │  │
│  │ • backdoor()  │  │ • read_epoch │  │ • what_if()      │  │
│  │ • doEffect()  │  │ • temporal   │  │ • trace()        │  │
│  └───────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│          │                  │                   │             │
│  ┌───────▼──────────────────▼───────────────────▼─────────┐  │
│  │              HydraDB Client (Python)                    │  │
│  │  • Neo4j driver (Bolt) for graph queries               │  │
│  │  • HTTP client for bookmarks + temporal reads          │  │
│  └──────────────────────────┬─────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                        HydraDB                               │
│  • graph-node (Bolt :17687, HTTP :18443, Admin :19091)      │
│  • GraphBLAS-accelerated traversal                           │
│  • Bookmark-based causal consistency                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Model

```cypher
// Vertices = Events, Entities, Facts
(Event {
  id: "e1",
  text: "User moved to SF",
  timestamp: 1700000000,
  session_id: "session-15",
  type: "fact"
})

// Edges = Causal relationships with metadata
(Event)-[:CAUSES {
  confidence: 0.85,
  mechanism: "relocation",
  evidence: ["session-15", "session-16"],
  timestamp: 1700000000
}]->(Event)

(Event)-[:OVERWRITES {
  reason: "moved",
  timestamp: 1700000000
}]->(Event)

(Event)-[:CONFLICTS {
  confidence: 0.3,
  resolution: "temporal_supersession"
}]->(Event)
```

### Key Queries

```cypher
// Find causal chain: "Why does the user love their neighborhood?"
CALL algo.SPpaths({
  sourceNode: $moving_id,
  targetNode: $loves_neighborhood_id,
  relTypes: ['CAUSES'],
  maxLen: 5,
  pathCount: 3
}) YIELD path, pathWeight
RETURN path

// Find all effects of a cause: "What did moving to SF cause?"
CALL algo.SSpaths({
  sourceNode: $moving_id,
  relTypes: ['CAUSES'],
  maxLen: 3,
  pathCount: 20
}) YIELD path
RETURN path

// Find contradictions: "What conflicts with this fact?"
MATCH (a)-[r:CONFLICTS]->(b)
WHERE a.id = $fact_id
RETURN b, r.confidence, r.resolution
```

---

## 9-Day Build Plan

### Phase 1: Infrastructure (Days 1-2)

**Goal:** HydraDB running, Python wrapper working, basic CRUD tested.

- [ ] Get HydraDB running locally (Docker or build from source)
- [ ] Verify Bolt + HTTP APIs work
- [ ] Build `CausalMemory` Python class (~200 LOC)
- [ ] Test basic CRUD: create events, create causal edges
- [ ] Test path queries with `algo.SPpaths`

**Deliverable:** Python script that stores 5 events with causal edges and queries causal paths.

### Phase 2: Causal Graph Model (Days 3-4)

**Goal:** Schema designed, causal edge types working, temporal properties stored.

- [ ] Design causal schema (events, causal edges, overwrite edges, conflict edges)
- [ ] Implement `store_event()` with temporal metadata
- [ ] Implement `store_causal_relation()` with confidence + mechanism
- [ ] Implement `store_overwrite()` for fact updates
- [ ] Test with toy dataset: "User moved to SF → found apartment → loves neighborhood"

**Deliverable:** Working causal graph with temporal reasoning.

### Phase 3: Causal Reasoning (Days 5-6)

**Goal:** Basic causal reasoning operations working.

- [ ] Implement `find_causal_paths()` using `algo.SPpaths`
- [ ] Implement `find_all_effects()` using `algo.SSpaths`
- [ ] Implement `find_confounders()` (find common causes)
- [ ] Implement `find_overwrites()` (temporal supersession)
- [ ] Implement basic `why()` — trace causal chain backward
- [ ] Implement basic `what_if()` — remove edge, recompute effects

**Deliverable:** Python methods for causal reasoning queries.

### Phase 4: LLM Integration (Day 7)

**Goal:** Natural language → causal query → natural language explanation.

- [ ] LLM extracts causal triples from conversation text
- [ ] LLM generates causal graph from domain description
- [ ] LLM interprets causal query results as natural language
- [ ] Implement `add_conversation()` — LLM extracts + stores causal facts
- [ ] Implement `ask()` — natural language → Cypher → LLM explanation

**Deliverable:** Chat interface that extracts causal facts and answers "why" questions.

### Phase 5: Demo App (Days 8-9)

**Goal:** Polished demo, README, video.

- [ ] Build Streamlit demo app
- [ ] Record 3-minute demo video
- [ ] Write README with quickstart, architecture, examples
- [ ] Write architecture doc (this document)
- [ ] Prepare submission form

**Deliverable:** Complete hackathon submission.

---

## What's Realistic vs. Ambitious

### Realistic (9 days)

- Causal knowledge graph storage in HydraDB
- Causal pathfinding using native procedures
- Basic causal reasoning (adjustment sets, backdoor paths)
- Temporal causal queries via bookmarks
- LLM integration for natural language interface
- Python wrapper (~500 LOC total)
- Streamlit demo
- 3-minute video

### Ambitious (skip unless ahead of schedule)

- Full do-calculus with complex interventions
- Custom Rust procedures (requires recompilation)
- Integration with causal-learn / DoWhy libraries
- Distributed causal inference across multiple cells
- Real-time causal discovery from streaming data
- Benchmark on LongMemEval (requires significant data prep)

---

## The Wedge: Why Developers Would Use This

### The Pain

Every AI agent developer hits this wall:
- "My agent forgets everything between sessions"
- "I store memories but can't ask WHY the user prefers something"
- "Vector search finds similar memories but not related ones"
- "When facts change, my agent gets confused"

### The Wedge

```python
pip install causal-memory

from causal_memory import CausalMemory

memory = CausalMemory()

# Add conversation — LLM extracts causal facts automatically
memory.add(messages=[
    {"role": "user", "content": "I'm moving to San Francisco"},
    {"role": "assistant", "content": "Exciting! What neighborhood?"}
], session_id="session-15")

# Query — causal retrieval, not just semantic
results = memory.search("Where does the user live?")
# → "San Francisco" (with causal chain: moved → found apartment → loves neighborhood)

# The killer feature — ask WHY
explanation = memory.why("Why does the user love their neighborhood?")
# → "Because they moved to SF → found apartment in Mission → loves the neighborhood"
```

### Why Causal > Vector

| Question | Vector DB | Causal Memory |
|----------|-----------|---------------|
| "Where does the user live?" | Returns 3 memories about SF | Returns "SF" with causal chain |
| "Why do they love their neighborhood?" | Returns similar memories | Traces: moved → apartment → loves |
| "What if they hadn't moved?" | Returns nothing | Simulates: no move → no apartment → ? |
| "When did they decide to move?" | Returns memories about "decide" | Traces causal chain backward |

### The "Wow" Moment

**Ask the agent WHY you prefer something, and it traces back to the original conversation where you explained it.**

This is impossible with vector search. This is causal memory's superpower.

---

## Reading List

### Must Read (Papers)

1. **Mem0** — arXiv:2504.19413 (production memory architecture)
2. **REMI** — arXiv:2509.06269 (causal schema memory for agents)
3. **GRAVITY** — arXiv:2605.01688 (causal traces for temporal reasoning)
4. **LongMemEval** — arXiv:2410.10813 (benchmark)
5. **BEAM** — arXiv:2510.27246 (scale benchmark)
6. **Causal Reasoning + LLMs** — arXiv:2305.00050 (GPT-4 at 97% causal discovery)

### Must Read (Books)

7. **"The Book of Why"** — Judea Pearl (2018). Causality for humans.
8. **"Elements of Causal Inference"** — Peters, Janzing, Schölkopf (2017). Free PDF from MIT Press.

### Must Read (Code)

9. **HydraDB architecture.md** — System design, query execution, traversal
10. **HydraDB src/shard/path_procedure.rs** — Native path procedures (SPpaths, SSpaths, MSpaths)
11. **HydraDB src/client/http.rs** — HTTP API format
12. **DoWhy** — github.com/py-why/dowhy (causal inference in Python)
13. **causal-learn** — github.com/py-why/causal-learn (causal discovery algorithms)

### Must Read (Competitors)

14. **Mem0 source** — github.com/mem0ai/mem0 (how they do memory extraction)
15. **Zep docs** — getzep.com (how they do temporal graphs)
16. **LangChain Memory** — python.langchain.com/docs/how_to/chatbots_memory/

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| HydraDB Cypher too limited | Medium | High | Build causal logic in Python, not Cypher |
| Causal discovery too slow | Low | Medium | Use LLM extraction instead of statistical methods |
| Demo too complex | Medium | High | Start simple: 5 events, 3 causal edges, 1 "why" query |
| Judges don't understand causal | Low | High | Lead with demo, explain after |
| Time runs out | Medium | High | Phase 1-3 is a complete project. Phase 4-5 is polish. |

---

## Decision: Build It

### Why This Project

1. **Fits Track 03 perfectly** — agent memory with temporal reasoning, contradictions, abstention
2. **Novel** — nobody combines causal discovery + graph storage for memory
3. **Feasible** — HydraDB gives you everything you need, build causal logic in Python
4. **Demoable** — "Ask WHY" is a 10-second wow moment
5. **Useful** — every AI agent developer needs this
6. **Academic backing** — recent papers (REMI, GRAVITY, Causal-AgentIR) validate the approach

### Why HydraDB

1. **Graph-native** — built for graph workloads, not bolted on
2. **GraphBLAS** — fast traversal for causal path queries
3. **Bookmark-based temporal reads** — read at a specific point in history
4. **OpenCypher** — familiar query language
5. **Open source** — Apache 2.0, hackathon-friendly
6. **SlateDB backend** — scales to millions of nodes/edges

### Why Now

1. **LLMs + causality breakthrough** — GPT-4 at 97% causal discovery (2023 paper)
2. **Agent memory is hot** — Mem0 raised $24M, 63k stars
3. **No integrated tool** — gap in the market
4. **Hackathon timeline** — 9 days is tight but doable
5. **Track 03 alignment** — perfect fit for the challenge

---

## Next Steps

1. **Read the reading list** (especially architecture.md, path_procedure.rs, REMI paper)
2. **Get HydraDB running** (Docker or build from source)
3. **Build the Python wrapper** (Day 1-2)
4. **Design the causal schema** (Day 3)
5. **Build the demo** (Day 8-9)

---

*This document is a living document. Update as you learn more.*
