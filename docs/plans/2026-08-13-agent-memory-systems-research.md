# Agent Memory Systems Research Report
**Date:** 2026-08-13  
**Purpose:** Inform architecture decision for causal memory layer using HydraDB

---

## Executive Summary

Current agent memory systems fall into three categories:
1. **Vector-based RAG** (Mem0, LangChain) - semantic similarity retrieval
2. **OS-inspired hierarchical** (MemGPT/Letta) - virtual memory management
3. **Graph-based** (Zep, Mem0 Platform) - relational structure with varying temporal support

**Key Gap:** No system combines **causal discovery** with **graph storage** for agent memory. Existing graph approaches (Zep, Mem0 Platform) use co-occurrence graphs, not causal graphs. This represents a significant opportunity for HydraDB.

---

## 1. Mem0

**GitHub:** https://github.com/mem0ai/mem0 (63.2k stars)  
**Paper:** arXiv:2504.19413 (April 2025)

### Architecture
- **Core:** Extracts facts from conversations → stores in vector DB + SQL + entity store
- **Retrieval:** Multi-signal fusion (semantic + BM25 + entity + temporal)
- **Graph Memory (Platform only):** Native entity graph, no external DB needed
  - Entities become nodes
  - Shared entities create connections
  - Co-occurrence-based (not causal)

### Memory Operations
```python
# Add memory
memory.add(messages, user_id="alice")

# Search
results = memory.search(query="What does Alice prefer?", top_k=3)
```

### Strengths
- Simple API, production-ready
- Multi-level memory (User, Session, Agent)
- Entity linking across memories
- Temporal reasoning (v3 algorithm, April 2026)
- Benchmarks: 92.5 LoCoMo, 94.4 LongMemEval

### Limitations
- **No causal reasoning** - stores facts, not cause-effect
- **ADD-only extraction** (v3) - memories accumulate, no overwrites
- **Contradictions:** Stores both old and new facts, relies on temporal scoring
- **Multi-hop:** Limited to entity co-occurrence, no causal chains
- **Open-source:** No graph memory (Platform-only feature)

### Temporal Handling
- Time-aware retrieval ranks dated instances
- Extracts timestamps at write time
- Scores based on query's temporal intent

---

## 2. MemGPT / Letta

**GitHub:** https://github.com/letta-ai/letta (24.2k stars)  
**Paper:** arXiv:2310.08560 (October 2023)

### Architecture
**OS-inspired virtual context management:**
- **Main Context (Working Memory):** In-context window, always visible to LLM
- **Archival Memory (Long-term):** Vector DB, retrieved on demand
- **Recall Memory (Episodic):** Conversation history, searchable
- **Core Memory Blocks:** Editable by agent via tools

### Memory Hierarchy
```
┌─────────────────────────────────────┐
│  System Prompt + Core Memory Blocks │  ← Always in context
├─────────────────────────────────────┤
│  Working Memory (Recent Messages)   │  ← Sliding window
├─────────────────────────────────────┤
│  Archival Memory (Vector DB)        │  ← Retrieved on demand
├─────────────────────────────────────┤
│  Recall Memory (Full History)       │  ← Searchable archive
└─────────────────────────────────────┘
```

### Key Features
- **Self-editing memory:** Agent can modify its own core memory via tools
- **Compaction:** Automatically summarizes old messages
- **Conversations:** Multiple threads per agent
- **Shared memory blocks:** Across agents

### Strengths
- Sophisticated memory management
- Agent can learn and self-improve
- Handles very long conversations
- Durable execution (state persisted)

### Limitations
- **No causal reasoning** - stores messages, not causal structure
- **No temporal reasoning** - no explicit time-aware retrieval
- **Contradictions:** Relies on LLM to resolve via core memory updates
- **Multi-hop:** Limited to vector similarity
- **Complex setup** - more infrastructure than Mem0

### API Model
```typescript
const agent = await client.createAgent({
  model: "claude-opus-4-8",
  human: "User info here",
  persona: "Agent personality"
});

await session.send("What do you know about me?");
```

---

## 3. LangChain Memory

**Docs:** https://python.langchain.com/docs/how_to/chatbots_memory/

### Memory Types
1. **ConversationBufferMemory** - raw message history
2. **ConversationBufferWindowMemory** - sliding window (last K turns)
3. **ConversationSummaryMemory** - LLM-summarized history
4. **ConversationSummaryBufferMemory** - hybrid (summary + recent)
5. **ConversationKGMemory** - knowledge graph extraction
6. **Entity Memory** - entity-focused extraction

### Architecture
- **Pluggable:** Swap memory types via config
- **Chain-based:** Memory injected into prompt template
- **LangGraph integration:** Stateful workflows

### Strengths
- Flexible, modular design
- Multiple memory strategies
- Good for prototyping
- Integrates with LangChain ecosystem

### Limitations
- **No native vector DB** - need external retriever
- **No temporal reasoning** - no time-aware retrieval
- **No causal reasoning** - KG memory is co-occurrence only
- **Contradictions:** No built-in resolution
- **Multi-hop:** Limited by retriever quality
- **Production gaps:** Summary memory loses detail, buffer memory hits context limits

### Key Insight
LangChain moved away from built-in memory toward **LangGraph** for stateful agents. Memory is now a "middleware" concern, not a first-class feature.

---

## 4. LlamaIndex Memory

**Docs:** https://docs.llamaindex.ai/

### Approach
- **ChatMemoryBuffer** - message buffer with token limit
- **ChatSummaryMemoryBuffer** - summary + buffer hybrid
- **VectorMemory** - vector-based retrieval
- **Composable:** Combine multiple memory types

### Comparison to LangChain
- **More RAG-focused:** Better integration with index/retrieval
- **Less memory variety:** Fewer built-in types
- **Similar limitations:** No causal, limited temporal

### Novel Features
- **Memory routing:** Dynamically choose memory source
- **Persisted memory:** Save/load across sessions

### Limitations
- Same as LangChain: no causal, weak temporal
- Less mature memory ecosystem

---

## 5. Zep

**Website:** https://www.getzep.com/  
**Focus:** Enterprise-grade agent memory

### Architecture
**Context Graph Engine:**
- **Episodes:** Raw conversation chunks
- **Facts:** Extracted from episodes
- **Entities:** People, places, concepts
- **Context Graph:** Relational structure with temporal validity

### Key Features
- **Temporal validity:** Facts have valid-from/valid-to timestamps
- **Contradiction handling:** Old facts invalidated, kept as history
- **Observations:** Patterns surfaced from graph structure
- **Context Lake:** Millions of graphs, governed as one system
- **Sub-200ms retrieval:** At scale (10K-100M graphs)

### Benchmarks
- **LoCoMo:** 94.7% accuracy, 155ms latency, 5,760 tokens
- **LongMemEval:** 90.2% accuracy, 162ms latency, 4,408 tokens

### Strengths
- **Temporal reasoning:** First-class support
- **Contradiction resolution:** Automatic invalidation
- **Enterprise features:** SOC 2, HIPAA, audit logs
- **Provenance:** Every fact traces to source episode
- **Scalable:** Sub-200ms at millions of graphs

### Limitations
- **No causal reasoning** - relational, not causal
- **Co-occurrence graph** - not causal graph
- **Closed source** - cloud-only (no self-hosted)
- **Expensive** - enterprise pricing

### API Model
```python
response = client.thread.add_messages(
    thread_id=thread_id,
    messages=[...],
    return_context=True
)

user_context = client.thread.get_user_context(thread_id)
```

### Key Insight
Zep is closest to what you want, but uses **relational graphs** (co-occurrence), not **causal graphs**. This is the gap.

---

## 6. LongMemEval Benchmark

**GitHub:** https://github.com/xiaowu0162/LongMemEval (1k stars)  
**Paper:** arXiv:2410.10813 (ICLR 2025)

### What It Measures
500 high-quality questions testing 5 core abilities:
1. **Information Extraction** - recall facts
2. **Multi-Session Reasoning** - integrate across sessions
3. **Knowledge Updates** - handle changing facts
4. **Temporal Reasoning** - time-based queries
5. **Abstention** - know when to say "I don't know"

### Dataset
- **LongMemEval_S:** ~115K tokens (~40 sessions)
- **LongMemEval_M:** ~500 sessions
- **Oracle:** Only evidence sessions included

### Key Findings
- **Current systems struggle** with temporal reasoning
- **Multi-hop reasoning** is weak across all systems
- **Abstention** is particularly hard (knowing what you don't know)
- **Retrieval quality** is the bottleneck, not generation

### Relevance to Your Project
- **Temporal reasoning** is a core capability you need
- **Multi-session reasoning** requires causal chains
- **Knowledge updates** need contradiction handling

---

## 7. LongMemEval-V2

**GitHub:** https://github.com/xiaowu0162/LongMemEval-V2 (126 stars)  
**Paper:** arXiv:2605.12493 (May 2026)

### What's New
**Focus:** Long-term memory for **agentic contexts** (not just chat)

### Key Changes
- **Multimodal:** Web agent trajectories with screenshots
- **5 memory abilities:**
  1. Static state recall
  2. Dynamic state tracking
  3. Workflow knowledge
  4. Environment gotchas
  5. Premise awareness
- **Scale:** Up to 500 trajectories, 115M tokens
- **Domains:** Web + Enterprise

### Memory Modules Tested
- `no_retrieval` - baseline
- `rag_query_to_slice` - RAG to raw state
- `rag_query_to_slice_notes` - RAG + notes
- `agentrunbook_r` - structured runbooks
- `codex` - coding agent baseline
- `agentrunbook_c` - collaborative runbooks

### Leaderboard Metric
**LAFS (Latency-Accuracy Frontier Score):** Measures accuracy vs. query latency tradeoff

### Relevance
- **Agentic memory** is the next frontier
- **Multimodal** matters for real agents
- **Workflow knowledge** requires causal understanding

---

## 8. BEAM Benchmark

**GitHub:** https://github.com/mohammadtavakoli78/BEAM (121 stars)  
**Paper:** arXiv:2510.27246 (ICLR 2026)

### What It Measures
**Beyond a Million Tokens:** Tests long-term memory at scale

### Dataset
- **100 conversations** across multiple scales:
  - 128K tokens (20 chats)
  - 500K tokens (35 chats)
  - 1M tokens (35 chats)
  - 10M tokens (10 chats)
- **2,000 validated questions**
- **10 memory abilities:**
  1. Abstention
  2. Contradiction Resolution
  3. Event Ordering
  4. Information Extraction
  5. Instruction Following
  6. Knowledge Update
  7. Multi-Session Reasoning
  8. Preference Following
  9. Summarization
  10. Temporal Reasoning

### LIGHT Framework
**Proposed solution:**
- **Episodic Memory** - long-term retrieval
- **Working Memory** - recent turns buffer
- **Scratchpad** - compressed semantic layer

### Results
- LIGHT: 3.5%-12.7% improvement over baselines
- Even 1M context models struggle as dialogues lengthen
- **Contradiction resolution** and **temporal reasoning** are hardest

### Comparison Table

| Benchmark | Domain | Chat Length | IE | MR | KU | TR | ABS | CR | EO | IF | PF | SUM |
|-----------|--------|-------------|----|----|----|----|-----|----|----|----|----|----|
| LongMemEval | Personal | 115K-1M | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| BEAM | Multi-domain | 128K-10M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Relevance
- **Contradiction resolution** is explicitly tested
- **Event ordering** requires causal understanding
- **10M tokens** is realistic scale for long-running agents

---

## 9. Causal Memory Research

### Key Papers Found

#### 1. **Causal-AgentIR** (arXiv:2607.21125, July 2026)
**Title:** "Causal-AgentIR: Self-Evolving Causal Memory for Adaptive Image Restoration Agents"

**Key Idea:** Hierarchical multi-agent framework with **causal memory graph**
- Stores: degradation patterns, regions, tools, actions, quality changes, preferences
- Supports: graph-based retrieval + multi-hop causal reasoning
- Agents infer: how operations affect quality under different conditions

**Relevance:** Direct precedent for causal memory in agents

#### 2. **REMI** (arXiv:2509.06269, September 2025)
**Title:** "REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents"

**Key Idea:** **Causal Schema Memory** for personalized agents
- **Personal causal knowledge graph** of user's life events/habits
- **Causal reasoning engine** with goal-directed traversals
- **Schema-based planning** with adaptable plan schemas
- LLM orchestrates components

**Metrics Introduced:**
- Personalization Salience Score
- Causal Reasoning Accuracy

**Relevance:** Direct precedent for causal memory + personalization

#### 3. **ADAM** (arXiv:2410.22194, October 2024)
**Title:** "ADAM: An Embodied Causal Agent in Open-World Environments"

**Key Idea:** Embodied agent with **ever-growing causal graph**
- Constructs causal graph from scratch (no prior knowledge)
- Uses causal graph for task decomposition
- Strong interpretability + generalization

**Relevance:** Causal graph for lifelong learning

#### 4. **GRAVITY** (arXiv:2605.01688, May 2026)
**Title:** "GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory"

**Key Idea:** Plug-and-play structured memory module
- **Entity profiles** grounded in relational graphs
- **Temporal event tuples** linked into **causal traces**
- **Cross-session topic summaries**
- Injects structured context into prompt

**Results:** 7.5-10.1% improvement across 5 memory systems on LongMemEval + LoCoMo

**Relevance:** Causal traces for temporal reasoning

#### 5. **ProPlay** (arXiv:2606.12780, June 2026)
**Title:** "ProPlay: Procedural World Models for Self-Evolving LLM Agents"

**Key Idea:** **Procedure graph** capturing causal transitions
- Abstracts trajectories into procedures
- Captures causal transitions among task stages
- Reliability records for each transition
- Preplay: rehearse future paths using learned world knowledge

**Relevance:** Causal graph for planning + learning

---

## 10. Comparison Table

| System | Architecture | Temporal | Contradictions | Multi-hop | Causal | Open Source |
|--------|--------------|----------|----------------|-----------|--------|-------------|
| **Mem0** | Vector + SQL + Entity | ✓ (v3) | ✗ (ADD-only) | ✗ | ✗ | ✓ |
| **MemGPT/Letta** | Hierarchical (OS-inspired) | ✗ | Manual | ✗ | ✗ | ✓ |
| **LangChain** | Pluggable (Buffer/Summary/KG) | ✗ | ✗ | ✗ | ✗ | ✓ |
| **LlamaIndex** | Vector + Buffer | ✗ | ✗ | ✗ | ✗ | ✓ |
| **Zep** | Context Graph (relational) | ✓ (validity) | ✓ (invalidate) | ✓ | ✗ | ✗ |
| **HydraDB (proposed)** | Causal Graph (SlateDB) | ✓ | ✓ (causal) | ✓ | ✓ | ✓ |

---

## 11. Key Gaps in Current Approaches

### 1. **No Causal Reasoning**
- All systems store **facts** or **co-occurrence**, not **cause-effect**
- Cannot answer: "What happens if I do X?" or "Why did Y happen?"
- **Opportunity:** HydraDB's causal discovery fills this gap

### 2. **Weak Temporal Reasoning**
- Mem0 v3 added temporal scoring, but not causal temporal
- Zep has validity timestamps, but not causal chains
- **Opportunity:** Causal graphs naturally encode temporal causality

### 3. **Poor Contradiction Handling**
- Mem0: Stores both, relies on temporal scoring
- MemGPT: Manual core memory updates
- Zep: Invalidates old facts (best approach)
- **Opportunity:** Causal graphs can model "A caused B, but C intervened, so D instead"

### 4. **Limited Multi-hop Reasoning**
- Vector similarity is 1-hop
- Entity graphs are N-hop but lack causal structure
- **Opportunity:** Causal graph traversal enables meaningful multi-hop

### 5. **No Procedural Knowledge**
- All systems store **declarative** facts
- None store **procedural** knowledge (how to do things)
- **Opportunity:** Causal graphs can encode procedures as causal chains

### 6. **Scalability vs. Structure Tradeoff**
- Vector DBs scale but lack structure
- Graph DBs have structure but scale poorly
- **Opportunity:** HydraDB's SlateDB backend scales to millions of nodes

---

## 12. Opportunities for Causal Graph Approach

### 1. **Causal Discovery from Conversations**
- Extract cause-effect relationships from dialogue
- Build causal graph incrementally
- Example: "User said X → Agent did Y → User was happy"

### 2. **Counterfactual Reasoning**
- "What if the user had said Z instead?"
- Traverse causal graph with hypothetical interventions
- Unique capability no other system offers

### 3. **Temporal Causal Chains**
- Not just "A happened before B" but "A caused B"
- Enables: "Why did this happen?" queries
- Supports: "What will happen if...?" predictions

### 4. **Procedural Memory**
- Store workflows as causal chains
- Retrieve: "How do I accomplish X?" → traverse causal graph
- Adapt: "How has this procedure changed?" → compare causal chains

### 5. **Contradiction Resolution via Causality**
- Not just "A is true, now B is true"
- But: "A caused B, but C intervened, so D instead"
- Preserves causal history, not just latest fact

### 6. **Explainability**
- Causal graphs are inherently interpretable
- Show user: "I recommend X because A → B → C"
- Audit trail of reasoning

### 7. **HydraDB Advantages**
- **SlateDB backend:** Scales to millions of nodes/edges
- **Graph-native:** Built for graph workloads
- **OpenCypher:** Familiar query language
- **Multi-modal:** Can store embeddings + causal structure

---

## 13. Architecture Proposal

### Core Components

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Interface                        │
├──────────────────────────────────────────────────────────┤
│  Memory Operations                                        │
│  - add(messages) → extract causal facts                  │
│  - search(query) → causal retrieval                      │
│  - why(question) → causal explanation                    │
│  - what_if(hypothesis) → counterfactual reasoning        │
├──────────────────────────────────────────────────────────┤
│  Causal Discovery Layer                                   │
│  - LLM extracts cause-effect from conversations          │
│  - Temporal ordering                                     │
│  - Intervention detection                                │
├──────────────────────────────────────────────────────────┤
│  Causal Graph Store (HydraDB)                             │
│  - Nodes: Entities, Events, States                       │
│  - Edges: Causal relationships with timestamps           │
│  - Properties: Confidence, evidence, temporal validity   │
├──────────────────────────────────────────────────────────┤
│  Retrieval Layer                                          │
│  - Causal traversal (multi-hop)                          │
│  - Temporal filtering                                    │
│  - Counterfactual simulation                             │
└──────────────────────────────────────────────────────────┘
```

### Key Differentiators
1. **Causal discovery** from conversations
2. **Counterfactual reasoning** (what-if queries)
3. **Explainable recommendations** (causal chains)
4. **Procedural memory** (how-to workflows)
5. **Temporal causal chains** (not just temporal facts)

### Integration Model
```python
from hydradb import CausalMemory

memory = CausalMemory()

# Add conversation
memory.add(
    messages=[
        {"role": "user", "content": "I'm allergic to peanuts"},
        {"role": "assistant", "content": "I'll avoid peanut oil"}
    ],
    user_id="alice"
)

# Causal retrieval
results = memory.search("What should I order for Alice?")

# Causal explanation
explanation = memory.why("Why did you recommend this?")
# Returns: "User is allergic to peanuts → Avoid peanut oil → Recommended olive oil"

# Counterfactual
hypothesis = memory.what_if("What if Alice wasn't allergic?")
# Returns: Alternative recommendations
```

---

## 14. Relevant Papers & Reading List

### Must Read
1. **Mem0** - arXiv:2504.19413 (production memory architecture)
2. **MemGPT** - arXiv:2310.08560 (OS-inspired memory)
3. **LongMemEval** - arXiv:2410.10813 (benchmark)
4. **BEAM** - arXiv:2510.27246 (scale benchmark)

### Causal Memory (Directly Relevant)
5. **Causal-AgentIR** - arXiv:2607.21125 (causal memory for agents)
6. **REMI** - arXiv:2509.06269 (causal schema memory)
7. **ADAM** - arXiv:2410.22194 (embodied causal agent)
8. **GRAVITY** - arXiv:2605.01688 (structured memory with causal traces)
9. **ProPlay** - arXiv:2606.12780 (procedural causal graphs)

### Supporting
10. **Zep** - https://www.getzep.com/ (enterprise graph memory)
11. **LongMemEval-V2** - arXiv:2605.12493 (agentic memory)

---

## 15. Key Resources

### GitHub Repos
- Mem0: https://github.com/mem0ai/mem0
- Letta: https://github.com/letta-ai/letta
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LongMemEval-V2: https://github.com/xiaowu0162/LongMemEval-V2
- BEAM: https://github.com/mohammadtavakoli78/BEAM

### Documentation
- Mem0 Docs: https://docs.mem0.ai/
- Letta Docs: https://docs.letta.com/
- LangChain Memory: https://python.langchain.com/docs/how_to/chatbots_memory/
- Zep: https://www.getzep.com/

### Benchmarks
- LongMemEval Leaderboard: https://xiaowu0162.github.io/long-mem-eval/
- LongMemEval-V2 Leaderboard: https://xiaowu0162.github.io/longmemeval-v2/

---

## 16. Conclusion

**The Opportunity:**
No existing system combines **causal discovery** with **graph-based storage** for agent memory. Zep comes closest with temporal graphs, but uses co-occurrence, not causation.

**Your Advantage:**
HydraDB's SlateDB backend provides the scalability. Adding causal discovery on top creates a unique capability:
- **Causal reasoning** (why did this happen?)
- **Counterfactual reasoning** (what if...?)
- **Explainable recommendations** (causal chains)
- **Procedural memory** (how to do things)

**Hackathon Strategy:**
1. **Start simple:** Extract causal triples from conversations
2. **Store in HydraDB:** Nodes = entities/events, Edges = causal relations
3. **Implement basic retrieval:** Causal traversal + temporal filtering
4. **Demo counterfactual:** "What if" queries (unique differentiator)
5. **Benchmark:** Run on LongMemEval temporal reasoning subset

**Key Differentiator:**
While others store **what happened**, you store **why it happened** and **what would happen if...**. This is the causal memory layer.
