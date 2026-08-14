# Hack Hydra Content Calendar
## Aug 13-20, 2026

**Strategy:**
- **X (Twitter):** Technical posts, build in public, show progress
- **Discord:** Genuine questions, show you're building something real
- **Narrative arc:** Announcement → Progress → Technical depth → Demo

---

## Day 1 — Aug 13 (Wednesday)

### X Post 1 (Morning)
```
Building a causal memory layer for AI agents using @hydradb for #HackHydra

The problem: Vector DBs find similar memories but can't answer WHY.
"My agent remembers I moved to SF, but can't trace WHY I love my neighborhood."

Causal graphs fix this. Store cause→effect relationships, not just facts.

Building in public → [link to repo when ready]

#AgentMemory #CausalAI #GraphDB
```

### Discord Message 1 (After X post, ~2 hours later)
```
Hey! I'm building a causal memory layer for agents using HydraDB for the hackathon.

Quick question: Does HydraDB support edge properties like `confidence`, `timestamp`, `mechanism` on causal edges? I want to store metadata like:

(causes)-[:CAUSES {confidence: 0.85, timestamp: 1700000000, mechanism: "relocation"}]->(effect)

Is this the right approach, or should I model causal metadata differently?
```

### X Post 2 (Evening)
```
Day 1 of building causal memory on @hydradb

Set up the graph schema:
- Events as vertices (with timestamps, session IDs)
- Causal edges with confidence scores
- Overwrite edges for fact updates
- Conflict edges for contradictions

Next: implementing causal path traversal to answer "why" questions

[optional: screenshot of schema diagram]
```

---

## Day 2 — Aug 14 (Thursday)

### X Post 1 (Morning)
```
Got causal path queries working on @hydradb

Using `algo.SPpaths` to trace causal chains:

"Why does the user love their neighborhood?"
→ moved to SF → found apartment → loves neighborhood

This is impossible with vector search. Causal graphs store WHY, not just WHAT.

#HackHydra #AgentMemory
```

### Discord Question 1
```
Question about temporal queries: I'm storing timestamps on edges for causal ordering. 

Is there a way to query "all causal effects that happened within 24 hours of event X"? 

Something like:
MATCH path = (e1)-[:CAUSES*]->(e2)
WHERE duration.between(e1.timestamp, e2.timestamp) < "24h"

Or do I need to handle temporal filtering in my application layer?
```

### X Post 2 (Evening)
```
Day 2 update: Built the Python SDK wrapper for HydraDB

```python
memory = CausalMemory()
memory.add(messages, session_id="s15")
memory.why("Why does user prefer dark mode?")
# → "switched last week because eye strain during late-night coding"
```

The "why" query traces back through causal edges. Vector DBs can't do this.

#HackHydra #BuildInPublic
```

---

## Day 3 — Aug 15 (Friday)

### X Post 1 (Morning)
```
Causal memory can handle contradictions now

User says "I live in NYC" (session 3)
User says "I moved to SF" (session 15)

Vector DB: confused, returns both
Causal graph: knows "moved" OVERWRITES "lives in NYC"

Query "Where does user live?" → "SF" (not "NYC")

#HackHydra #AgentMemory
```

### Discord Question 2
```
Working on contradiction resolution. When a fact gets overwritten (e.g., "moved to SF" overwrites "lives in NYC"), I'm creating an OVERWRITES edge.

For queries like "Where does the user live?", should I:
1. Only follow non-overwritten facts?
2. Follow the most recent fact in the causal chain?
3. Store a "current" flag on vertices?

What's the idiomatic way to handle this in a graph DB?
```

### X Post 2 (Evening)
```
Day 3: Contradiction handling works

When facts change over time, causal graphs know which fact superseded which.

"I moved to SF" → OVERWRITES → "I live in NYC"

Query "Where does user live?" → follows causal chain → "SF"

Vector DBs would return both. Causal graphs know the truth.

#HackHydra #CausalAI
```

---

## Day 4 — Aug 16 (Saturday)

### X Post 1 (Morning)
```
Weekend hackathon update: LLM integration working

The pipeline:
1. User talks to agent
2. LLM extracts causal triples: (event1)-[:CAUSES]->(event2)
3. Store in HydraDB
4. Query causal paths for "why" questions

LLMs are surprisingly good at causal extraction. GPT-4 hits 97% on pairwise causal discovery (paper: arXiv:2305.00050)

#HackHydra #AgentMemory
```

### Discord Question 3
```
Question about performance: I'm storing ~1000 events with causal edges. Path queries (`algo.SPpaths`) are fast (<10ms).

At what scale should I start worrying about performance? Are there any optimizations for large causal graphs (10k+ events)?

Also, does GraphBLAS traversal help with causal path queries, or is it mainly for BFS-style traversals?
```

### X Post 2 (Evening)
```
Day 4: Built the demo app

Streamlit UI showing:
- Chat interface (add memories)
- Causal graph visualization
- "Why" query interface
- Temporal reasoning demo

The killer feature: Ask the agent WHY you prefer something, and it traces back to the original conversation.

Demo video coming tomorrow.

#HackHydra #BuildInPublic
```

---

## Day 5 — Aug 17 (Sunday)

### X Post 1 (Morning)
```
Day 5: Demo video recorded

3-minute walkthrough:
1. Problem: LLMs forget everything (30s)
2. Solution: Causal memory layer (30s)
3. Live demo: Add memories, query "why" (90s)
4. Technical insight: Causal > vector (30s)
5. Impact: Memory layer for agent era (30s)

Polishing README and docs now.

#HackHydra #AgentMemory
```

### Discord Question 4
```
Final question before submission: What's the best way to explain the causal advantage to judges who might not be familiar with causal inference?

I'm thinking: "Vector search finds similar memories. Causal graphs find RELATED memories — the decision that led to the preference, the context that explains it."

Is there a better way to frame this? Any analogies that work well?
```

### X Post 2 (Evening)
```
Day 6: Final polish

README done:
- Quickstart (< 5 min)
- Architecture diagram
- API reference
- Examples

Code complete:
- Python SDK (500 LOC)
- Causal reasoning engine
- LLM integration
- Streamlit demo

Submitting tomorrow.

#HackHydra #BuildInPublic
```

---

## Day 6 — Aug 18 (Monday)

### X Post 1 (Morning)
```
Submitted to #HackHydra Track 03: Agent Memory + Context Retrieval

Project: Causal Memory Layer on HydraDB

The problem: Vector DBs can't answer WHY.
The solution: Causal graphs store cause→effect relationships.
The result: Agents that remember not just WHAT, but WHY.

Repo: [link]
Demo: [link]

Wish me luck! 🤞
```

### Discord Message (After submission)
```
Just submitted my project for Track 03!

**Causal Memory Layer on HydraDB**

Built a Python SDK that stores causal relationships (not just facts) in HydraDB. Agents can now answer "why" questions by tracing causal chains.

Demo shows: "Why does the user prefer dark mode?" → traces back to conversation where they explained eye strain.

Thanks for the great docs and quick answers on Discord! Really helped me get unblocked on temporal queries.

[repo link]
[demo link]
```

---

## Day 7 — Aug 19 (Tuesday)

### X Post (Optional - if you want to stay active)
```
Post-submission thoughts on #HackHydra

What I learned building causal memory:
1. Causal graphs are natural for agent memory (not just a nice-to-have)
2. HydraDB's path procedures are perfect for causal traversal
3. The "why" query is the killer feature (impossible with vector DBs)
4. LLMs + causal discovery = powerful combo

Thanks @hydradb for running this! Learned a ton.

#AgentMemory #CausalAI
```

### Discord (Optional - thank you message)
```
Quick thanks to the HydraDB team for running Hack Hydra!

Built a causal memory layer and learned a ton about graph databases in the process. The path procedures (`algo.SPpaths`, `algo.SSpaths`) turned out to be perfect for causal path traversal.

Looking forward to seeing what everyone else built!

[optional: link to your repo/demo]
```

---

## Day 8 — Aug 20 (Wednesday - Submission Deadline)

### X Post (Reminder)
```
Reminder: #HackHydra submissions due tonight at 11:59 PM PT!

If you're building something with @hydradb, make sure to submit:
✅ Google Form
✅ Public GitHub repo
✅ 3-min demo video
✅ Open-source license

Good luck everyone! 🚀
```

### Discord (Optional - final engagement)
```
Anyone else doing last-minute polish? I'm finalizing my demo video and README.

What's everyone else building? Curious to see the different approaches!
```

---

## Engagement Strategy

### Discord Best Practices
1. **Ask genuine questions** — shows you're actually building
2. **Share specific technical details** — not just "I'm building something cool"
3. **Thank people who answer** — build relationships
4. **Don't over-promote** — 1 announcement post, rest should be questions/engagement
5. **Be helpful** — if you see someone struggling with something you solved, help them

### X Best Practices
1. **Build in public** — share progress, not just announcements
2. **Technical depth** — show you understand the problem
3. **Use hashtags** — #HackHydra #AgentMemory #CausalAI #GraphDB
4. **Tag @hydradb** — they'll likely retweet
5. **Visuals help** — diagrams, screenshots, code snippets
6. **Thread long posts** — break into threads for readability

### Content Themes
- **Day 1-2:** Setup + basic functionality
- **Day 3-4:** Core features (contradictions, temporal reasoning)
- **Day 5-6:** Polish + demo
- **Day 7-8:** Submission + engagement

### Engagement Metrics to Track
- Discord: Questions asked, answers received, helpful interactions
- X: Likes, retweets, replies, link clicks
- GitHub: Stars, forks, issues (if people engage with your repo)

---

## Backup Content (If You Fall Behind)

### X Posts
- "Day X update: [specific technical detail]"
- "Learned today: [insight about causal graphs / HydraDB]"
- "Challenge: [problem you're solving]. Solution: [approach]"

### Discord Questions
- "How do I [specific technical question]?"
- "Is there a better way to [approach]?"
- "Has anyone tried [approach]? What worked?"
- "What's the recommended way to [task]?"

---

## Final Checklist

### Before Submission (Aug 20)
- [ ] Google Form filled out
- [ ] Public GitHub repo with open-source license
- [ ] README with quickstart, architecture, examples
- [ ] 3-minute demo video (unlisted YouTube is fine)
- [ ] Demo link works (if hosted)
- [ ] Team members listed correctly
- [ ] Submitted before 11:59 PM PT

### Content Calendar
- [ ] Day 1: X announcement + Discord question
- [ ] Day 2-5: Daily X posts + Discord questions
- [ ] Day 6: Submission announcement
- [ ] Day 7-8: Optional engagement posts

---

**Remember:** The goal is to show you're genuinely building something interesting, not just spamming promotions. Genuine questions on Discord > promotional posts.
