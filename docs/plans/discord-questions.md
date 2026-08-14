# Discord Questions for HydraDB Founder

## Strategy
- Ask genuine technical questions that show you're building
- Space them out (1-2 per day)
- Follow up on answers to show engagement
- Be specific about your use case (causal memory layer)

---

## Day 1 Questions (Aug 13)

### Question 1: Edge Properties
```
Hey! I'm building a causal memory layer for AI agents using HydraDB for the hackathon.

Quick question: Does HydraDB support edge properties like `confidence`, `timestamp`, `mechanism` on causal edges? I want to store metadata like:

(causes)-[:CAUSES {confidence: 0.85, timestamp: 1700000000, mechanism: "relocation"}]->(effect)

Is this the right approach, or should I model causal metadata differently?
```

### Question 2: Path Query Performance
```
Follow-up question: I'm using `algo.SPpaths` to trace causal chains. 

At what scale should I start worrying about performance? Are there any optimizations for large causal graphs (10k+ events)?

Also, does GraphBLAS traversal help with causal path queries, or is it mainly for BFS-style traversals?
```

---

## Day 2 Questions (Aug 14)

### Question 3: Temporal Queries
```
Question about temporal queries: I'm storing timestamps on edges for causal ordering. 

Is there a way to query "all causal effects that happened within 24 hours of event X"? 

Something like:
MATCH path = (e1)-[:CAUSES*]->(e2)
WHERE duration.between(e1.timestamp, e2.timestamp) < "24h"

Or do I need to handle temporal filtering in my application layer?
```

### Question 4: Bookmark-Based Temporal Reads
```
I saw HydraDB supports bookmark-based causal consistency. 

For my causal memory layer, I want to query "what did the user believe last week?" by reading at a specific point in the graph's history.

Can I use bookmarks for this? Or are bookmarks only for causal read consistency (like "read after this write")?
```

---

## Day 3 Questions (Aug 15)

### Question 5: Contradiction Handling
```
Working on contradiction resolution. When a fact gets overwritten (e.g., "moved to SF" overwrites "lives in NYC"), I'm creating an OVERWRITES edge.

For queries like "Where does the user live?", should I:
1. Only follow non-overwritten facts?
2. Follow the most recent fact in the causal chain?
3. Store a "current" flag on vertices?

What's the idiomatic way to handle this in a graph DB?
```

### Question 6: Multi-Hop Causal Queries
```
I'm building "why" queries that trace causal chains backward. 

For example: "Why does the user love their neighborhood?" 
→ loves neighborhood ← found apartment ← moved to SF

Is there a way to do reverse path traversal in HydraDB? Something like:
CALL algo.SPpaths({
  sourceNode: $target_id,
  targetNode: $source_id,
  relDirection: 'incoming'
})

Or should I just reverse the logic in my application?
```

---

## Day 4 Questions (Aug 16)

### Question 7: LLM Integration Patterns
```
I'm integrating LLMs to extract causal triples from conversations and interpret causal query results.

Any recommendations for patterns? I'm thinking:
1. LLM extracts (event1)-[:CAUSES]->(event2) from text
2. Store in HydraDB
3. Query causal paths
4. LLM interprets results as natural language

Is there a better way to structure this? Any gotchas I should watch out for?
```

### Question 8: Confidence Scores
```
I'm storing confidence scores on causal edges (0.0-1.0). 

When querying causal paths, should I:
1. Filter by minimum confidence threshold?
2. Weight paths by average confidence?
3. Return all paths and let the application decide?

What's the best practice for handling uncertain causal relationships?
```

---

## Day 5 Questions (Aug 17)

### Question 9: Graph Visualization
```
For my demo, I want to visualize the causal graph. 

Does HydraDB have any built-in visualization tools? Or should I export to a format like GraphML/DOT and use external tools?

Any recommendations for visualizing large causal graphs without making them unreadable?
```

### Question 10: Scaling to Production
```
My causal memory layer works great for demos, but I'm thinking about production scale.

If I have 1000 users, each with 1000 events and 5000 causal edges, that's:
- 1M events
- 5M causal edges

Any recommendations for partitioning/sharding? Should I use separate cells per user, or is there a better approach?
```

---

## Follow-Up Templates

### If they answer your question:
```
Thanks! That makes sense. I'll implement it that way.

[Optional: Share what you built based on their answer]
```

### If they ask what you're building:
```
I'm building a causal memory layer for AI agents. Instead of just storing facts like vector DBs, I store cause→effect relationships so agents can answer "why" questions.

For example: "Why does the user prefer dark mode?" → traces back to conversation where they explained eye strain.

Built on HydraDB because the path traversal primitives (algo.SPpaths, algo.SSpaths) are perfect for causal chain queries.
```

### If they offer to help:
```
That would be amazing! I'm currently working on [specific feature]. 

Would you be open to a quick chat this week? Or I can share my repo and you can take a look when you have time.

Either way, really appreciate the help!
```

---

## Engagement Tips

1. **Be specific** - Don't ask "how does HydraDB work?" Ask "how does X work for Y use case?"
2. **Show progress** - Share what you've built, not just what you're planning
3. **Be responsive** - Reply to their answers quickly
4. **Be grateful** - Thank them for their time
5. **Be genuine** - Don't over-promote, focus on learning

---

## What NOT to Ask

- Questions easily answered by reading the docs
- Questions about features that don't exist (yet)
- Questions that show you haven't tried to solve it yourself
- Questions that are too broad or vague

---

## Tracking

Keep track of:
- Questions asked (date, question, answer)
- Follow-ups needed
- Insights gained
- Features implemented based on their advice

This shows you're actually building and listening to feedback.
