# Causal Discovery & Causal Reasoning on Knowledge Graphs
## Research Report for HydraDB Agent Memory Integration

**Date:** 2026-08-13
**Purpose:** Inform technical architecture for combining causal discovery with graph databases for agent memory

---

## 1. Causal Discovery Fundamentals

### What is Causal Discovery?

Causal discovery (also called exploratory causal analysis or data causality) uses statistical algorithms to infer potentially causal associations from observational data. Unlike correlation, which only identifies co-occurrence patterns, causal discovery attempts to identify **directional cause-effect relationships** under strict assumptions.

**Key distinction:** Correlation says "X and Y move together." Causation says "X *makes* Y change."

### Causal Discovery vs. Causal Inference

| Aspect      | Causal Discovery                                            | Causal Inference                              |
| ----------- | ----------------------------------------------------------- | --------------------------------------------- |
| Goal        | Learn causal graph structure from data                      | Estimate causal effects given a known graph   |
| Input       | Observational data                                          | Data + causal graph (DAG)                     |
| Output      | DAG / CPDAG / PAG                                           | Treatment effects, counterfactuals            |
| Assumptions | Faithfulness, causal Markov, no hidden confounders (varies) | Graph correctness, identifiability conditions |

### Key Algorithms

#### Constraint-Based Methods

**PC Algorithm** (Spirtes, Glymour, Scheines 1990/2000)
- Starts with fully connected undirected graph
- Tests conditional independence between all variable pairs
- Removes edges when variables are conditionally independent given some subset
- Orient edges using v-structure rules (collider detection) and Meek rules
- **Output:** CPDAG (Completed Partially Directed Acyclic Graph) - Markov equivalence class
- **Assumptions:** Causal sufficiency (no hidden confounders), faithfulness, causal Markov condition
- **Complexity:** O(p^d) independence tests where p = variables, d = max degree
- **Variants:** PC-stable (order-independent), Conservative PC, Parallel PC

**FCI Algorithm** (Fast Causal Inference)
- Extension of PC that handles **latent confounders**
- Outputs PAG (Partial Ancestral Graph) with additional edge marks
- Can distinguish between direct causation, latent confounding, and selection bias
- More conservative but more realistic for observational data
- **Key feature:** Can detect presence of hidden common causes

#### Score-Based Methods

**GES (Greedy Equivalence Search)** (Chickering 2002)
- Searches over Markov equivalence classes of DAGs
- Two phases: forward (add edges to improve score), backward (remove edges)
- Uses scoring functions like BIC, BDeu
- **Advantage:** Globally optimal under certain conditions
- **Complexity:** More scalable than exhaustive search

**NOTEARS** (Zheng et al. 2018, 2020)
- **Breakthrough:** Reformulates discrete DAG search as **continuous optimization**
- Uses algebraic characterization: h(W) = tr(e^{W∘W}) - d = 0 iff W is a DAG
- Converts combinatorial constraint to differentiable equality constraint
- Solves: min L(W; X) s.t. h(W) = 0 using augmented Lagrangian
- **Extensions:** Nonlinear NOTEARS (1909.13189), NOTEARS-MLP for nonlinear relationships
- **Advantage:** Scales to hundreds of variables, differentiable end-to-end
- **Limitation:** Assumes acyclicity, sensitive to hyperparameters

#### Functional Causal Model-Based Methods

**LiNGAM** (Linear Non-Gaussian Acyclic Model) (Shimizu et al. 2006)
- Assumes linear relationships + non-Gaussian noise
- Uses ICA (Independent Component Analysis) to identify causal order
- **Key insight:** Non-Gaussianity enables full identification (not just Markov equivalence class)
- **Variants:** DirectLiNGAM (faster), RCD (handles latent confounders), VAR-LiNGAM (time series)
- **Assumption:** Non-Gaussian error terms (violated if Gaussian)

**ANM (Additive Noise Models)**
- Assumes X → Y means Y = f(X) + noise where noise ⊥ X
- Can determine direction from observational data alone
- More general than LiNGAM (allows nonlinear f)
- Requires specific functional form assumptions

#### Time-Series Methods

**Granger Causality** (Granger 1969)
- X "Granger-causes" Y if past X helps predict Y beyond past Y alone
- Implemented via VAR (Vector Autoregression) models
- **Test:** F-test or likelihood ratio test on lagged coefficients
- **Limitation:** Only captures linear/predictive causality, not true causation
- **Extension:** Conditional Granger causality (controls for other variables)

**PCMCI** (Runge et al. 2019)
- Combines PC algorithm with time-series structure
- Handles autocorrelation and high-dimensional time series
- Used extensively in climate science, neuroscience

### Comparison Table

| Algorithm | Handles Latent Confounders | Nonlinear | Time Series | Scalability | Identifiability |
|-----------|---------------------------|-----------|-------------|-------------|-----------------|
| PC | No | With tests | No | Medium | CPDAG |
| FCI | Yes | With tests | No | Medium | PAG |
| GES | No | With score | No | High | CPDAG |
| NOTEARS | No | Yes (extension) | No | High | DAG (with assumptions) |
| LiNGAM | Extension | No (linear) | VAR-LiNGAM | High | Full DAG |
| Granger | No | No (linear) | Yes | High | Predictive only |
| PCMCI | Yes | With tests | Yes | High | Time-lagged CPDAG |

---

## 2. Causal Discovery Libraries

### 2.1 causal-learn (Python)

**Repository:** https://github.com/py-why/causal-learn
**Stars:** 1.7k | **License:** MIT
**Paper:** Zheng et al. JMLR 2024

**Algorithms Supported:**
- **Constraint-based:** PC, PC-stable, FCI, FCI+, CCD (Cyclic Causal Discovery)
- **Score-based:** GES, BIC scoring, BDeu scoring
- **Functional model:** LiNGAM (DirectLiNGAM, RCD, VAR-LiNGAM), ANM
- **Hidden causal representation:** ICA-based methods
- **Permutation-based:** methods leveraging variable ordering
- **Granger causality:** time-series causal discovery
- **Continuous optimization:** NOTEARS, DAG-GNN

**Key Features:**
- Unified API across all methods
- Independence tests: Fisher-z, Chi-square, KCI (kernel), mutual information
- Score functions: BIC, BDeu, K2
- Graph operations and evaluation metrics (SHD, F1, precision, recall)
- Benchmark datasets maintained by CMU CLeaR group
- Integration with Tetrad via py-tetrad for Java algorithms

**Use Case:** Best for research, benchmarking, and when you need multiple algorithms to compare.

### 2.2 DoWhy (Microsoft)

**Repository:** https://github.com/py-why/dowhy
**Stars:** 8.3k | **License:** MIT
**Paper:** Sharma & Kiciman 2020, Blöbaum et al. JMLR 2024

**Core Philosophy:** Unified language for causal inference combining graphical models + potential outcomes.

**Key Features:**
1. **Effect Estimation:** Identification via do-calculus, estimation via backdoor/frontdoor/IV
2. **Graphical Causal Models (GCM):** Full SCM (Structural Causal Model) framework
3. **Root Cause Analysis:** Attribute anomalies to their root causes in causal graph
4. **What-if Analysis:** Interventional sampling, counterfactual estimation
5. **Refutation API:** Test robustness of causal estimates (random common cause, placebo test, data subset)

**Algorithms:**
- Identification: ID algorithm, do-calculus rules
- Estimation: Propensity score matching, IPW, double ML, causal forests, instrumental variables
- GCM: Auto-assign causal mechanisms, anomaly attribution, distribution change analysis

**Example (Effect Estimation):**
```python
from dowhy import CausalModel
model = CausalModel(data=df, treatment="X", outcome="Y", graph="dag.dot")
estimand = model.identify_effect()
estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")
refute = model.refute_estimate(estimand, estimate, method_name="random_common_cause")
```

**Example (Root Cause Analysis):**
```python
from dowhy import gcm
causal_model = gcm.StructuralCausalModel(nx.DiGraph([('X', 'Y'), ('Y', 'Z')]))
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
anomaly_attribution = gcm.attribute_anomalies(causal_model, "Z", anomalous_sample)
```

**Use Case:** Production causal inference, robustness checking, root cause analysis. Best for when you have a known causal graph and want to estimate effects.

### 2.3 pgmpy

**Repository:** https://github.com/pgmpy/pgmpy
**License:** MIT
**Docs:** https://pgmpy.org/

**Key Features:**
- **Causal Discovery / Structure Learning:** PC, Hill Climbing, Tabu Search, BIC/BDeu scoring
- **Parameter Estimation:** MLE, Bayesian estimation for CPDs
- **Probabilistic Inference:** Variable elimination, belief propagation, approximate inference
- **Causal Identification:** Determine if causal query is identifiable from graph
- **Causal Inference:** Compute interventional and counterfactual distributions
- **Simulation:** Generate data from models under various scenarios
- **scikit-learn compatible:** Works in sklearn pipelines

**Use Case:** Probabilistic graphical models + causal inference. Good for Bayesian networks and when you need both structure learning and inference.

### 2.4 CausalNex (QuantumBlack/McKinsey)

**Repository:** https://github.com/quantumblacklabs/causalnex
**License:** Apache 2.0
**Docs:** https://causalnex.readthedocs.io/

**Key Features:**
- **Structure Learning:** NOTEARS algorithm (continuous optimization for DAG learning)
- **Bayesian Networks:** Fit conditional probability distributions
- **Inference Engine:** Query marginals, perform do-interventions
- **Do-Calculus:** Compute interventional distributions
- **Latent Variables:** Add latent confounders, identify candidate locations
- **scikit-learn Interface:** DAGRegressor, DAGClassifier
- **Plotting:** Customizable causal graph visualization

**Why NOTEARS?**
- Handles nonlinear relationships via MLP extension
- Scales better than constraint-based methods for large graphs
- Differentiable - can integrate with deep learning pipelines

**Use Case:** When you want NOTEARS specifically, or need Bayesian network inference with causal semantics. Good for business applications (McKinsey origin).

### 2.5 Other Notable Libraries

**Tetrad** (CMU)
- Java-based, most comprehensive collection of causal discovery algorithms
- GUI + command line + Python/R wrappers (py-tetrad)
- Includes PC, FCI, GES, LiNGAM, GFCI, BOSS, many more
- **Best for:** Researchers who want access to every algorithm

**EconML** (Microsoft)
- Focus on **treatment effect estimation** (CATE)
- Double ML, causal forests, meta-learners (S/T/X/R)
- Integrates with DoWhy for identification
- **Best for:** When you need heterogeneous treatment effects

**CausalML** (Uber)
- Uplift modeling, treatment effect estimation
- Meta-learners, tree-based methods
- **Best for:** Business applications, A/B test analysis

**CDT (Causal Discovery Toolbox)**
- Python, includes PC, FCI, GES, LiNGAM, CGNN
- Causal Generative Neural Networks (CGNN) - deep learning approach
- **Best for:** Deep learning + causal discovery

**PCMCI** (PCMCI package by Runge)
- Specialized for time-series causal discovery
- Used in climate science, neuroscience, epidemiology
- **Best for:** Temporal causal discovery

---

## 3. Causal Reasoning on Knowledge Graphs

### 3.1 Representing Causal Relationships in Knowledge Graphs

**Basic Model:**
```
(Cause)-[:CAUSES {strength: 0.8, evidence: ["study1", "study2"]}]->(Effect)
```

**Extended Model with Metadata:**
```
(Cause)-[:CAUSES {
  strength: 0.8,
  direction: "positive",
  temporal_lag: "2h",
  context: "under condition X",
  evidence: ["RCT-2024", "observational-2023"],
  confidence: 0.9,
  mechanism: "pathway description"
}]->(Effect)
```

**Key Design Decisions:**

1. **Reification vs. Direct Edges**
   - Direct: `(A)-[:CAUSES]->(B)` - simple but limited
   - Reified: `(A)-[:HAS_CAUSAL_EFFECT]->(CausalEffect)-[:AFFECTS]->(B)` - allows attaching rich metadata to the causal relationship itself

2. **Causal vs. Associative Edges**
   - Separate edge types: `CAUSES`, `CORRELATES_WITH`, `PRECEDES`
   - Allows querying specifically for causal paths

3. **Temporal Information**
   - Edge properties: `time_lag`, `valid_from`, `valid_until`
   - Separate temporal graph layer: `(Event)-[:OCCURRED_AT]->(TimePoint)`

4. **Counterfactual Representation**
   - Store interventions: `(Intervention)-[:DO_ACTION {value: X=1}]->(Outcome)`
   - Counterfactual worlds as separate graph snapshots

### 3.2 Querying Causal Paths

**Cypher Examples (Neo4j/HydraDB):**

Find all causes of an effect:
```cypher
MATCH (cause)-[:CAUSES]->(effect:Entity {name: "disease_X"})
RETURN cause.name, type(cause)
```

Find causal chains (mediation analysis):
```cypher
MATCH path = (root)-[:CAUSES*1..5]->(outcome:Entity {name: "symptom_Y"})
WHERE ALL(n IN nodes(path) WHERE single(m IN nodes(path) WHERE id(m) = id(n)))
RETURN path, reduce(s = 1.0, r IN relationships(path) | s * r.strength) AS chain_strength
ORDER BY chain_strength DESC
```

Find confounders between two variables:
```cypher
MATCH (confounder)-[:CAUSES]->(x:Entity {name: "X"}),
      (confounder)-[:CAUSES]->(y:Entity {name: "Y"})
WHERE NOT (x)-[:CAUSES]->(y) AND NOT (y)-[:CAUSES]->(x)
RETURN confounder.name AS potential_confounder
```

Find instrumental variables:
```cypher
MATCH (iv:Entity)-[:CAUSES]->(treatment:Entity {name: "T"}),
      (treatment)-[:CAUSES]->(outcome:Entity {name: "O"})
WHERE NOT (iv)-[:CAUSES]->(outcome)
  AND NOT EXISTS((iv)-[:CAUSES]-()-[:CAUSES]->(outcome))
RETURN iv.name AS instrumental_variable
```

Compute total causal effect along paths:
```cypher
MATCH path = (x)-[:CAUSES*]->(y)
WHERE x.name = "X" AND y.name = "Y"
RETURN path,
       length(path) AS path_length,
       reduce(product = 1.0, r IN relationships(path) | product * r.strength) AS multiplicative_effect
```

### 3.3 Handling Temporal Causality

**Temporal Causal Graph Models:**

1. **Time-Expanded Graph**
   - Each time point has its own copy of entities
   - `(X_t=0)-[:CAUSES]->(Y_t=1)-[:CAUSES]->(Z_t=2)`
   - **Pros:** Explicit temporal reasoning
   - **Cons:** Graph size explodes with time resolution

2. **Temporal Edge Properties**
   - `(X)-[:CAUSES {time_lag: "2h", valid_from: 1000, valid_to: 2000}]->(Y)`
   - Query with temporal constraints:
   ```cypher
   MATCH (x)-[r:CAUSES]->(y)
   WHERE r.time_lag < "24h" AND r.valid_from < timestamp() < r.valid_to
   RETURN x, r, y
   ```

3. **Event Graph with Temporal Ordering**
   - `(Event)-[:OCCURRED_AT]->(TimePoint)`
   - `(Event1)-[:PRECEDES]->(Event2)`
   - Causal edges respect temporal ordering: causes must precede effects

4. **Granger Causality in Graph**
   - Store predictive relationships separately: `(X)-[:GRANGER_CAUSES {p_value: 0.01, lag: 5}]->(Y)`
   - Distinguish from interventional causality

### 3.4 Handling Contradictory Causal Claims

**Problem:** Different studies/sources may claim contradictory causal relationships.

**Solutions:**

1. **Evidence-Weighted Edges**
   ```
   (A)-[:CAUSES {strength: 0.6, evidence_count: 5, sources: [...]}]->(B)
   (A)-[:CAUSES {strength: -0.3, evidence_count: 2, sources: [...]}]->(B)
   ```
   - Aggregate by evidence weight: `weighted_avg_strength = Σ(strength × evidence_count) / Σ(evidence_count)`

2. **Context-Dependent Causation**
   ```
   (A)-[:CAUSES {context: "in_population_X", strength: 0.8}]->(B)
   (A)-[:CAUSES {context: "in_population_Y", strength: -0.2}]->(B)
   ```
   - Query with context matching

3. **Temporal Versioning**
   - Causal claims have validity periods
   - New evidence updates/strengthens/weaken edges
   - Maintain history of belief updates

4. **Multi-Layer Graph**
   - Layer 1: Observed associations
   - Layer 2: Causal claims from studies
   - Layer 3: Meta-analysis / consensus
   - Query different layers for different confidence levels

5. **Confidence Scoring**
   ```
   (A)-[:CAUSES {
     confidence: 0.7,
     study_types: {rct: 0.9, observational: 0.5, expert_opinion: 0.3},
     meta_analysis_result: 0.75
   }]->(B)
   ```

---

## 4. Academic Papers on Causal Knowledge Graphs

### Foundational Papers

1. **"Causation, Prediction, and Search"** - Spirtes, Glymour, Scheines (2000, 2nd ed.)
   - Book. Foundation of constraint-based causal discovery.
   - PC and FCI algorithms.

2. **"Causal Inference in Statistics: A Primer"** - Pearl, Glymour, Jewell (2016)
   - Accessible introduction to causal inference using DAGs.
   - do-calculus, identifiability.

3. **"The Book of Why"** - Judea Pearl (2018)
   - Popular science. Ladder of causation: association → intervention → counterfactuals.

4. **"Elements of Causal Inference"** - Peters, Janzing, Schölkopf (2017)
   - MIT Press. Modern treatment of causal discovery from ML perspective.
   - Free PDF: https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/

### Causal Discovery Papers

5. **"An Algorithm for Fast Recovery of Sparse Causal Graphs"** - Spirtes & Glymour (1991)
   - Original PC algorithm paper.

6. **"Learning Sparse Nonparametric DAGs"** - Zheng, Dan, Aragam, Ravikumar, Xing (2020)
   - arXiv:1909.13189. NOTEARS extension to nonparametric models.
   - AISTATS 2020.

7. **"DAGs with NO TEARS: Continuous Optimization for Structure Learning"** - Zheng et al. (2018)
   - Original NOTEARS paper. NeurIPS 2018.
   - Reformulates DAG learning as continuous optimization.

8. **"A Survey of Learning Causality with Data"** - Guo, Cheng, Li, Hahn, Liu (2020)
   - arXiv:1809.09337. ACM Computing Surveys.
   - Comprehensive survey of causal discovery and inference methods.

9. **"Causal Discovery with Reinforcement Learning"** - Zhu et al. (2020)
   - ICLR 2020. Uses RL for DAG structure learning.

10. **"Causal Discovery in the Geosciences"** - Ebert-Uphoff & Deng (2017)
    - Applications of PC-stable to climate data.

### Causal Knowledge Graph Papers

11. **"Causal Knowledge Graphs"** - Various works
    - Combining causal inference with KG representation.
    - Key idea: Use KG structure to improve causal discovery, use causal discovery to enrich KG.

12. **"Causal Reasoning and Large Language Models: Opening a New Frontier for Causality"** - Kıcıman, Ness, Sharma, Tan (2023)
    - arXiv:2305.00050. TMLR 2024.
    - **Key finding:** GPT-3.5/4 achieves 97% on pairwise causal discovery, 92% on counterfactual reasoning.
    - LLMs can generate causal graphs from text metadata.
    - Code: https://github.com/py-why/pywhy-llm

13. **"Integrating Causal Reasoning into Knowledge Graphs"** - Various
    - Adding causal edges to existing KGs (e.g., ConceptNet, Wikidata).
    - Challenges: Distinguishing causal from associative facts.

14. **"Temporal Knowledge Graphs for Causal Discovery"** - Various
    - Extending KGs with temporal dimension for causal reasoning.
    - TKG embedding methods: TTransE, HyTE, TNTComplEX.

15. **"Causal Graph Neural Networks"** - Various
    - Using GNNs for causal discovery and inference.
    - DAG-GNN (Yu et al. 2019), Graph Autoencoders for causal structure.

### Temporal Causal Graph Papers

16. **"PCMCI: Discovering Causal Structure in High-Dimensional Time Series"** - Runge et al. (2019)
    - Nature Communications. Climate science applications.

17. **"Temporal Causal Discovery with Continuous-Time Models"** - Various
    - Hawkes processes for temporal point processes.
    - Neural temporal point processes.

18. **"Granger Causal Analysis in Discrete Time"** - Various
    - Foundation for time-series causality.

### LLM + Causal Reasoning Papers

19. **"Causal Reasoning and Large Language Models"** - Kıcıman et al. (2023)
    - arXiv:2305.00050. See #12 above.

20. **"Can Large Language Models Infer Causal Relations?"** - Various benchmarks
    - Evaluating LLM causal reasoning capabilities.
    - Findings: LLMs good at common-sense causality, struggle with novel/counterintuitive cases.

21. **"LLM-based Causal Discovery"** - Emerging work (2024-2025)
    - Using LLMs to generate causal graph candidates from text.
    - Combining LLM priors with statistical causal discovery.

### Causal Inference on Graphs Papers

22. **"Causal Inference on Graphs"** - Various
    - Treatment effect estimation when interference exists (network effects).
    - SUTVA violations in networked settings.

23. **"Estimating Causal Effects with Graph Neural Networks"** - Various
    - Using GNNs for causal effect estimation.
    - Combining graph structure with observational data.

---

## 5. Causal Graphs for Temporal Reasoning

### 5.1 Time-Series Causality

**Key Methods:**

1. **Granger Causality**
   - X Granger-causes Y if past X improves prediction of Y
   - Implemented via VAR models
   - **Limitation:** Only predictive, not interventional

2. **Transfer Entropy**
   - Information-theoretic measure of directed information flow
   - Nonlinear generalization of Granger causality
   - `TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})`

3. **Convergent Cross Mapping (CCM)**
   - For nonlinear dynamical systems
   - Uses Takens' embedding theorem
   - Can detect causation in chaotic systems

4. **PCMCI**
   - Combines PC with time-series structure
   - Handles autocorrelation, high dimensionality
   - Two steps: (1) PC for contemporaneous, (2) time-lagged conditional independence

### 5.2 Event Ordering

**Temporal Point Processes:**
- **Hawkes Processes:** Self-exciting point processes for event cascades
- **Neural Temporal Point Processes:** Neural networks for intensity functions
- **Application:** Model causal chains of events (e.g., social media, disease spread)

**Representation in Graph:**
```
(Event1)-[:OCCURRED_AT]->(Time1)
(Event2)-[:OCCURRED_AT]->(Time2)
(Event1)-[:CAUSES {time_lag: "5m"}]->(Event2)
```

**Querying Temporal Causal Chains:**
```cypher
MATCH path = (e1)-[:CAUSES*]->(eN)
WHERE e1.timestamp < eN.timestamp
  AND ALL(r IN relationships(path) | r.time_lag < "1h")
RETURN path
```

### 5.3 Fact Updates/Overwrites

**Problem:** Causal knowledge evolves. New evidence contradicts old beliefs.

**Solutions:**

1. **Versioned Edges**
   ```
   (A)-[:CAUSES {
     strength: 0.8,
     version: 1,
     valid_from: 2020-01-01,
     valid_until: 2023-06-01,
     superseded_by: "v2"
   }]->(B)
   ```

2. **Belief Revision**
   - Maintain belief state as probability distribution
   - Update via Bayesian updating when new evidence arrives
   - Store history of belief updates

3. **Conflict Resolution Strategies**
   - **Recency:** Prefer most recent evidence
   - **Quality:** Prefer higher-quality studies (RCT > observational)
   - **Consensus:** Aggregate multiple sources

### 5.4 Contradiction Resolution

**Approaches:**

1. **Contextual Causation**
   - Causal relationship holds under specific conditions
   - Store context as edge property or separate context node
   - Query with context matching

2. **Multi-World Graphs**
   - Different causal graphs for different populations/contexts
   - Query specific world or compare across worlds

3. **Probabilistic Causal Graphs**
   - Edge weights represent probability of causal relationship
   - Contradictions resolved by aggregating evidence

---

## 6. Practical Applications

### 6.1 Root Cause Analysis

**Use Case:** Identify root cause of system failures, bugs, outages.

**Example:** Microservice architecture failure
```
(ServiceA)-[:CALLS]->(ServiceB)-[:CALLS]->(ServiceC)
(ServiceB)-[:CAUSES {evidence: "latency_spike"}]->(ServiceC_timeout)
(ServiceA)-[:CAUSES {evidence: "memory_leak"}]->(ServiceB_OOM)
```

**Tools:** DoWhy GCM for anomaly attribution, causal-learn for discovery from logs/metrics.

### 6.2 Medical Diagnosis

**Use Case:** Causal reasoning over symptoms, diseases, treatments.

**Example:**
```
(Smoking)-[:CAUSES {strength: 2.0, evidence: "multiple_RCTs"}]->(LungCancer)
(LungCancer)-[:CAUSES]->(Cough)
(LungCancer)-[:CAUSES]->(ChestPain)
```

**Tools:** pgmpy for Bayesian network inference, DoWhy for treatment effect estimation.

### 6.3 Financial Modeling

**Use Case:** Causal relationships between economic indicators, market events.

**Example:**
```
(InterestRateHike)-[:CAUSES {lag: "3-6m"}]->(StockMarketDecline)
(Inflation)-[:CAUSES]->(InterestRateHike)
```

**Tools:** Time-series causal discovery (PCMCI, VAR-LiNGAM), Granger causality.

### 6.4 AI/Agent Applications

**Use Case:** Agent memory with causal reasoning.

**Example:**
- Agent observes events, builds causal model of environment
- Uses causal model for planning, prediction, explanation
- Updates causal model as new evidence arrives

**Tools:** causal-learn for discovery, DoWhy for inference, graph DB for storage.

### 6.5 Scientific Discovery

**Use Case:** Discover causal mechanisms from experimental/observational data.

**Examples:**
- Gene regulatory networks
- Climate causal networks
- Neuroscience (brain connectivity)

**Tools:** causal-learn, Tetrad, PCMCI.

---

## 7. Integration with LLMs

### 7.1 How LLMs Reason About Causality

**Findings from Kıcıman et al. (2023):**
- GPT-3.5/4 achieve 97% on pairwise causal discovery (13 points above previous SOTA)
- 92% on counterfactual reasoning (20 points above previous SOTA)
- 86% on event causality (necessary/sufficient causes)
- **Key insight:** LLMs operate on **text metadata**, not raw data
- Can generate causal graphs from natural language descriptions
- Generalize to novel datasets created after training cutoff

**Limitations:**
- Unpredictable failure modes
- Struggle with novel/counterintuitive causal relationships
- Cannot replace statistical analysis on actual data
- May hallucinate causal relationships

### 7.2 Can LLMs Discover Causal Relationships?

**Yes, but with caveats:**
- LLMs can identify **candidate** causal relationships from text/domain knowledge
- Useful for generating initial causal graph hypotheses
- Must be validated with statistical methods on actual data
- Best for common-sense causality, not novel scientific discovery

**Approach:**
1. Use LLM to generate candidate causal graph from domain description
2. Use statistical causal discovery to validate/refine
3. Iterate: LLM interprets results, suggests refinements

### 7.3 Combining LLM + Causal Graphs

**Architecture Options:**

1. **LLM as Causal Graph Generator**
   - Input: Natural language description of domain
   - LLM outputs: Candidate causal graph (DAG)
   - Refinement: Statistical validation, human review

2. **LLM as Causal Reasoner**
   - Input: Causal graph + query (e.g., "What happens if X changes?")
   - LLM uses graph structure to reason about interventions
   - Output: Natural language explanation of causal effects

3. **LLM + Statistical Causal Discovery**
   - LLM generates prior causal graph from text
   - Statistical methods refine graph from data
   - LLM interprets results, generates explanations

4. **Causal Graph as LLM Memory**
   - Store agent's causal knowledge in graph DB
   - Query graph for causal reasoning
   - Update graph as agent learns new causal relationships

**Code Example (LLM + DoWhy):**
```python
# Use LLM to generate causal graph from text
prompt = """Given these variables: X, Y, Z, W
Generate a causal DAG in DOT format based on domain knowledge."""
causal_graph_dot = llm.generate(prompt)

# Use DoWhy for causal inference
model = CausalModel(data=df, treatment="X", outcome="Y", graph=causal_graph_dot)
estimand = model.identify_effect()
estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

# Use LLM to interpret results
interpretation = llm.generate(f"Interpret this causal effect estimate: {estimate}")
```

### 7.4 Papers on LLM + Causal Knowledge Graphs

1. **"Causal Reasoning and Large Language Models"** - Kıcıman et al. (2023)
   - arXiv:2305.00050. See Section 4.

2. **"LLM-based Causal Discovery"** - Emerging work (2024-2025)
   - Using LLMs to generate causal graph candidates
   - Combining LLM priors with statistical methods

3. **"CausalBench"** - Benchmark for LLM causal reasoning
   - Evaluating LLMs on causal discovery tasks

---

## 8. Challenges and Limitations

### 8.1 Scalability

**Problem:** Causal discovery is NP-hard in general.

**Challenges:**
- PC algorithm: O(p^d) independence tests (p = variables, d = max degree)
- Score-based methods: Exponential number of DAGs
- NOTEARS: Scales better but still O(p^2) parameters

**Mitigations:**
- Constraint-based: Use prior knowledge to reduce search space
- Score-based: Use greedy search (GES) instead of exhaustive
- NOTEARS: Continuous optimization, but memory-intensive for large graphs
- Divide-and-conquer: Learn local causal neighborhoods, then combine

### 8.2 Noise in Data

**Problem:** Real-world data is noisy, leading to spurious causal relationships.

**Challenges:**
- Measurement error
- Missing data
- Outliers
- Small sample sizes

**Mitigations:**
- Robust independence tests (e.g., KCI for nonlinear)
- Bootstrap resampling for confidence intervals
- Regularization (e.g., NOTEARS with L1 penalty)
- Multiple algorithm consensus

### 8.3 Confounding Variables

**Problem:** Hidden common causes create spurious associations.

**Challenges:**
- Cannot distinguish confounding from direct causation without measuring confounders
- FCI can detect latent confounders but cannot identify full causal structure
- Instrumental variable methods require valid instruments (rare in practice)

**Mitigations:**
- FCI algorithm for latent confounder detection
- Sensitivity analysis: How strong must confounding be to explain away effect?
- Domain knowledge: Include known confounders in analysis
- Multiple environments: Use heterogeneity to identify causal structure

### 8.4 Temporal Resolution

**Problem:** Causal relationships operate at different time scales.

**Challenges:**
- Choosing appropriate time lag for Granger causality
- Aggregating/disaggregating temporal data
- Handling asynchronous events

**Mitigations:**
- Multi-scale temporal analysis
- Continuous-time models (Hawkes processes)
- Event-based rather than time-bucketed analysis

### 8.5 Identifiability

**Problem:** Observational data alone cannot always identify causal structure.

**Challenges:**
- Markov equivalence: Multiple DAGs encode same conditional independencies
- Latent confounders: Cannot distinguish from direct causation
- Non-Gaussianity required for LiNGAM

**Mitigations:**
- Interventional data: Do experiments to break Markov equivalence
- Prior knowledge: Constrain search space
- Multiple algorithms: Compare results across methods
- Sensitivity analysis: Quantify uncertainty

---

## 9. Graph Databases for Causal Reasoning

### 9.1 Neo4j + Causal Reasoning

**Current State:**
- Neo4j stores causal relationships as labeled, directed edges
- Cypher supports path queries, pattern matching
- No built-in causal inference algorithms

**Approach:**
1. Store causal graph in Neo4j
2. Use Python libraries (DoWhy, causal-learn) for inference
3. Query Neo4j for graph structure, pass to Python for analysis
4. Write results back to Neo4j

**Example Workflow:**
```python
# Query causal graph from Neo4j
graph_query = """
MATCH (a)-[r:CAUSES]->(b)
RETURN a.id AS source, b.id AS target, r.strength AS weight
"""
edges = neo4j_session.run(graph_query).data()

# Build NetworkX graph
G = nx.DiGraph()
for edge in edges:
    G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

# Use DoWhy for causal inference
model = CausalModel(data=df, treatment="X", outcome="Y", graph=G)
estimand = model.identify_effect()
estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

# Write results back to Neo4j
neo4j_session.run("""
MATCH (a {id: 'X'})-[r:CAUSES]->(b {id: 'Y'})
SET r.estimated_effect = $effect
""", effect=estimate.value)
```

### 9.2 Cypher Queries for Causal Paths

**Find all causal paths between two nodes:**
```cypher
MATCH path = (start {name: "X"})-[:CAUSES*]->(end {name: "Y"})
RETURN path, length(path) AS path_length
```

**Find causal chains with mediation:**
```cypher
MATCH path = (x)-[:CAUSES]->(m)-[:CAUSES]->(y)
WHERE x.name = "X" AND y.name = "Y"
RETURN m.name AS mediator, path
```

**Find confounders:**
```cypher
MATCH (conf)-[:CAUSES]->(x {name: "X"}),
      (conf)-[:CAUSES]->(y {name: "Y"})
WHERE NOT (x)-[:CAUSES]->(y) AND NOT (y)-[:CAUSES]->(x)
RETURN conf.name AS confounder
```

**Compute causal effect along path:**
```cypher
MATCH path = (x)-[r:CAUSES*]->(y)
WHERE x.name = "X" AND y.name = "Y"
RETURN path, reduce(s = 1.0, rel IN r | s * rel.strength) AS total_effect
```

### 9.3 Existing Work on Causal + Graph DB

**Limited existing work:**
- Most causal inference done in Python/R, not in graph DB
- Some integration via Python drivers (Neo4j + DoWhy)
- No native causal inference in graph DBs yet

**Opportunity:**
- HydraDB could be first graph DB with native causal reasoning
- Embed causal discovery algorithms in query language
- Real-time causal reasoning as data arrives

---

## 10. Relevant Tools and Frameworks

### 10.1 Tools Combining Causal Reasoning + Knowledge Graphs

**Current Landscape:**
- **No integrated tool** combines causal discovery + KG storage + causal inference
- Most workflows: Python libraries for discovery/inference, graph DB for storage
- Opportunity for HydraDB to fill this gap

**Closest Existing Tools:**

1. **DoWhy + NetworkX + Neo4j**
   - DoWhy for causal inference
   - NetworkX for graph manipulation
   - Neo4j for storage
   - Manual integration

2. **causal-learn + Neo4j**
   - causal-learn for discovery
   - Neo4j for storage
   - Manual integration

3. **CausalNex**
   - NOTEARS + Bayesian networks
   - No KG storage

### 10.2 Emerging Tools

1. **PyWhy Ecosystem**
   - DoWhy, EconML, causal-learn
   - Moving toward unified causal AI platform
   - No native KG integration yet

2. **CausalML (Uber)**
   - Treatment effect estimation
   - No causal discovery or KG storage

3. **CDT (Causal Discovery Toolbox)**
   - Includes CGNN (Causal Generative Neural Networks)
   - No KG integration

---

## 11. Integration with HydraDB for Agent Memory

### 11.1 Architecture Proposal

**HydraDB as Causal Knowledge Graph for Agent Memory:**

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Memory Layer                     │
│  (Episodic memory, semantic memory, procedural memory)   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Causal Knowledge Graph (HydraDB)            │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Entities   │  │   Causal     │  │   Temporal   │  │
│  │   (Nodes)    │  │   Edges      │  │   Events     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  Metadata: evidence, confidence, context, temporal info  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           Causal Reasoning Engine                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Causal     │  │   Effect     │  │   Counter-   │  │
│  │  Discovery   │  │  Estimation  │  │  factuals    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           LLM Integration Layer                          │
│                                                          │
│  - Generate causal hypotheses from text                  │
│  - Interpret causal inference results                    │
│  - Natural language explanations                         │
└─────────────────────────────────────────────────────────┘
```

### 11.2 Key Features for HydraDB

1. **Native Causal Edge Types**
   ```rust
   enum CausalRelation {
       Causes { strength: f64, evidence: Vec<String> },
       Prevents { strength: f64, evidence: Vec<String> },
       Enables { strength: f64, evidence: Vec<String> },
       CorrelatesWith { strength: f64 },
   }
   ```

2. **Causal Discovery Queries**
   ```cypher
   DISCOVER CAUSAL STRUCTURE FROM observations
   WHERE variables = [X, Y, Z, W]
   USING algorithm = "PC"
   RETURN causal_graph
   ```

3. **Causal Effect Queries**
   ```cypher
   ESTIMATE EFFECT OF treatment = "X" ON outcome = "Y"
   USING backdoor_adjustment
   RETURN average_treatment_effect, confidence_interval
   ```

4. **Counterfactual Queries**
   ```cypher
   WHAT_IF X = 1
   GIVEN observed_data
   RETURN counterfactual_outcome
   ```

5. **Temporal Causal Queries**
   ```cypher
   MATCH (e1:Event)-[:CAUSES]->(e2:Event)
   WHERE e1.timestamp < e2.timestamp
     AND duration.between(e1.timestamp, e2.timestamp) < "24h"
   RETURN e1, e2
   ```

### 11.3 Use Cases for Agent Memory

1. **Episodic Memory with Causal Structure**
   - Agent stores events with causal relationships
   - Query: "What caused this outcome?"
   - Query: "What will happen if I do X?"

2. **Semantic Memory with Causal Knowledge**
   - Store domain knowledge as causal graph
   - Update as agent learns new causal relationships
   - Resolve contradictions via evidence weighting

3. **Procedural Memory with Causal Models**
   - Store action-outcome relationships
   - Plan by simulating causal chains
   - Learn from successes/failures

4. **Explanation Generation**
   - Agent explains decisions using causal graph
   - "I chose X because it causes Y, which leads to desired outcome Z"

### 11.4 Implementation Roadmap

**Phase 1: Causal Edge Types**
- Add causal edge types to HydraDB schema
- Support metadata: strength, evidence, confidence, context

**Phase 2: Causal Query Language**
- Extend Cypher with causal query primitives
- Path queries for causal chains
- Confounder detection queries

**Phase 3: Causal Discovery Integration**
- Embed causal-learn algorithms in HydraDB
- Real-time causal discovery as data arrives
- Incremental graph updates

**Phase 4: LLM Integration**
- LLM generates causal hypotheses from text
- LLM interprets causal inference results
- Natural language causal queries

**Phase 5: Agent Memory API**
- High-level API for agent memory operations
- Episodic/semantic/procedural memory with causal structure
- Explanation generation

---

## 12. Summary and Recommendations

### Key Takeaways

1. **Causal discovery is mature** - Multiple algorithms (PC, FCI, GES, NOTEARS, LiNGAM) with Python libraries (causal-learn, DoWhy, pgmpy, CausalNex).

2. **LLMs enable causal reasoning from text** - GPT-3.5/4 achieve 97% on pairwise causal discovery. Can generate causal graph candidates from natural language.

3. **No integrated tool exists** - No system combines causal discovery + KG storage + causal inference + LLM integration. HydraDB can fill this gap.

4. **Temporal causal reasoning is critical for agent memory** - Agents need to reason about events, causes, effects over time.

5. **Contradiction resolution is hard but solvable** - Use evidence weighting, contextual causation, belief revision.

### Recommendations for HydraDB

1. **Start with causal edge types** - Add `CAUSES`, `PREVENTS`, `ENABLES` with metadata (strength, evidence, confidence).

2. **Integrate causal-learn** - Embed causal discovery algorithms for real-time structure learning.

3. **Extend Cypher with causal queries** - `DISCOVER CAUSAL STRUCTURE`, `ESTIMATE EFFECT`, `WHAT_IF`.

4. **LLM integration layer** - Use LLMs to generate causal hypotheses, interpret results, generate explanations.

5. **Focus on agent memory use cases** - Episodic memory with causal structure, explanation generation, planning via causal simulation.

### Next Steps

1. **Prototype causal edge types** in HydraDB
2. **Benchmark causal discovery algorithms** on agent memory datasets
3. **Design causal query language extensions**
4. **Build LLM + causal graph integration**
5. **Evaluate on agent memory tasks** (explanation, planning, prediction)

---

## References

### Papers
- Spirtes, Glymour, Scheines. "Causation, Prediction, and Search." 2000.
- Pearl, Glymour, Jewell. "Causal Inference in Statistics: A Primer." 2016.
- Zheng et al. "DAGs with NO TEARS." NeurIPS 2018. arXiv:1803.01422.
- Zheng et al. "Learning Sparse Nonparametric DAGs." AISTATS 2020. arXiv:1909.13189.
- Kıcıman et al. "Causal Reasoning and Large Language Models." TMLR 2024. arXiv:2305.00050.
- Guo et al. "A Survey of Learning Causality with Data." ACM Computing Surveys 2020. arXiv:1809.09337.
- Runge et al. "PCMCI." Nature Communications 2019.

### Libraries
- causal-learn: https://github.com/py-why/causal-learn
- DoWhy: https://github.com/py-why/dowhy
- pgmpy: https://github.com/pgmpy/pgmpy
- CausalNex: https://github.com/quantumblacklabs/causalnex
- Tetrad: https://github.com/cmu-phil/tetrad
- PyWhy ecosystem: https://www.pywhy.org/

### Books
- Pearl. "The Book of Why." 2018.
- Peters, Janzing, Schölkopf. "Elements of Causal Inference." 2017. Free PDF: https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/

---

**Report compiled:** 2026-08-13
**For:** HydraDB hackathon project
**Purpose:** Inform technical architecture for causal discovery + knowledge graph integration for agent memory
