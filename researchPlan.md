# ML Research Plan: 10x/100x/1000x Agent Memory & Learning

## Overview

EventGraphDB already captures the full event lifecycle: Event → Episode → Memory → Strategy → Graph. Every `process_event()` call generates training data. The models below replace hand-tuned heuristics with learned intelligence, then build toward a foundation model for agent cognition.

---

## 10x — Learned Replacements for Hand-Tuned Heuristics

### 1. Episode Boundary Detection Network

**Problem:** Rule-based `is_episode_start()` / `is_episode_end()` miss implicit episode structure. Only fires on GoalFormation, Planning, LearningUpdate, or explicit outcomes.

**Architecture:** 1D CNN or lightweight LSTM over sliding windows of event embeddings (~500K params, <1ms per event on CPU).

**Training Signal:** Self-supervised — use time gaps, context shifts, and outcome events as noisy labels, then fine-tune on human-annotated episode boundaries.

**Coding Agent Example:**
- Current rules miss this natural episode: `ReadFile → grep for pattern → edit 3 files → run tests → fix lint error → run tests again → commit`. There's no GoalFormation event — the agent just started coding. The model learns that `ReadFile` after a long idle gap followed by edits is an episode start, and `commit` is an episode end.
- Retry patterns like `run tests → fail → edit → run tests → pass` get split into two episodes today (first test failure ends it). The model learns this is one coherent episode.

**Support Agent Example:**
- A customer conversation flows: `receive ticket → read KB articles → ask clarifying question → receive reply → escalate to tier 2 → receive resolution → close ticket`. Current rules might segment this at the escalation (context shift). The model learns that escalation is mid-episode, not a boundary.
- Agent handles two interleaved tickets simultaneously. The model learns to segment by ticket context, not by time ordering.

---

### 2. Significance Predictor

**Problem:** `calculate_significance()` uses hand-tuned weights (+0.1 for Planning, +0.08 for Reasoning). These are guesses about what matters.

**Architecture:** Small MLP or gradient-boosted tree over event features (type, context hash, goal count, chain length, novelty, temporal position in episode).

**Training Signal:** Hindsight labeling — events that ended up in memories that were later retrieved AND led to successful outcomes get high significance. Events in memories that were never retrieved get low significance.

**Coding Agent Example:**
- The model discovers that `ReadFile` events for test files before editing source code are highly significant (the agent that reads tests first succeeds more). Current heuristics give all `Observation` events the same +0.08.
- `ActionOutcome::Failure` on a build step after changing a dependency is more significant than a lint warning failure. The model learns failure context matters.

**Support Agent Example:**
- The model learns that a customer's second message (the clarification) is more significant than the first (the complaint), because resolutions that reference the clarification succeed more often.
- KB article lookups that match the eventual resolution are high-significance. KB lookups that were dead ends get low scores. Current heuristics treat all lookups equally.

---

### 3. Neural Memory Retriever

**Problem:** Memory retrieval uses context-hash matching (exact) or cosine similarity on optional embeddings. Misses semantically similar but structurally different situations.

**Architecture:** Bi-encoder (two small transformers, 5-20M params). Query encoder takes current context, memory encoder takes stored memory. Contrastive training with in-batch negatives.

**Training Signal:** Positive pairs = (query context at retrieval time, memory that was retrieved and led to success). Hard negatives = memories that were retrieved but didn't help.

**Coding Agent Example:**
- Agent is debugging a null pointer in a React component. The retriever finds a memory from 3 weeks ago about debugging an undefined property in a Vue component — different framework, same pattern. Context hashes are completely different, but the learned embeddings are close.
- Agent is writing a REST endpoint. The retriever surfaces a memory about writing a GraphQL resolver in the same codebase, because the model learned they share auth middleware patterns.

**Support Agent Example:**
- Customer reports "app crashes on login." The retriever finds a memory from a different customer who said "can't sign in, screen goes white." Different words, same root cause. Hash matching would miss this entirely.
- Agent handling a billing dispute retrieves a memory about a refund policy edge case from a different product line, because the model learned the dispute patterns are structurally similar.

---

## 100x — Structural Intelligence

### 4. Causal Discovery Network

**Problem:** Strategies capture "this sequence preceded success." But correlation isn't causation — many steps in a strategy may be irrelevant or even harmful.

**Architecture:** Temporal causal transformer. Attention weights become causal edge probabilities. Trained with interventional objectives (NOTEARS constraint + transformer, hybrid approach).

**Training Signal:** Counterfactual — mask out events from episodes and predict whether the outcome changes. Events whose removal flips Success→Failure are causal. Events whose removal changes nothing are noise.

**Coding Agent Example:**
- Episode: `read docs → create branch → write code → write tests → run tests → fix typo → run tests → commit`. Causal discovery finds that `write tests` causally determines `run tests → pass`, but `read docs` and `create branch` don't affect the outcome. The resulting strategy is compact: "write tests before running them" not "read docs, create branch, write code, write tests, run tests, fix typo, run tests, commit."
- Agent discovers that `cargo clippy` before `cargo test` causally reduces test failures by catching type errors early. This isn't in any strategy — it's an emergent causal pattern across hundreds of episodes.

**Support Agent Example:**
- Episode: `read ticket → check KB → ask customer OS version → receive "Windows 11" → apply known fix → confirm resolution`. Causal discovery finds that `ask customer OS version` is the causal pivot — without it, the agent applies wrong fixes 60% of the time. The strategy becomes: "always ask OS version for crash tickets" not the full 6-step sequence.
- Across 1000 escalation episodes, the model discovers that tickets where the agent summarized the problem before escalating resolve 3x faster. "Summarize before escalate" becomes a causal strategy, not just a correlation.

---

### 5. Graph Neural Network for Knowledge Graph Reasoning

**Problem:** The knowledge graph has rich relational structure (Agent→Episode→Memory→Strategy→Goal) but queries are flat key lookups. No reasoning over structure.

**Architecture:** R-GCN (Relational Graph Convolutional Network) or GAT over the heterogeneous NodeType/EdgeType schema.

**Tasks:**
- Link prediction: "Should this strategy be connected to this memory?"
- Node importance: Learned replacement for hand-coded `signal()` weights
- Subgraph matching: "Find the subgraph most similar to agent 2's experience"

**Coding Agent Example:**
- Link prediction discovers that a `Strategy` for "retry with exponential backoff" should be connected to a `Memory` about API rate limiting, even though they were formed in different episodes months apart. The agent now retrieves the retry strategy when it encounters rate limits.
- Subgraph matching: Agent 1 debugged a complex race condition across 5 episodes. Agent 2 encounters a similar concurrency bug. The GNN finds the structural similarity (interleaved read/write events, failure after concurrent access, fix by adding mutex) and transfers the entire subgraph of strategies.

**Support Agent Example:**
- The GNN discovers that `Customer→Ticket→Resolution` subgraphs cluster into 15 canonical patterns (password reset, billing dispute, feature request, etc.). New tickets are classified by subgraph similarity in real-time, routing to the right resolution pattern before the agent even reads the ticket.
- Link prediction connects a rarely-used `Strategy` about GDPR data deletion to a `Memory` about a customer requesting account removal. The agent proactively applies the strategy when it sees deletion requests.

---

### 6. Differentiable Memory Consolidation (CLS-Inspired)

**Problem:** The Episodic→Semantic→Schema consolidation pipeline uses hard thresholds (min episodes, similarity cutoff). It can't adapt to what's worth remembering per agent.

**Architecture:** Inspired by Complementary Learning Systems theory (neuroscience):
- Fast pathway (hippocampus): Current episodic store — verbatim event recording
- Slow pathway (neocortex): Neural network that gradually distills episodic memories into compressed semantic representations
- Replay mechanism: Periodic "dreaming" where the slow pathway replays episodic memories, prioritized by salience score

**Training Signal:** Reconstruction loss (can the semantic representation reconstruct key episode features?) + utility loss (do downstream strategy extractions improve with the consolidated memory?).

**Coding Agent Example:**
- After 50 episodes of writing Python functions, the consolidator distills: "Functions with type hints and docstrings pass review faster." This is a semantic memory that no single episode contains — it's the statistical pattern across all 50. Current threshold-based consolidation would need explicit similarity between episodes to merge them.
- The agent has 200 episodic memories about debugging. The consolidator compresses them into 5 semantic memories: "check logs first," "reproduce before fixing," "regression test after fix," "git bisect for mysterious breaks," "read the error message carefully." Each semantic memory is a lossy compression of ~40 episodes, keeping only what generalizes.

**Support Agent Example:**
- 300 password-reset episodes consolidate into one semantic memory with the causal core: "verify identity → reset → confirm new login works." Variations (phone vs email verification, temporary vs permanent password) are preserved as conditional branches in the semantic representation.
- The consolidator learns that some episodic memories should NOT be consolidated — rare edge cases (e.g., "customer had two accounts with same email") need to stay episodic because they're individually important and don't generalize.

---

## 1000x — Foundation Models for Agent Cognition

### 7. Event Sequence Foundation Model

**Problem:** Episode detection, significance scoring, memory retrieval, and strategy extraction are all separate hand-tuned systems. They should be one model that *understands* event sequences.

**Architecture:** State Space Model (Mamba-2) over tokenized event sequences. Events become tokens:
```
[GoalFormation|ctx=0x3F|goals=2|t=+0ms]
[Planning|ctx=0x3F|goals=2|t=+50ms]
[Action:write_file|outcome=success|dur=200ms|t=+300ms]
[Action:run_tests|outcome=failure|dur=5000ms|t=+800ms]
```

**Pre-training:** Next-event prediction + masked event prediction over all agents' event histories (50-200M params, linear-time in sequence length via SSM).

**Fine-tuning tasks (zero-shot or few-shot after pre-training):**
- Episode segmentation → "where do episode boundaries fall?"
- Outcome prediction → "will this episode succeed?"
- Anomaly detection → "this event sequence is unusual"
- Strategy generation → "given this context, what action sequence maximizes P(success)?"
- Memory importance → "which events should be remembered?"

**Coding Agent Example:**
- Pre-trained on 10M event sequences from coding agents. The model learns the "rhythm" of coding: read → plan → edit → test → fix → test → commit. It predicts that after `run_tests → failure`, the next event is likely `edit` (not `commit`), and flags a `commit` after test failure as anomalous.
- Given a new codebase and task, the model generates an action sequence: "read README → read existing tests → write failing test → implement feature → run tests → iterate." This isn't a stored strategy — it's generated from the model's understanding of what works across millions of coding episodes.
- Zero-shot episode segmentation: Feed the model a stream of 500 events from a day of coding. It outputs episode boundaries with 95% accuracy, no rules needed.

**Support Agent Example:**
- Pre-trained on 5M support ticket event sequences. The model learns that `customer_message → KB_lookup → agent_reply` is the basic unit, and that tickets with more than 3 of these cycles tend to escalate.
- Outcome prediction: 2 messages into a conversation, the model predicts with 80% accuracy whether this ticket will resolve in one interaction or require escalation. The agent proactively adjusts its approach.
- Anomaly detection: The model flags when a support agent skips identity verification before resetting a password — this deviates from the learned distribution of successful episodes.

---

### 8. World Model for Counterfactual Planning

**Problem:** Strategies are extracted from *what happened*. Agents can't reason about *what could have happened* — no counterfactual thinking.

**Architecture:** JEPA (Joint Embedding Predictive Architecture) or Dreamer-v3 style:
- Encoder: Event → latent state
- Dynamics model: latent state + action → next latent state
- Decoder: latent state → predicted outcome/next event

**Training:** Self-supervised from event sequences — predict future states from past states + actions taken.

**Coding Agent Example:**
- Agent is deciding between two approaches: refactor the module first then add the feature, or add the feature inline. The world model simulates both:
  - Path A (refactor first): 12 events, P(test pass) = 0.85, estimated time = 20 min
  - Path B (inline): 6 events, P(test pass) = 0.60, P(follow-up refactor needed) = 0.70
  - Decision: Path A. This reasoning happens in latent space in milliseconds, without executing any code.
- After a deployment failure, the agent asks: "What if I had run integration tests instead of just unit tests?" The world model simulates the counterfactual and estimates that integration tests would have caught the bug with 73% probability. This becomes a causal lesson: "run integration tests before deploying."

**Support Agent Example:**
- Agent is mid-conversation and considering: offer a refund immediately, or troubleshoot first? The world model simulates:
  - Path A (immediate refund): 3 events, customer satisfaction = 0.7, cost = $50
  - Path B (troubleshoot): 8 events, P(resolution without refund) = 0.6, customer satisfaction if resolved = 0.95
  - Decision depends on customer sentiment signals and ticket priority.
- Post-escalation analysis: "What if I had asked for the error screenshot in message 2 instead of message 5?" The world model estimates this would have shortened resolution by 3 messages. The strategy "ask for screenshots early" gets a causal confidence boost.

---

### 9. Meta-Learning for Cross-Agent Strategy Transfer

**Problem:** Agent 1 learns a strategy through 50 episodes of trial and error. Agent 2 faces a structurally similar problem but starts from scratch. No mechanism for instant knowledge transfer.

**Architecture:** Model-Agnostic Meta-Learning (MAML) adapted for strategy networks:
- Inner loop: Agent fine-tunes a strategy network on its own episodes (few-shot, 3-5 episodes)
- Outer loop: Meta-learner optimizes initial parameters across all agents so strategies transfer maximally

**Training:** Across all agents' episode histories. The meta-learner discovers what's universal (retry on transient failure) vs. agent-specific (this agent's codebase has flaky tests).

**Coding Agent Example:**
- Agent 1 (Python specialist) learns over 100 episodes: "when tests fail due to import errors, check virtual environment first." Agent 3 (Node.js specialist) encounters module-not-found errors. The meta-learner recognizes structural similarity: both are dependency resolution problems. Agent 3 receives a transferred strategy: "when module errors occur, check node_modules / package.json first" — adapted from Agent 1's learning in 0 episodes instead of 100.
- 50 coding agents across different repositories. The meta-learner discovers a universal strategy: "read test files before writing implementation." This transfers to every new agent on first episode. Repository-specific knowledge (project structure, CI pipeline quirks) requires agent-specific fine-tuning.

**Support Agent Example:**
- Agent handling enterprise accounts learns: "for outage reports, immediately check status page before troubleshooting." The meta-learner transfers this to the consumer support agent: "for service complaints, check system status first." Same pattern, different customer segment.
- New support agent onboards. Instead of learning from scratch over 500 tickets, the meta-learner provides a warm-start strategy set distilled from all existing agents: top 20 strategies with transfer confidence scores. The new agent is effective from ticket #1.

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 months)
| Model | Effort | Impact | Training Data Source |
|---|---|---|---|
| Significance Predictor (#2) | 1 week | 10x | Memory retrieval logs + outcome labels |
| Episode Boundary Detector (#1) | 2 weeks | 10x | Self-supervised from existing episodes |
| Neural Memory Retriever (#3) | 3 weeks | 10x | Retrieval + outcome pairs |

### Phase 2: Structural Models (3-6 months)
| Model | Effort | Impact | Training Data Source |
|---|---|---|---|
| Causal Discovery Network (#4) | 6 weeks | 100x | Episode event sequences + outcomes |
| GNN for Knowledge Graph (#5) | 6 weeks | 100x | Full graph structure |
| Differentiable Consolidation (#6) | 8 weeks | 100x | Episodic→semantic memory transitions |

### Phase 3: Foundation (6-12 months)
| Model | Effort | Impact | Training Data Source |
|---|---|---|---|
| Event Sequence Foundation Model (#7) | 4 months | 1000x | All event sequences across all agents |
| World Model (#8) | 4 months | 1000x | Event sequences as state transitions |
| Meta-Learning Transfer (#9) | 3 months | 1000x | Cross-agent episode histories |

---

## Training Data — What You Already Have

Every function in the pipeline generates labeled data:

| Pipeline Stage | Generates Labels For |
|---|---|
| `process_event()` | Event sequences, temporal patterns |
| `complete_episode()` | Episode boundaries, outcomes |
| `calculate_significance()` | Current significance scores (weak labels) |
| `retrieve_memories_by_context()` | (query, retrieved memory) pairs |
| `process_episode_for_strategy()` | Episode→strategy mappings |
| `update_strategy_outcome()` | Strategy quality ground truth |
| Memory retrieval → action → outcome | End-to-end utility signal |

The critical addition: **log every memory retrieval and what happened after**. This closes the feedback loop and generates the hindsight labels that models #2 and #3 need.

---

## Research Publication Targets

| Model | Venue | Paper Angle |
|---|---|---|
| Causal Discovery (#4) | NeurIPS / ICML | "Causal Strategy Extraction from Agent Behavioral Traces" |
| Differentiable Consolidation (#6) | ICLR | "Complementary Learning Systems for Autonomous Agent Memory" |
| Foundation Model (#7) | NeurIPS | "Event Sequence Foundation Models for Agent Cognition" |
| World Model (#8) | ICML | "World Models for Symbolic Agent Environments" |
| Meta-Learning Transfer (#9) | AAMAS / ICLR | "Few-Shot Strategy Transfer in Multi-Agent Systems" |
