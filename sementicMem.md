Decisions captured from this conversation

D-1 The event stream includes a ContextEvent type (explicit).
D-2 Any non-ContextEvent whose context exceeds a configured size threshold SHALL also be processed for summarisation/claim extraction to avoid storing massive text inline.
D-3 Distillation pipeline order: NER first → embedding → LLM summarised key facts/points.
D-4 Each derived key point is persisted as a DerivedClaim: stored in KV (canonical for retrieval) and also as a graph node (for explainability).
D-5 Users can perform general claim search and memory search (memories returned with attached claims).
D-6 Every claim MUST include supporting evidence spans; no evidence means no persistence.
D-7 Geometric distance (embedding similarity) SHALL be used as a sanity check to detect likely hallucinations/ungrounded claims.
D-8 CPU-only constraint applies to NER; use external NER service (rust-bert in microservice).
D-9 No semantic claim_type taxonomy is required; growth is controlled via governance mechanics (gates, dedup/merge, TTL, quotas, compaction, contradiction handling).

1. Context ingestion and “large-context promotion”

FR-1 The system SHALL ingest Events into an append-only stream.

FR-2 The system SHALL support ContextEvent as an event type.

FR-3 The system SHALL additionally process non-ContextEvents for distillation if context_size_bytes >= PROMOTION_THRESHOLD_BYTES (configurable).

FR-4 For any eligible distillation input (ContextEvent or promoted), the system SHALL avoid storing large text inline by writing large context to durable segment storage and retaining only a pointer in the Event or side-record.

FR-5 Each eligible distillation input SHALL be linkable to episode_id OR thread_id (at least one required). Optional: user_id, workspace_id.

2. Canonical text resolution and span conventions

FR-6 The system SHALL resolve a canonical source text buffer for each eligible input from:

inline text when small, OR

dereferencing segment storage using a stable pointer.

FR-7 The system SHALL define and enforce one global offset convention for all spans (e.g., UTF-8 byte offsets) used by NER and evidence spans.

3. Deterministic NER extraction (CPU-only; Rust library)

FR-8 The system SHALL perform deterministic NER as the first processing step on eligible inputs.

FR-9 The NER implementation SHALL be CPU-only and SHALL use an external NER service (rust-bert in a separate microservice, or equivalent explicitly approved CPU-capable stack) to produce entity spans with offsets.

FR-10 The system SHALL persist ExtractedFeatures for audit/replay, including at minimum:

entity spans (label + start/end offsets),

tokenization/span mapping sufficient to validate offsets,

a feature fingerprint/hash for idempotency.

4. Embedding generation and indexing

FR-11 The system SHALL compute an embedding for each eligible input (context embedding).

FR-12 The system SHALL compute embeddings for claims (claim embeddings) after claim text is produced and accepted.

FR-13 Embeddings SHALL be persisted and used for:

semantic retrieval (claim search and/or memory search ranking support),

dedup/merge candidate discovery,

geometric-distance sanity checks (anti-hallucination).

5. ANN (where it is required and what it powers)

FR-14 The system SHALL support an ANN index for embeddings when claim volumes exceed a configured threshold or latency targets require it.

FR-15 ANN SHALL be used for general claim search:

retrieve top candidate claims by embedding similarity under user/workspace scope filters.

FR-16 ANN SHOULD be used for deduplication/merge control:

before persisting a new claim, query ANN for nearest existing claims in the same scope; if similarity exceeds a threshold, the system SHALL merge by adding support rather than creating a new claim node.

FR-17 ANN MAY be used for context-level retrieval (optional). Correctness MUST NOT depend on ANN; KV + pointers are canonical.

6. LLM distillation into DerivedClaims (key points/facts)

FR-18 The system SHALL call an LLM to distil canonical text + NER features into candidate key facts/points.

FR-19 Candidate claims SHALL be constrained to be atomic and include:

claim_text

supporting_evidence_spans[] (required)

confidence in [0,1]

linkage (episode/thread and optional user/workspace)

source_event_id (origin)

FR-20 Any claim without evidence spans SHALL be rejected and MUST NOT be persisted.

FR-21 Evidence spans SHALL be validated by the system against canonical text:

bounds correctness,

substring existence under the global offset convention,

integrity checks as needed (e.g., content hash / snippet hash).

7. Claim persistence (KV canonical + Graph for explainability)

FR-22 Each accepted DerivedClaim SHALL be persisted to KV as the canonical retrieval record.

FR-23 Each accepted DerivedClaim SHALL also be persisted as a graph node for traversal/explainability, including edges:

DERIVED_FROM (claim → source event)

SUPPORTED_BY (claim → evidence spans)

ASSOCIATED_WITH (claim → memory) when memory exists

optional ABOUT (claim → entity nodes) derived from NER

8. Retrieval requirements

FR-24 The system SHALL provide general claim search:

input: user_id, query_text, optional thread_id/workspace_id, k

retrieval: ANN over claim embeddings (when enabled) + deterministic filtering

output: claims + confidence + evidence pointers/snippets (configurable)

FR-25 The system SHALL provide memory search returning claims:

retrieve top-K memories

attach top-N associated claims per memory (configurable)

claims included MUST provide evidence references and confidence

9. Geometric distance anti-hallucination controls

FR-26 The system SHALL compute a geometric consistency score for each candidate claim using embeddings.

FR-27 The system SHALL implement at least:

Claim–Evidence proximity: claim embedding must be within threshold of evidence-span embedding(s)

Claim–Context proximity: claim embedding must be within threshold of the source context embedding

FR-28 Claims failing geometric checks SHALL be:

rejected pre-persistence (hard gate) OR

persisted but marked non-active / downranked (soft gate),
as configured.

10. Growth control (no semantic claim taxonomy)

FR-29 The system SHALL implement claim overgrowth controls without relying on semantic claim_type categories.

FR-30 The system SHALL enforce admission controls:

max claims per eligible input

minimum evidence quality (length/content-bearing checks)

atomicity enforcement

FR-31 The system SHALL implement dedup/merge:

exact fingerprint checks

ANN-based near-duplicate checks (when enabled)

merge action = add support to an existing claim rather than creating a new claim

FR-32 The system SHALL implement lifecycle governance:

scope (Episode/Thread/User)

TTL/aging into Dormant

quotas per scope and compaction/eviction policy

contradiction handling (do not overwrite; mark disputed and link)

11. Retirement and association

FR-33 When an episode completes, the system SHALL associate claims to the resulting Memory deterministically (same episode/thread default).

FR-34 Retiring a Memory SHALL retire/deactivate associated claims and remove them from active retrieval indexes, including ANN (while keeping by-id audit access).

12. Rust NER implementation constraint

NFR-1 NER SHALL operate on CPU only.

NFR-2 The NER component SHALL be implemented as an external service using rust-bert unless a replacement is explicitly approved with equivalent CPU-only capability and span/offset support.


Requirements Addendum: Cost Controls for LLM Usage (standard economics)
1) Token and call minimisation

FR-COST-1 The system SHALL NOT send full conversation history to an LLM for claim extraction. Claim extraction SHALL operate only on:

eligible ContextEvent payloads, and

promoted large-context payloads from other events (per the Promotion Threshold rule).

FR-COST-2 The system SHALL bound LLM input size for claim extraction by using deterministic chunking when canonical text exceeds a configurable maximum length.

FR-COST-3 The system SHALL enforce a maximum number of claims generated per eligible input (configurable cap) to bound output tokens and downstream storage growth.

2) Model routing (cheap-by-default)

FR-COST-4 The system SHALL support model routing for claim extraction such that:

a low-cost model is the default for claim/key-point distillation,

higher-cost models are used only when explicitly configured or when escalation rules trigger.

FR-COST-5 The system SHALL support provider abstraction for the claim model, enabling:

OpenAI models,

third-party APIs, and/or

self-hosted open-weights models.

3) Asynchronous, batched extraction

FR-COST-6 Claim distillation SHALL be asynchronous and off the critical path of event ingestion and user query serving.

FR-COST-7 The system SHOULD batch claim distillation requests where feasible to reduce overhead and improve throughput.

4) Gating and escalation rules (reduce wasted calls)

FR-COST-8 The system SHALL apply deterministic pre-gates before any LLM call, including:

basic payload validity checks,

NER extraction completion,

idempotency checks (no re-distillation for identical inputs).

FR-COST-9 The system SHALL support escalation rules that route to a stronger model only when necessary, including (configurable):

low confidence from the default model,

failure of geometric-distance sanity checks (claim–evidence / claim–context),

high user-impact contexts (policy-controlled).

5) Use embeddings and geometric distance to reduce churn

FR-COST-10 The system SHALL compute embeddings for contexts and claims and SHALL use geometric-distance checks to prevent low-quality claims from becoming active, reducing reprocessing and downstream cost.

6) Deduplication/merge to prevent runaway growth and repeated spend

FR-COST-11 The system SHALL implement deduplication/merge logic so that semantically similar claims do not create new nodes unnecessarily; instead, the system SHALL add support to existing claims.

FR-COST-12 When ANN is enabled, the system SHOULD use ANN-based nearest-neighbour search to identify merge candidates prior to persisting new claims.

7) Observability for cost governance

FR-COST-13 The system SHALL record per-claim-distillation job metrics, including:

tokens in/out (or provider usage units),

model/provider used,

latency,

number of accepted claims,

number of rejected claims and rejection reasons (no-evidence, invalid spans, geometric failure, duplicate/merged).

FR-COST-14 The system SHALL expose aggregated cost/usage metrics per workspace/user/thread to support quotas, billing, and operational governance.