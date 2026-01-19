// ============================================================================
// Core Types
// ============================================================================

export type AgentId = number;
export type ContextHash = number; // u64 in Rust
export type EventId = number; // u128 in Rust
export type SessionId = number; // u64 in Rust

// ============================================================================
// Event Types
// ============================================================================

// Metadata value types
export type MetadataValue =
  | { String: string }
  | { Integer: number }
  | { Float: number }
  | { Boolean: boolean }
  | { Json: any };

export interface Event {
  id: EventId;
  timestamp: number;
  agent_id: AgentId;
  agent_type: string;
  session_id: number;
  event_type: EventType;
  causality_chain: EventId[];
  context: EventContext;
  metadata: Record<string, MetadataValue>;
}

export type EventType =
  | { Action: ActionEvent }
  | { Observation: ObservationEvent }
  | { Cognitive: CognitiveEvent }
  | { Communication: CommunicationEvent }
  | { Learning: LearningEvent };

export type LearningEvent =
  | { MemoryRetrieved: { query_id: string; memory_ids: number[] } }
  | { MemoryUsed: { query_id: string; memory_id: number } }
  | { StrategyServed: { query_id: string; strategy_ids: number[] } }
  | { StrategyUsed: { query_id: string; strategy_id: number } }
  | { Outcome: { query_id: string; success: boolean } };

export interface ActionEvent {
  action_name: string;
  parameters: any;
  outcome: ActionOutcome;
  duration_ns: number;
}

export interface ObservationEvent {
  observation_type: string;
  data: any;
  confidence: number;
  source: string;
}

export interface CognitiveEvent {
  process_type: CognitiveType;
  input: any;
  output: any;
  reasoning_trace: string[];
}

export interface CommunicationEvent {
  message_type: string;
  sender: AgentId;
  recipient: AgentId;
  content: any;
}

export type ActionOutcome =
  | { Success: { result: any } }
  | { Failure: { error: string; error_code: number } }
  | { Partial: { result: any; issues: string[] } };

export type CognitiveType =
  | "GoalFormation"
  | "Planning"
  | "Reasoning"
  | "MemoryRetrieval"
  | "LearningUpdate";

// Spatial context
export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface SpatialContext {
  location: [number, number, number]; // x, y, z coordinates
  bounds: BoundingBox | null;
  reference_frame: string;
}

// Temporal context
export interface TimeOfDay {
  hour: number;
  minute: number;
  timezone: string;
}

export interface Deadline {
  goal_id: number;
  timestamp: number;
  priority: number;
}

export interface TemporalPattern {
  pattern_name: string;
  frequency: number; // Duration in nanoseconds
  phase: number;
}

export interface TemporalContext {
  time_of_day: TimeOfDay | null;
  deadlines: Deadline[];
  patterns: TemporalPattern[];
}

// Environment state
export interface EnvironmentState {
  variables: Record<string, any>;
  spatial: SpatialContext | null;
  temporal: TemporalContext;
}

// Goals
export interface Goal {
  id: number;
  description: string;
  priority: number;
  deadline: number | null;
  progress: number;
  subgoals: number[];
}

// Resources
export interface ComputationalResources {
  cpu_percent: number;
  memory_bytes: number;
  storage_bytes: number;
  network_bandwidth: number;
}

export interface ResourceAvailability {
  available: boolean;
  capacity: number;
  current_usage: number;
  estimated_cost: number | null;
}

export interface ResourceState {
  computational: ComputationalResources;
  external: Record<string, ResourceAvailability>;
}

// Event context
export interface EventContext {
  environment: EnvironmentState;
  active_goals: Goal[];
  resources: ResourceState;
  fingerprint?: ContextHash; // Optional - auto-computed by server if not provided
  embeddings: number[] | null;
}

// ============================================================================
// Request/Response Types
// ============================================================================

export interface ProcessEventRequest {
  event: Event;
}

export interface ProcessEventResponse {
  success: boolean;
  nodes_created: number;
  patterns_detected: number;
  processing_time_ms: number;
}

export interface PaginationQuery {
  limit?: number;
}

export interface ActionSuggestionsQuery {
  context_hash: ContextHash;
  last_action_node?: number;
  limit?: number;
}

export interface MemoryResponse {
  id: number;
  agent_id: AgentId;
  session_id: SessionId;
  strength: number;
  relevance_score: number;
  access_count: number;
  formed_at: number;
  last_accessed: number;
  context_hash: ContextHash;
  context: EventContext;
  outcome: string;
  memory_type: string;
}

export interface ContextMemoriesRequest {
  context: EventContext;
  limit?: number;
  min_similarity?: number;
  agent_id?: AgentId;
  session_id?: SessionId;
}

export interface StrategyResponse {
  id: number;
  name: string;
  agent_id: AgentId;
  quality_score: number;
  success_count: number;
  failure_count: number;
  reasoning_steps: ReasoningStepResponse[];
  strategy_type: string;
  support_count: number;
  expected_success: number;
  expected_cost: number;
  expected_value: number;
  confidence: number;
  goal_bucket_id: number;
  behavior_signature: string;
  precondition: string;
  action_hint: string;
}

export interface SimilarStrategyResponse {
  score: number;
  id: number;
  name: string;
  agent_id: AgentId;
  quality_score: number;
  success_count: number;
  failure_count: number;
  reasoning_steps: ReasoningStepResponse[];
  strategy_type: string;
  support_count: number;
  expected_success: number;
  expected_cost: number;
  expected_value: number;
  confidence: number;
  goal_bucket_id: number;
  behavior_signature: string;
  precondition: string;
  action_hint: string;
}

export interface StrategySimilarityRequest {
  goal_ids?: number[];
  tool_names?: string[];
  result_types?: string[];
  context_hash?: ContextHash;
  agent_id?: AgentId;
  limit?: number;
  min_score?: number;
}

export interface ReasoningStepResponse {
  description: string;
  sequence_order: number;
}

export interface ActionSuggestionResponse {
  action_name: string;
  success_probability: number;
  evidence_count: number;
  reasoning: string;
}

export interface EpisodeResponse {
  id: number;
  agent_id: AgentId;
  event_count: number;
  significance: number;
  outcome: string | null;
}

export interface StatsResponse {
  total_events_processed: number;
  total_nodes_created: number;
  total_episodes_detected: number;
  total_memories_formed: number;
  total_strategies_extracted: number;
  total_reinforcements_applied: number;
  average_processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  is_healthy: boolean;
  node_count: number;
  edge_count: number;
  processing_rate: number;
}

export interface ErrorResponse {
  error: string;
  details?: string;
}
