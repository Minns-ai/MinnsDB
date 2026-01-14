// ============================================================================
// Core Types
// ============================================================================

export type AgentId = number;
export type ContextHash = string;
export type EventId = string;

// ============================================================================
// Event Types
// ============================================================================

export interface Event {
  id: EventId;
  timestamp: number;
  agent_id: AgentId;
  agent_type: string;
  session_id: number;
  event_type: EventType;
  causality_chain: EventId[];
  context: EventContext;
  metadata: Record<string, any>;
}

export type EventType =
  | { Action: ActionEvent }
  | { Observation: ObservationEvent }
  | { Cognitive: CognitiveEvent };

export interface ActionEvent {
  action_name: string;
  parameters: any;
  outcome: ActionOutcome;
  duration_ns: number;
}

export interface ObservationEvent {
  observation_type: string;
  data: any;
  source: string;
}

export interface CognitiveEvent {
  process_type: CognitiveType;
  input: any;
  output: any;
  reasoning_trace: string[];
}

export type ActionOutcome =
  | { Success: { result: any } }
  | { Failure: { error: string } }
  | { Partial: { progress: number } };

export type CognitiveType =
  | "Reasoning"
  | "Planning"
  | "Learning"
  | "Reflection";

export interface EventContext {
  task_description: string;
  code_snapshot: string;
  error_messages: string[];
  recent_actions: string[];
  environment_state: Record<string, any>;
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
  strength: number;
  relevance_score: number;
  access_count: number;
  created_at: number;
  last_accessed: number;
}

export interface StrategyResponse {
  id: number;
  name: string;
  agent_id: AgentId;
  quality_score: number;
  success_count: number;
  failure_count: number;
  reasoning_steps: ReasoningStepResponse[];
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
