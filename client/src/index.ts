import {
  Event,
  ProcessEventResponse,
  ContextMemoriesRequest,
  EventContext,
  MemoryResponse,
  StrategyResponse,
  StrategySimilarityRequest,
  SimilarStrategyResponse,
  ActionSuggestionResponse,
  EpisodeResponse,
  StatsResponse,
  HealthResponse,
  ErrorResponse,
  AgentId,
  ContextHash,
  PaginationQuery,
  ActionSuggestionsQuery,
} from "./types";

export * from "./types";

// ============================================================================
// Client Configuration
// ============================================================================

export interface EventGraphDBClientConfig {
  baseUrl?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

const DEFAULT_CONFIG: Required<EventGraphDBClientConfig> = {
  baseUrl: "http://127.0.0.1:3000",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
};

// ============================================================================
// EventGraphDB Client
// ============================================================================

export class EventGraphDBClient {
  private baseUrl: string;
  private timeout: number;
  private headers: Record<string, string>;

  constructor(config: EventGraphDBClientConfig = {}) {
    this.baseUrl = config.baseUrl || DEFAULT_CONFIG.baseUrl;
    this.timeout = config.timeout || DEFAULT_CONFIG.timeout;
    this.headers = { ...DEFAULT_CONFIG.headers, ...config.headers };
  }

  /**
   * Make HTTP request with error handling
   */
  private async request<T>(
    method: string,
    path: string,
    body?: any
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json() as ErrorResponse;
        throw new EventGraphDBError(
          error.error,
          response.status,
          error.details
        );
      }

      return await response.json() as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof EventGraphDBError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new EventGraphDBError("Request timeout", 408);
        }
        throw new EventGraphDBError(error.message, 500);
      }

      throw new EventGraphDBError("Unknown error occurred", 500);
    }
  }

  // ==========================================================================
  // Event Processing
  // ==========================================================================

  /**
   * Process a new event through the EventGraphDB system.
   * This triggers automatic episode detection, memory formation,
   * strategy extraction, and reinforcement learning.
   *
   * @param event - The event to process
   * @returns Processing result with statistics
   *
   * @example
   * const result = await client.processEvent({
   *   id: "evt_123",
   *   timestamp: Date.now() * 1000000,
   *   agent_id: 1,
   *   agent_type: "code-assistant",
   *   session_id: 42,
   *   event_type: {
   *     Action: {
   *       action_name: "fix_bug",
   *       parameters: { bug_type: "null_reference" },
   *       outcome: { Success: { result: { tests_pass: true } } },
   *       duration_ns: 1500000000,
   *     },
   *   },
   *   causality_chain: [],
   *   context: {
   *     task_description: "Fix null reference error",
   *     code_snapshot: "function foo() {...}",
   *     error_messages: ["TypeError: null"],
   *     recent_actions: [],
   *     environment_state: {},
   *   },
   *   metadata: {},
   * });
   *
   * console.log(`Created ${result.nodes_created} nodes`);
   * console.log(`Detected ${result.patterns_detected} patterns`);
   */
  async processEvent(event: Event): Promise<ProcessEventResponse> {
    return this.request<ProcessEventResponse>("POST", "/api/events", {
      event,
    });
  }

  // ==========================================================================
  // Memory Queries
  // ==========================================================================

  /**
   * Get memories for a specific agent, sorted by strength.
   *
   * @param agentId - The agent ID
   * @param limit - Maximum number of memories to return (default: 10)
   * @returns Array of memories
   *
   * @example
   * const memories = await client.getAgentMemories(1, 5);
   * memories.forEach(m => {
   *   console.log(`Memory ${m.id}: strength=${m.strength}, accessed=${m.access_count} times`);
   * });
   */
  async getAgentMemories(
    agentId: AgentId,
    limit: number = 10
  ): Promise<MemoryResponse[]> {
    const query = new URLSearchParams({ limit: limit.toString() });
    return this.request<MemoryResponse[]>(
      "GET",
      `/api/memories/agent/${agentId}?${query}`
    );
  }

  /**
   * Retrieve memories for a similar context with optional filters.
   *
   * @param context - Current context snapshot
   * @param request - Optional retrieval settings and filters
   * @returns Array of context-relevant memories
   */
  async getContextMemories(
    context: EventContext,
    request: Omit<ContextMemoriesRequest, "context"> = {}
  ): Promise<MemoryResponse[]> {
    const payload: ContextMemoriesRequest = {
      context,
      limit: request.limit ?? 10,
      min_similarity: request.min_similarity,
      agent_id: request.agent_id,
      session_id: request.session_id,
    };

    return this.request<MemoryResponse[]>(
      "POST",
      "/api/memories/context",
      payload
    );
  }

  // ==========================================================================
  // Strategy Queries
  // ==========================================================================

  /**
   * Get strategies learned by a specific agent, sorted by quality score.
   *
   * @param agentId - The agent ID
   * @param limit - Maximum number of strategies to return (default: 10)
   * @returns Array of strategies with reasoning steps
   *
   * @example
   * const strategies = await client.getAgentStrategies(1, 5);
   * strategies.forEach(s => {
   *   console.log(`${s.name}: quality=${s.quality_score}, success_rate=${
   *     s.success_count / (s.success_count + s.failure_count)
   *   }`);
   *   s.reasoning_steps.forEach(step => {
   *     console.log(`  ${step.sequence_order}. ${step.description}`);
   *   });
   * });
   */
  async getAgentStrategies(
    agentId: AgentId,
    limit: number = 10
  ): Promise<StrategyResponse[]> {
    const query = new URLSearchParams({ limit: limit.toString() });
    return this.request<StrategyResponse[]>(
      "GET",
      `/api/strategies/agent/${agentId}?${query}`
    );
  }

  /**
   * Find strategies similar to a signature (goal/tool/result/context).
   */
  async getSimilarStrategies(
    request: StrategySimilarityRequest
  ): Promise<SimilarStrategyResponse[]> {
    const payload: StrategySimilarityRequest = {
      goal_ids: request.goal_ids,
      tool_names: request.tool_names,
      result_types: request.result_types,
      context_hash: request.context_hash,
      agent_id: request.agent_id,
      limit: request.limit ?? 10,
      min_score: request.min_score,
    };

    return this.request<SimilarStrategyResponse[]>(
      "POST",
      "/api/strategies/similar",
      payload
    );
  }

  // ==========================================================================
  // Policy Guide (Action Suggestions)
  // ==========================================================================

  /**
   * Get action suggestions based on context (Policy Guide feature).
   * Returns recommended next actions with success probabilities.
   *
   * @param contextHash - Hash of the current context
   * @param lastActionNode - Optional node ID of last action taken
   * @param limit - Maximum number of suggestions (default: 5)
   * @returns Array of action suggestions with probabilities
   *
   * @example
   * const suggestions = await client.getActionSuggestions(
   *   "ctx_abc123",
   *   undefined,
   *   5
   * );
   *
   * console.log("What should I do next?");
   * suggestions.forEach((s, i) => {
   *   console.log(`${i + 1}. ${s.action_name}: ${(s.success_probability * 100).toFixed(1)}% success`);
   *   console.log(`   Evidence: ${s.evidence_count} cases`);
   *   console.log(`   ${s.reasoning}`);
   * });
   */
  async getActionSuggestions(
    contextHash: ContextHash,
    lastActionNode?: number,
    limit: number = 5
  ): Promise<ActionSuggestionResponse[]> {
    const params: Record<string, string> = {
      context_hash: contextHash.toString(),
      limit: limit.toString(),
    };

    if (lastActionNode !== undefined) {
      params.last_action_node = lastActionNode.toString();
    }

    const query = new URLSearchParams(params);
    return this.request<ActionSuggestionResponse[]>(
      "GET",
      `/api/suggestions?${query}`
    );
  }

  // ==========================================================================
  // Episode Queries
  // ==========================================================================

  /**
   * Get completed episodes detected by the system.
   *
   * @param limit - Maximum number of episodes to return (default: 10)
   * @returns Array of episodes with metadata
   *
   * @example
   * const episodes = await client.getEpisodes(10);
   * episodes.forEach(e => {
   *   console.log(`Episode ${e.id}: ${e.event_count} events, significance=${e.significance}`);
   *   if (e.outcome) {
   *     console.log(`  Outcome: ${e.outcome}`);
   *   }
   * });
   */
  async getEpisodes(limit: number = 10): Promise<EpisodeResponse[]> {
    const query = new URLSearchParams({ limit: limit.toString() });
    return this.request<EpisodeResponse[]>("GET", `/api/episodes?${query}`);
  }

  // ==========================================================================
  // System Queries
  // ==========================================================================

  /**
   * Get system-wide statistics about EventGraphDB.
   *
   * @returns Statistics including event counts, memory stats, etc.
   *
   * @example
   * const stats = await client.getStats();
   * console.log(`Total events: ${stats.total_events_processed}`);
   * console.log(`Memories formed: ${stats.total_memories_formed}`);
   * console.log(`Strategies extracted: ${stats.total_strategies_extracted}`);
   * console.log(`Avg processing time: ${stats.average_processing_time_ms}ms`);
   */
  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>("GET", "/api/stats");
  }

  /**
   * Health check endpoint to verify system status.
   *
   * @returns Health information including version and metrics
   *
   * @example
   * const health = await client.healthCheck();
   * if (health.is_healthy) {
   *   console.log(`System healthy: v${health.version}`);
   *   console.log(`Nodes: ${health.node_count}, Edges: ${health.edge_count}`);
   * } else {
   *   console.log(`System degraded: ${health.status}`);
   * }
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/api/health");
  }
}

// ============================================================================
// Error Class
// ============================================================================

export class EventGraphDBError extends Error {
  public readonly statusCode: number;
  public readonly details?: string;

  constructor(message: string, statusCode: number, details?: string) {
    super(message);
    this.name = "EventGraphDBError";
    this.statusCode = statusCode;
    this.details = details;

    // Maintains proper stack trace for where error was thrown (V8 only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, EventGraphDBError);
    }
  }
}

// ============================================================================
// Convenience Function
// ============================================================================

/**
 * Create a new EventGraphDB client with the given configuration.
 *
 * @param config - Client configuration
 * @returns EventGraphDB client instance
 *
 * @example
 * const client = createClient({ baseUrl: "http://localhost:3000" });
 */
export function createClient(
  config: EventGraphDBClientConfig = {}
): EventGraphDBClient {
  return new EventGraphDBClient(config);
}

// Default export
export default EventGraphDBClient;
