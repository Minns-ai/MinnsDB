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
    GraphQuery,
    GraphContextQuery,
    GraphResponse,
    AnalyticsResponse,
    ClaimSearchRequest,
    ClaimSearchResponse,
    GraphNodeQueryRequest,
    GraphNodeQueryResponse,
    GraphTraverseQuery,
    GraphTraverseResponse,
    EventType,
    Goal,
    SessionId
  } from "./types.js";
  
  export * from "./types.js";
export * from "./intent_registry.js";
export * from "./intent_sidecar.js";
  export * from "./intent_registry.js";
  export * from "./intent_sidecar.js";
  
  // ============================================================================
  // Client Configuration
  // ============================================================================
  
  export interface EventGraphDBClientConfig {
    baseUrl?: string;
    timeout?: number;
    headers?: Record<string, string>;
    onTelemetry?: (data: TelemetryData) => void;
    enableDefaultTelemetry?: boolean;
    /**
     * Maximum payload size in bytes (checked after serialization). 
     * Default: 1MB (1,048,576 bytes).
     */
    maxPayloadSize?: number;
    /**
     * Maximum number of events allowed in the local queue. 
     * When exceeded, `enqueue()` will throw a QueueFull error.
     * Default: 1000.
     */
    maxQueueSize?: number;
    /**
     * If true, `processEvent()` will return immediately and process in the background.
     * Use this to make all event processing low-latency by default.
     */
    defaultAsync?: boolean;
    /**
     * If true, events will be buffered and sent in batches.
     */
    autoBatch?: boolean;
    /**
     * Maximum time (ms) to wait before flushing the batch queue. Default: 100ms.
     */
    batchInterval?: number;
    /**
     * Maximum number of events to buffer before forcing a flush. Default: 10.
     */
    batchMaxSize?: number;
  }

  export interface TelemetryData {
    type: "request" | "error" | "intent_parse" | "background_process" | "batch_flush";
    path?: string;
    method?: string;
    duration_ms?: number;
    statusCode?: number;
    error?: string;
    tokenCount?: number;
    metadata?: Record<string, any>;
    agent_id?: AgentId;
    session_id?: SessionId;
  }
  
  const DEFAULT_CONFIG: Required<Omit<EventGraphDBClientConfig, "onTelemetry">> = {
    baseUrl: "http://127.0.0.1:3000",
    timeout: 30000,
    headers: {
      "Content-Type": "application/json",
    },
    enableDefaultTelemetry: false,
    maxPayloadSize: 1024 * 1024,
    maxQueueSize: 1000,
    defaultAsync: false,
    autoBatch: false,
    batchInterval: 100,
    batchMaxSize: 10,
  };

  /**
   * Safe JSON stringify that handles circular references and dangerous keys.
   */
  function safeStringify(obj: unknown): string {
    const cache = new WeakSet();
    return JSON.stringify(obj, (key, value) => {
      // 1. Prevent Prototype Pollution
      if (key === "__proto__" || key === "constructor") return undefined;

      // 2. Handle Circular References
      if (typeof value === "object" && value !== null) {
        if (cache.has(value)) return "[Circular]";
        cache.add(value);
      }
      return value;
    });
  }

  // ============================================================================
  // Event Builder
  // ============================================================================

  export interface LocalAck {
    success: true;
    queued: boolean;
    eventId: string;
  }

  export interface EventBuilderConfig {
    agentId?: AgentId;
    sessionId?: SessionId;
    enableSemantic?: boolean;
  }

  export class EventBuilder {
    private client: EventGraphDBClient;
    private agentType: string;
    private config: EventBuilderConfig;
    
    private eventPayload?: EventType;
    private contextVariables: Record<string, unknown> = {};
    private goals: Goal[] = [];
    private metadata: Record<string, unknown> = {};
    private causality: string[] = [];

    constructor(client: EventGraphDBClient, agentType: string, config: EventBuilderConfig = {}) {
      this.client = client;
      this.agentType = agentType;
      this.config = config;
    }

    /**
     * Define an Action taken by the agent.
     */
    action(name: string, params: Record<string, unknown>): this {
      this.eventPayload = {
        Action: {
          action_name: name,
          parameters: params,
          outcome: { Success: { result: "ok" } },
          duration_ns: 0
        }
      };
      return this;
    }

    /**
     * Attach an outcome to the previously defined action.
     * Throws if no action has been defined yet.
     */
    outcome(result: unknown): this {
      if (!this.eventPayload || !("Action" in this.eventPayload)) {
        throw new Error("outcome() requires an action() to be defined first in the builder chain.");
      }
      this.eventPayload.Action.outcome = { Success: { result } };
      return this;
    }

    /**
     * Define an Observation from the environment.
     */
    observation(type: string, data: unknown, options: { confidence?: number; source?: string } = {}): this {
      this.eventPayload = {
        Observation: {
          observation_type: type,
          data,
          confidence: options.confidence ?? 1.0,
          source: options.source ?? "environment"
        }
      };
      return this;
    }

    /**
     * Define a state-change event. Shortcut that builds an Observation event
     * with canonical metadata keys for structured memory auto-detection.
     */
    stateChange(entity: string, newState: string, options: { oldState?: string; trigger?: string } = {}): this {
      this.eventPayload = {
        Observation: {
          observation_type: "state_update",
          data: { entity, new_state: newState, old_state: options.oldState },
          confidence: 1.0,
          source: "agent"
        }
      };
      this.metadata = {
        ...this.metadata,
        entity,
        new_state: newState,
        ...(options.oldState ? { old_state: options.oldState } : {}),
        ...(options.trigger ? { trigger: options.trigger } : {}),
      };
      return this;
    }

    /**
     * Define a transaction event. Shortcut that builds an Action event
     * with canonical metadata keys for structured memory ledger auto-detection.
     */
    transaction(from: string, to: string, amount: number, options: { direction?: "Credit" | "Debit"; description?: string } = {}): this {
      this.eventPayload = {
        Action: {
          action_name: "transaction",
          parameters: { from, to, amount },
          outcome: { Success: { result: "ok" } },
          duration_ns: 0
        }
      };
      this.metadata = {
        ...this.metadata,
        from,
        to,
        amount,
        transaction: true,
        ...(options.direction ? { direction: options.direction } : {}),
        ...(options.description ? { description: options.description } : {}),
      };
      return this;
    }

    /**
     * Define a Context/Text event for claim extraction.
     */
    context(text: string, type: string = "general"): this {
      this.eventPayload = {
        Context: {
          text,
          context_type: type,
          language: "en"
        }
      };
      return this;
    }

    /**
     * Add environmental state (runtime variables).
     */
    state(variables: Record<string, unknown>): this {
      this.contextVariables = { ...this.contextVariables, ...variables };
      return this;
    }

    /**
     * Add a goal with a priority (1-5).
     */
    goal(text: string, priority: 1 | 2 | 3 | 4 | 5 = 3, progress: number = 0): this {
      this.goals.push({
        id: Math.floor(Math.random() * 1000000),
        description: text,
        priority: priority / 5, 
        progress,
        deadline: null,
        subgoals: []
      });
      return this;
    }

    /**
     * Link to a previous event (causality).
     */
    causedBy(parentId: string): this {
      this.causality.push(parentId);
      return this;
    }

    /**
     * Build the final nested Event object.
     */
    build(): Event {
      if (!this.eventPayload) {
        throw new Error("Event payload (action, observation, or context) is required.");
      }

      return {
        agent_id: this.config.agentId ?? 0,
        session_id: this.config.sessionId ?? 0,
        agent_type: this.agentType,
        event_type: this.eventPayload,
        causality_chain: this.causality,
        metadata: this.metadata as any,
        context: {
          environment: {
            variables: this.contextVariables,
            spatial: null,
            temporal: { time_of_day: null, deadlines: [], patterns: [] }
          },
          active_goals: this.goals,
          resources: {
            computational: { cpu_percent: 0, memory_bytes: 0, storage_bytes: 0, network_bandwidth: 0 },
            external: {}
          },
          embeddings: null
        }
      };
    }

    /**
     * Build and submit the event. 
     * Waits for server response.
     */
    async send(): Promise<ProcessEventResponse> {
      return this.client.processEvent(this.build(), { 
        enableSemantic: this.config.enableSemantic 
      });
    }

    /**
     * Build and enqueue the event for background processing.
     * Returns a local acknowledgement receipt immediately.
     */
    async enqueue(): Promise<LocalAck> {
      const event = this.build();
      const eventId = event.id ?? "pending";
      
      this.client.processEvent(event, { 
        enableSemantic: this.config.enableSemantic,
        forceAsync: true 
      }).catch(() => {
        // Handled via telemetry
      });

      return {
        success: true,
        queued: true,
        eventId
      };
    }
  }

  // ============================================================================
  // EventGraphDB Client
  // ============================================================================
  
  export class EventGraphDBClient {
    private baseUrl: string;
    private timeout: number;
    private headers: Record<string, string>;
    private onTelemetry?: (data: TelemetryData) => void;
    private enableDefaultTelemetry: boolean;
    private maxPayloadSize: number;
    private maxQueueSize: number;
    private defaultAsync: boolean;
    private autoBatch: boolean;
    private batchInterval: number;
    private batchMaxSize: number;
    
    private eventBuffer: Event[] = [];
    private flushTimer: ReturnType<typeof setTimeout> | null = null;
  
    constructor(config: EventGraphDBClientConfig = {}) {
      this.baseUrl = config.baseUrl || DEFAULT_CONFIG.baseUrl;
      this.timeout = config.timeout || DEFAULT_CONFIG.timeout;
      this.headers = { ...DEFAULT_CONFIG.headers, ...config.headers };
      this.onTelemetry = config.onTelemetry;
      this.enableDefaultTelemetry = config.enableDefaultTelemetry ?? DEFAULT_CONFIG.enableDefaultTelemetry;
      this.maxPayloadSize = config.maxPayloadSize ?? DEFAULT_CONFIG.maxPayloadSize;
      this.maxQueueSize = config.maxQueueSize ?? DEFAULT_CONFIG.maxQueueSize;
      this.defaultAsync = config.defaultAsync ?? DEFAULT_CONFIG.defaultAsync;
      this.autoBatch = config.autoBatch ?? DEFAULT_CONFIG.autoBatch;
      this.batchInterval = config.batchInterval ?? DEFAULT_CONFIG.batchInterval;
      this.batchMaxSize = config.batchMaxSize ?? DEFAULT_CONFIG.batchMaxSize;
    }

    /**
     * Create a new EventBuilder for fluent event construction.
     */
    event(agentType: string, config: EventBuilderConfig = {}): EventBuilder {
      return new EventBuilder(this, agentType, config);
    }
  
    /**
     * Internal telemetry helper.
     */
    private emitTelemetry(data: TelemetryData) {
      if (this.onTelemetry) {
        try {
          this.onTelemetry(data);
        } catch (e) {
          // Internal logging only if telemetry fails
        }
      }

      if (this.enableDefaultTelemetry) {
        this.sendTelemetryToBackend(data);
      }
    }

    /**
     * Sends telemetry data to the /api/telemetry endpoint (Fire and Forget)
     */
    private sendTelemetryToBackend(data: TelemetryData): void {
      const url = `${this.baseUrl}/api/telemetry`;
      
      const body = safeStringify(data);
      if (body.length > this.maxPayloadSize) return;

      fetch(url, {
        method: "POST",
        headers: this.headers,
        body,
        signal: AbortSignal.timeout(2000),
      }).catch(() => {
        /* silently ignore telemetry delivery failures */
      });
    }
  
    /**
     * Make HTTP request with error handling and payload size guarding.
     */
    private async request<T>(
      method: string,
      path: string,
      body?: unknown
    ): Promise<T> {
      const url = `${this.baseUrl}${path}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      const startTime = Date.now();
  
      try {
        const serializedBody = body ? safeStringify(body) : undefined;

        if (serializedBody && serializedBody.length > this.maxPayloadSize) {
          throw new EventGraphDBError(
            `Payload size (${serializedBody.length} bytes) exceeds maximum allowed (${this.maxPayloadSize} bytes).`,
            413
          );
        }

        const response = await fetch(url, {
          method,
          headers: this.headers,
          body: serializedBody,
          signal: controller.signal,
        });
  
        clearTimeout(timeoutId);
        const duration_ms = Date.now() - startTime;
  
        if (!response.ok) {
          let errorMsg = `Request failed with status ${response.status}`;
          let details: string | undefined;

          try {
            const errorJson = await response.json() as ErrorResponse;
            errorMsg = errorJson.error || errorMsg;
            details = errorJson.details;
          } catch (e) {
            // If response is not JSON, try to get text
            try {
              const text = await response.text();
              if (text) errorMsg += `: ${text}`;
            } catch (inner) { /* ignore */ }
          }

          const error = new EventGraphDBError(
            errorMsg,
            response.status,
            details
          );
          
          this.emitTelemetry({
            type: "error",
            path,
            method,
            statusCode: response.status,
            error: error.message,
            duration_ms
          });

          throw error;
        }
  
        const data = await response.json() as T;

        this.emitTelemetry({
          type: "request",
          path,
          method,
          statusCode: response.status,
          duration_ms
        });

        return data;
      } catch (error) {
        clearTimeout(timeoutId);
        const duration_ms = Date.now() - startTime;
  
        let finalError: EventGraphDBError;
        if (error instanceof EventGraphDBError) {
          finalError = error;
        } else if (error instanceof Error && error.name === "AbortError") {
          finalError = new EventGraphDBError("Request timeout", 408);
        } else {
          finalError = new EventGraphDBError(error instanceof Error ? error.message : "Unknown error", 500);
        }

        this.emitTelemetry({
          type: "error",
          path,
          method,
          statusCode: finalError.statusCode,
          error: finalError.message,
          duration_ms
        });
  
        throw finalError;
      }
    }
  
    // ==========================================================================
    // Event Processing
    // ==========================================================================
  
    /**
     * Process a new event through the EventGraphDB system.
     * 
     * Behavior depends on client configuration:
     * - If `autoBatch` is on: Event is queued and a local receipt is returned.
     * - Else if `defaultAsync` or `options.forceAsync` is on: Request is fired in background and receipt is returned.
     * - Else: Waits for server response.
     */
    async processEvent(
      event: Event,
      options: { enableSemantic?: boolean; forceAsync?: boolean } = {}
    ): Promise<ProcessEventResponse> {
      if (this.autoBatch && !options.forceAsync) {
        if (this.eventBuffer.length >= this.maxQueueSize) {
          throw new EventGraphDBError("Local event queue is full.", 429);
        }

        this.eventBuffer.push(event);
        
        if (this.eventBuffer.length >= this.batchMaxSize) {
          this.flushEvents(options);
        } else if (!this.flushTimer) {
          this.flushTimer = setTimeout(() => this.flushEvents(options), this.batchInterval);
        }

        return this.createLocalAck(event.id ?? "queued", true);
      }

      if (this.defaultAsync || options.forceAsync) {
        this.request<ProcessEventResponse>("POST", "/api/events", {
          event,
          enable_semantic: options.enableSemantic,
        }).catch(err => {
          this.emitTelemetry({
            type: "background_process",
            error: err instanceof Error ? err.message : "Background event processing failed",
            metadata: { eventId: event.id ?? "unknown" }
          });
        });

        return this.createLocalAck(event.id ?? "pending", false);
      }

      return this.request<ProcessEventResponse>("POST", "/api/events", {
        event,
        enable_semantic: options.enableSemantic,
      });
    }

    /**
     * Batch process multiple events. Chunks large arrays into `batchMaxSize` requests.
     */
    async processEvents(
      events: Event[],
      options: { enableSemantic?: boolean; forceAsync?: boolean } = {}
    ): Promise<ProcessEventResponse> {
      if (events.length === 0) return this.createLocalAck("empty", false);

      for (let i = 0; i < events.length; i += this.batchMaxSize) {
        const chunk = events.slice(i, i + this.batchMaxSize);
        
        const promise = this.request<ProcessEventResponse>("POST", "/api/events/batch", {
          events: chunk,
          enable_semantic: options.enableSemantic,
        });

        if (this.defaultAsync || options.forceAsync) {
          promise.catch(err => {
            this.emitTelemetry({
              type: "background_process",
              error: err instanceof Error ? err.message : "Batch background processing failed"
            });
          });
        } else {
          await promise;
        }
      }

      return this.createLocalAck("batch", !!(this.defaultAsync || options.forceAsync));
    }

    private createLocalAck(eventId: string, queued: boolean): ProcessEventResponse {
      return {
        success: true,
        nodes_created: 0,
        patterns_detected: 0,
        processing_time_ms: 0,
        ...({ queued, eventId } as any)
      };
    }
  
    // ==========================================================================
    // Memory Queries
    // ==========================================================================
  
    /**
     * Get memories for a specific agent, sorted by strength.
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
     */
    async getStats(): Promise<StatsResponse> {
      return this.request<StatsResponse>("GET", "/api/stats");
    }
  
    /**
     * Search claims extracted from events via semantic memory.
     */
    async searchClaims(
      request: ClaimSearchRequest
    ): Promise<ClaimSearchResponse[]> {
      return this.request<ClaimSearchResponse[]>("POST", "/api/claims/search", {
        query_text: request.query_text,
        top_k: request.top_k ?? 5,
        min_similarity: request.min_similarity,
      });
    }
  
    /**
     * Get graph analytics including learning metrics.
     */
    async getAnalytics(): Promise<AnalyticsResponse> {
      return this.request<AnalyticsResponse>("GET", "/api/analytics");
    }
  
    /**
     * Get graph structure for visualization.
     */
    async getGraph(query: GraphQuery = {}): Promise<GraphResponse> {
      const params = new URLSearchParams();
      if (typeof query.limit === "number") params.set("limit", String(query.limit));
      if (typeof query.session_id === "number") params.set("session_id", String(query.session_id));
      if (typeof query.agent_type === "string" && query.agent_type) params.set("agent_type", query.agent_type);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      return this.request<GraphResponse>("GET", `/api/graph${suffix}`);
    }
  
    /**
     * Get context-anchored subgraph.
     */
    async getGraphByContext(query: GraphContextQuery): Promise<GraphResponse> {
      const params = new URLSearchParams({ context_hash: String(query.context_hash) });
      if (typeof query.limit === "number") params.set("limit", String(query.limit));
      if (typeof query.session_id === "number") params.set("session_id", String(query.session_id));
      if (typeof query.agent_type === "string" && query.agent_type) params.set("agent_type", query.agent_type);
      return this.request<GraphResponse>("GET", `/api/graph/context?${params.toString()}`);
    }
  
    /**
     * Direct node search for hard facts (IDs, member numbers).
     */
    async queryGraphNodes(request: GraphNodeQueryRequest): Promise<GraphNodeQueryResponse> {
      return this.request<GraphNodeQueryResponse>("POST", "/api/v1/graph/query", request);
    }
  
    /**
     * Traverse graph relationships from a starting node.
     */
    async traverseGraph(query: GraphTraverseQuery): Promise<GraphTraverseResponse> {
      const params = new URLSearchParams({ start: query.start });
      if (typeof query.max_depth === "number") params.set("max_depth", String(query.max_depth));
      if (Array.isArray(query.node_types) && query.node_types.length > 0) {
        params.set("node_types", query.node_types.join(","));
      }
      return this.request<GraphTraverseResponse>("GET", `/api/v1/graph/traverse?${params.toString()}`);
    }
  
    // ==========================================================================
    // Typed Event Shortcuts
    // ==========================================================================

    /**
     * Emit a state-change event. The server maps the fields to canonical metadata
     * keys and the pipeline auto-detects state transitions for structured memory.
     */
    async stateChange(params: {
      agentId: AgentId;
      agentType: string;
      sessionId: SessionId;
      entity: string;
      newState: string;
      oldState?: string;
      trigger?: string;
      extraMetadata?: Record<string, unknown>;
      enableSemantic?: boolean;
    }): Promise<ProcessEventResponse> {
      return this.request<ProcessEventResponse>("POST", "/api/events/state-change", {
        agent_id: params.agentId,
        agent_type: params.agentType,
        session_id: params.sessionId,
        entity: params.entity,
        new_state: params.newState,
        old_state: params.oldState,
        trigger: params.trigger,
        extra_metadata: params.extraMetadata ?? {},
        enable_semantic: params.enableSemantic ?? false,
      });
    }

    /**
     * Emit a transaction event. The server maps the fields to canonical metadata
     * keys and the pipeline auto-detects ledger entries for structured memory.
     */
    async transaction(params: {
      agentId: AgentId;
      agentType: string;
      sessionId: SessionId;
      from: string;
      to: string;
      amount: number;
      direction?: "Credit" | "Debit";
      description?: string;
      extraMetadata?: Record<string, unknown>;
      enableSemantic?: boolean;
    }): Promise<ProcessEventResponse> {
      return this.request<ProcessEventResponse>("POST", "/api/events/transaction", {
        agent_id: params.agentId,
        agent_type: params.agentType,
        session_id: params.sessionId,
        from: params.from,
        to: params.to,
        amount: params.amount,
        direction: params.direction,
        description: params.description,
        extra_metadata: params.extraMetadata ?? {},
        enable_semantic: params.enableSemantic ?? false,
      });
    }

    // ==========================================================================
    // System
    // ==========================================================================

    /**
     * Health check endpoint to verify system status.
     */
    async healthCheck(): Promise<HealthResponse> {
      return this.request<HealthResponse>("GET", "/api/health");
    }

    /**
     * Flushes the local event buffer.
     */
    public async flush(options: { enableSemantic?: boolean } = {}): Promise<void> {
      await this.flushEvents(options);
    }

    private async flushEvents(options: { enableSemantic?: boolean } = {}): Promise<void> {
      if (this.flushTimer) {
        clearTimeout(this.flushTimer);
        this.flushTimer = null;
      }

      if (this.eventBuffer.length === 0) return;

      const eventsToProcess = [...this.eventBuffer];
      this.eventBuffer = [];

      try {
        await this.processEvents(eventsToProcess, options);
        this.emitTelemetry({
          type: "batch_flush",
          metadata: { count: eventsToProcess.length }
        });
      } catch (err) {
        this.emitTelemetry({
          type: "error",
          error: err instanceof Error ? err.message : "Batch flush failed",
          metadata: { count: eventsToProcess.length }
        });
      }
    }
  }
  
  export class EventGraphDBError extends Error {
    public readonly statusCode: number;
    public readonly details?: string;
  
    constructor(message: string, statusCode: number, details?: string) {
      super(message);
      this.name = "EventGraphDBError";
      this.statusCode = statusCode;
      this.details = details;
  
      if ((Error as any).captureStackTrace) {
        (Error as any).captureStackTrace(this, EventGraphDBError);
      }
    }
  }
  
  export function createClient(
    config: EventGraphDBClientConfig = {}
  ): EventGraphDBClient {
    return new EventGraphDBClient(config);
  }
  
  export default EventGraphDBClient;
