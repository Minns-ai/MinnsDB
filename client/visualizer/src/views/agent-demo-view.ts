// Agent Demo View
// Browser-side OpenAI demo for event -> graph -> memory -> strategy

import { EventGraphDBClient } from '../../../dist/index.js';

type DemoTask = {
  id: string;
  title: string;
  description: string;
  goalId: number;
  script: string;
};

type AgentPlan = {
  tool: string;
  action_name: string;
  parameters: Record<string, any>;
  reasoning: string[];
  result_summary: string;
  revised_script?: string;
};

export class AgentDemoView {
  private client: EventGraphDBClient;
  private tasks: DemoTask[] = [
    {
      id: 'fix-script',
      title: 'Fix broken script',
      description: 'A script fails with "undefined is not a function". Provide a fix.',
      goalId: 101,
      script: `function run(items) {
  const total = items.sum();
  return total * 2;
}

console.log(run([1, 2, 3]));`
    },
    {
      id: 'refactor',
      title: 'Refactor to classes',
      description: 'Convert a module to a class-based design without changing behavior.',
      goalId: 102,
      script: `export function add(a, b) {
  return a + b;
}

export function multiply(a, b) {
  return a * b;
}`
    }
  ];
  private lastContext: any = null;
  private lastTask: DemoTask | null = null;
  private lastPlan: AgentPlan | null = null;
  private running = false;
  private lastEventId?: number;
  private lastQueryId?: string;
  private lastMemoryIds: number[] = [];
  private lastStrategyIds: number[] = [];

  constructor(client: EventGraphDBClient) {
    this.client = client;
  }

  async render() {
    const container = document.getElementById('agent-demo-view');
    if (!container) return;

    container.innerHTML = `
      <div class="view-header">
        <h2>Agent Demo</h2>
        <div class="view-controls">
          <button id="demo-refresh-graph" class="btn btn-secondary">Refresh Graph</button>
        </div>
      </div>
      <div class="demo-panel">
        <div class="demo-grid">
          <div class="demo-section">
            <h3>OpenAI Settings</h3>
            <label class="demo-label">API Key</label>
            <input type="password" id="demo-api-key" class="demo-input" placeholder="sk-..." />
            <label class="demo-label">Model</label>
            <input type="text" id="demo-model" class="demo-input" value="gpt-4o-mini" />
            <label class="demo-label">Agent ID</label>
            <input type="number" id="demo-agent-id" class="demo-input" value="1" />
            <label class="demo-label">Session ID</label>
            <input type="number" id="demo-session-id" class="demo-input" value="${Date.now()}" />
          </div>
          <div class="demo-section">
            <h3>Task</h3>
            <label class="demo-label">Select Task</label>
            <select id="demo-task" class="demo-input">
              ${this.tasks.map(task => `<option value="${task.id}">${task.title}</option>`).join('')}
            </select>
            <label class="demo-label">Script (editable)</label>
            <textarea id="demo-script" class="demo-input demo-textarea" rows="8">${this.tasks[0].script}</textarea>
            <label class="demo-label">User Constraint (optional)</label>
            <input type="text" id="demo-constraint" class="demo-input" placeholder="e.g. always use classes" />
            <div class="demo-actions">
              <button id="demo-run" class="btn">Run Agent</button>
              <button id="demo-variant" class="btn btn-secondary">Run Variant</button>
            </div>
          </div>
          <div class="demo-section">
            <h3>Feedback</h3>
            <label class="demo-label">What should the agent learn?</label>
            <input type="text" id="demo-feedback-text" class="demo-input" placeholder="e.g. avoid pattern X" />
            <div class="demo-actions">
              <button id="demo-success" class="btn">Mark Success</button>
              <button id="demo-failure" class="btn btn-secondary">Mark Failure</button>
            </div>
          </div>
        </div>
        <div class="demo-section">
          <h3>Demo Log</h3>
          <div id="demo-log" class="demo-log"></div>
        </div>
      </div>
    `;

    this.setupEventListeners();
  }

  async refresh() {
    // No-op; demo is user-driven
  }

  private setupEventListeners() {
    const runButton = document.getElementById('demo-run');
    const variantButton = document.getElementById('demo-variant');
    const successButton = document.getElementById('demo-success');
    const failureButton = document.getElementById('demo-failure');
    const refreshButton = document.getElementById('demo-refresh-graph');
    const taskSelect = document.getElementById('demo-task') as HTMLSelectElement | null;

    runButton?.addEventListener('click', () => this.runAgent(false));
    variantButton?.addEventListener('click', () => this.runAgent(true));
    successButton?.addEventListener('click', () => this.submitFeedback(true));
    failureButton?.addEventListener('click', () => this.submitFeedback(false));
    refreshButton?.addEventListener('click', () => (window as any).visualizer?.refresh());
    taskSelect?.addEventListener('change', () => this.updateScriptForTask());
  }

  private async runAgent(isVariant: boolean) {
    if (this.running) return;
    this.running = true;
    this.log(`Starting agent (${isVariant ? 'variant' : 'base'})...`);
    this.lastEventId = undefined;

    const apiKey = (document.getElementById('demo-api-key') as HTMLInputElement)?.value.trim();
    const model = (document.getElementById('demo-model') as HTMLInputElement)?.value.trim() || 'gpt-4o-mini';
    if (!apiKey) {
      this.log('Missing API key.');
      this.running = false;
      return;
    }

    const task = this.getSelectedTask();
    if (!task) {
      this.log('Select a task.');
      this.running = false;
      return;
    }

    const constraint = (document.getElementById('demo-constraint') as HTMLInputElement)?.value.trim();
    const script = this.getScriptInput();
    const context = this.buildContext(task, isVariant, constraint || undefined, script);
    this.lastContext = context;
    this.lastTask = task;

    this.log('Sending observation event...');
    await this.emitObservation(task, context, script);
    this.log('Observation event sent.');

    this.log('Fetching prior memories/strategies...');
    this.lastQueryId = `${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    this.lastMemoryIds = [];
    this.lastStrategyIds = [];
    const prior = await this.fetchPriorKnowledge(task, context, this.lastQueryId);
    this.log('Calling OpenAI...');
    const plan = await this.requestAgentPlan(apiKey, model, task, script, constraint, isVariant, prior);
    if (!plan) {
      this.running = false;
      return;
    }
    this.lastPlan = plan;

    this.log('Sending reasoning event...');
    await this.emitReasoning(plan.reasoning, context);
    this.log('Sending action event...');
    await this.emitAction(plan, context);
    if (plan.revised_script) {
      this.applyRevisedScript(plan.revised_script);
    }

    this.log(`Agent output: ${plan.result_summary}`);
    this.log('Waiting for feedback...');
    this.running = false;
  }

  private getSelectedTask(): DemoTask | null {
    const selected = (document.getElementById('demo-task') as HTMLSelectElement)?.value;
    return this.tasks.find(task => task.id === selected) || null;
  }

  private buildContext(task: DemoTask, isVariant: boolean, constraint?: string, script?: string) {
    const fingerprint = this.computeFingerprint({
      task: task.id,
      variant: isVariant,
      constraint: constraint || '',
      script: script || ''
    });

    return {
      environment: {
        variables: {
          task: task.id,
          variant: isVariant ? 'variant' : 'base',
          constraint: constraint || '',
          script_preview: (script || '').slice(0, 80),
          source: 'agent_demo'
        },
        spatial: null,
        temporal: {
          time_of_day: null,
          deadlines: [],
          patterns: []
        }
      },
      active_goals: [
        {
          id: task.goalId,
          description: task.description,
          priority: 0.8,
          deadline: null,
          progress: 0.0,
          subgoals: []
        }
      ],
      resources: {
        computational: {
          cpu_percent: 15,
          memory_bytes: 1024 * 1024 * 1024,
          storage_bytes: 20 * 1024 * 1024 * 1024,
          network_bandwidth: 1024 * 1024
        },
        external: {}
      },
      fingerprint,
      embeddings: null
    };
  }

  private async requestAgentPlan(
    apiKey: string,
    model: string,
    task: DemoTask,
    script: string,
    constraint?: string,
    isVariant?: boolean,
    prior?: { memories: string[]; strategies: string[] }
  ): Promise<AgentPlan | null> {
    const toolList = ['code_writer', 'refactorer', 'explainer'];
    const prompt = `
You are a coding agent demo. Return JSON only.
Task: ${task.description}
Script:
${script}
Variant: ${isVariant ? 'yes' : 'no'}
User constraint: ${constraint || 'none'}
Known memories: ${prior?.memories.length ? prior.memories.join(' | ') : 'none'}
Known strategies: ${prior?.strategies.length ? prior.strategies.join(' | ') : 'none'}
Choose one tool from: ${toolList.join(', ')}.
Output JSON with fields:
tool, action_name, parameters (object), reasoning (array of strings), result_summary, revised_script.
`;

    let response: Response;
    try {
      response = await Promise.race([
        fetch('https://api.openai.com/v1/responses', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model,
            input: prompt,
            temperature: 0.3
          })
        }),
        new Promise<Response>((_, reject) =>
          setTimeout(() => reject(new Error('OpenAI request timed out')), 20000)
        )
      ]);
    } catch (error) {
      this.log(`OpenAI request failed: ${error instanceof Error ? error.message : 'unknown error'}`);
      return null;
    }

    if (!response.ok) {
      const text = await response.text();
      this.log(`OpenAI error (${response.status}): ${text}`);
      return null;
    }

    const data = await response.json();
    const outputText = this.extractOutputText(data);
    const normalizedText = this.normalizeJsonOutput(outputText);

    try {
      const parsed = JSON.parse(normalizedText);
      return {
        tool: parsed.tool || 'code_writer',
        action_name: parsed.action_name || 'propose_fix',
        parameters: parsed.parameters || {},
        reasoning: Array.isArray(parsed.reasoning) ? parsed.reasoning : ['Reasoning unavailable'],
        result_summary: parsed.result_summary || 'No summary provided',
        revised_script: parsed.revised_script
      };
    } catch (error) {
      this.log('Failed to parse OpenAI output as JSON.');
      this.log(outputText);
      return null;
    }
  }

  private extractOutputText(data: any): string {
    if (data.output_text) return data.output_text;
    if (Array.isArray(data.output) && data.output.length > 0) {
      const content = data.output[0].content || [];
      const textItem = content.find((item: any) => item.type === 'output_text');
      if (textItem?.text) return textItem.text;
      const fallback = content.find((item: any) => item.text);
      if (fallback?.text) return fallback.text;
    }
    return '';
  }

  private normalizeJsonOutput(text: string) {
    const trimmed = text.trim();
    if (trimmed.startsWith('```')) {
      return trimmed.replace(/^```[a-zA-Z]*\s*/m, '').replace(/```$/m, '').trim();
    }
    return trimmed;
  }

  private async emitObservation(task: DemoTask, context: any, script: string) {
    await this.sendEvent({
      event_type: {
        Observation: {
          observation_type: 'task_received',
          data: { task: task.description, script },
          confidence: 0.9,
          source: 'agent_demo'
        }
      },
      context,
      metadata: {
        tool_name: 'observer',
        task_id: task.id,
        script
      }
    });
  }

  private async emitReasoning(reasoning: string[], context: any) {
    await this.sendEvent({
      event_type: {
        Cognitive: {
          process_type: 'Reasoning',
          input: { prompt: 'Plan the solution' },
          output: { reasoning },
          reasoning_trace: reasoning
        }
      },
      context,
      metadata: {
        tool_name: 'explainer'
      }
    });
  }

  private async emitAction(plan: AgentPlan, context: any) {
    await this.sendEvent({
      event_type: {
        Action: {
          action_name: plan.action_name,
          parameters: plan.parameters,
          outcome: {
            Partial: {
              result: {
                summary: plan.result_summary
              },
              issues: ['pending_user_feedback']
            }
          },
          duration_ns: 1_000_000
        }
      },
      context,
      metadata: {
        tool_name: plan.tool,
        result_summary: plan.result_summary
      }
    });

    if (this.lastQueryId && this.lastStrategyIds.length) {
      await this.sendEvent({
        event_type: {
          Learning: {
            StrategyUsed: {
              query_id: this.lastQueryId,
              strategy_id: this.lastStrategyIds[0]
            }
          }
        },
        context,
        metadata: {}
      });
    }

    if (this.lastQueryId && this.lastMemoryIds.length) {
      await this.sendEvent({
        event_type: {
          Learning: {
            MemoryUsed: {
              query_id: this.lastQueryId,
              memory_id: this.lastMemoryIds[0]
            }
          }
        },
        context,
        metadata: {}
      });
    }
  }

  private async submitFeedback(success: boolean) {
    if (!this.lastContext || !this.lastTask || !this.lastPlan) {
      this.log('Run the agent first.');
      return;
    }

    const feedbackText = (document.getElementById('demo-feedback-text') as HTMLInputElement)?.value.trim();
    const outcome = success
      ? { Success: { result: { feedback: feedbackText || 'approved' } } }
      : { Failure: { error: feedbackText || 'user_rejected', error_code: 1 } };

    const feedbackContext = {
      ...this.lastContext,
      environment: {
        ...this.lastContext.environment,
        variables: {
          ...this.lastContext.environment.variables,
          feedback: feedbackText || (success ? 'approved' : 'user_rejected')
        }
      }
    };

    await this.sendEvent({
      event_type: {
        Action: {
          action_name: 'user_feedback',
          parameters: {
            task_id: this.lastTask.id,
            tool: this.lastPlan.tool
          },
          outcome,
          duration_ns: 1_000_000
        }
      },
      context: feedbackContext,
      metadata: {
        tool_name: 'user_feedback',
        feedback: {
          success,
          constraint: feedbackText || null
        },
        significance: 0.9
      }
    });

    if (this.lastQueryId) {
      await this.sendEvent({
        event_type: {
          Learning: {
            Outcome: {
              query_id: this.lastQueryId,
              success
            }
          }
        },
        context: feedbackContext,
        metadata: {}
      });
    }

    await this.fetchInsights();
  }

  private async fetchPriorKnowledge(task: DemoTask, context: any, queryId: string) {
    const summaries: { memories: string[]; strategies: string[] } = {
      memories: [],
      strategies: []
    };

    try {
      const memories = await this.client.getContextMemories(context, {
        limit: 3,
        min_similarity: 0.3
      });
      summaries.memories = memories.map(memory => this.formatMemorySummary(memory));
      this.lastMemoryIds = memories.map(memory => memory.id);
      if (memories.length) {
        this.log(`Fetched ${memories.length} memories`);
        await this.sendEvent({
          event_type: {
            Learning: {
              MemoryRetrieved: {
                query_id: queryId,
                memory_ids: memories.map(memory => memory.id)
              }
            }
          },
          context,
          metadata: {}
        });
      }
    } catch (error) {
      this.log('Failed to fetch memories.');
    }

    try {
      const strategies = await this.client.getSimilarStrategies({
        goal_ids: [task.goalId],
        tool_names: [],
        result_types: [],
        limit: 3,
        min_score: 0.1
      });
      summaries.strategies = strategies.map(strategy => `Strategy#${strategy.id} score=${strategy.score.toFixed(2)}`);
      this.lastStrategyIds = strategies.map(strategy => strategy.id);
      if (strategies.length) {
        this.log(`Fetched ${strategies.length} strategies`);
        await this.sendEvent({
          event_type: {
            Learning: {
              StrategyServed: {
                query_id: queryId,
                strategy_ids: strategies.map(strategy => strategy.id)
              }
            }
          },
          context,
          metadata: {}
        });
      }
    } catch (error) {
      this.log('Failed to fetch strategies.');
    }

    return summaries;
  }

  private async fetchInsights() {
    if (!this.lastContext || !this.lastTask || !this.lastPlan) return;

    try {
      const memories = await this.client.getContextMemories(this.lastContext, {
        limit: 5,
        min_similarity: 0.3
      });
      if (memories.length) {
        this.log(`Memories: ${memories.length} (latest: ${this.formatMemorySummary(memories[0])})`);
      } else {
        this.log('Memories: 0');
      }
    } catch (error) {
      this.log('Failed to fetch memories.');
    }

    try {
      const strategies = await this.client.getSimilarStrategies({
        goal_ids: [this.lastTask.goalId],
        tool_names: [this.lastPlan.tool],
        result_types: ['action_success', 'action_partial', 'action_failure'],
        limit: 5,
        min_score: 0.1
      });
      this.log(`Similar strategies: ${strategies.length}`);
    } catch (error) {
      this.log('Failed to fetch strategies.');
    }
  }

  private async sendEvent(payload: any) {
    const now = Date.now();
    const agentId = parseInt((document.getElementById('demo-agent-id') as HTMLInputElement)?.value || '1', 10);
    const sessionId = parseInt((document.getElementById('demo-session-id') as HTMLInputElement)?.value || '1', 10);

    const event = {
      id: now + Math.floor(Math.random() * 1000),
      timestamp: now * 1_000_000,
      agent_id: agentId,
      agent_type: 'browser-demo',
      session_id: sessionId,
      event_type: payload.event_type,
      causality_chain: this.lastEventId ? [this.lastEventId] : [],
      context: payload.context,
      metadata: this.wrapMetadata(payload.metadata || {})
    };

    try {
      const baseUrl = (this.client as any)['baseUrl'] || 'http://127.0.0.1:3000';
      const response = await fetch(`${baseUrl}/api/events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event })
      });
      if (!response.ok) {
        const text = await response.text();
        this.log(`Event rejected (${response.status}): ${text}`);
        return;
      }
      this.log('Event accepted.');
      this.lastEventId = event.id;
    } catch (error) {
      this.log(`Event send failed: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }

  private updateScriptForTask() {
    const task = this.getSelectedTask();
    const scriptEl = document.getElementById('demo-script') as HTMLTextAreaElement | null;
    if (!task || !scriptEl) return;
    scriptEl.value = task.script;
  }

  private getScriptInput() {
    return (document.getElementById('demo-script') as HTMLTextAreaElement)?.value.trim() || '';
  }

  private applyRevisedScript(script: string) {
    const scriptEl = document.getElementById('demo-script') as HTMLTextAreaElement | null;
    if (!scriptEl) return;
    scriptEl.value = script;
    this.log('Applied revised script.');
  }

  private formatMemorySummary(memory: any) {
    const feedback = memory?.context?.environment?.variables?.feedback;
    const constraint = memory?.context?.environment?.variables?.constraint;
    const summaryBits = [];
    if (feedback) summaryBits.push(`feedback="${feedback}"`);
    if (constraint) summaryBits.push(`constraint="${constraint}"`);
    const detail = summaryBits.length ? ` ${summaryBits.join(' ')}` : '';
    return `Memory#${memory.id} ${memory.memory_type} outcome=${memory.outcome} strength=${memory.strength.toFixed(2)}${detail}`;
  }

  private wrapMetadata(metadata: Record<string, any>) {
    const wrapped: Record<string, any> = {};
    for (const [key, value] of Object.entries(metadata)) {
      wrapped[key] = this.wrapMetadataValue(value);
    }
    return wrapped;
  }

  private wrapMetadataValue(value: any) {
    if (typeof value === 'string') {
      return { String: value };
    }
    if (typeof value === 'number') {
      return Number.isInteger(value) ? { Integer: value } : { Float: value };
    }
    if (typeof value === 'boolean') {
      return { Boolean: value };
    }
    return { Json: value };
  }

  private computeFingerprint(input: Record<string, any>) {
    const str = JSON.stringify(input);
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return Math.abs(hash);
  }

  private log(message: string) {
    const logEl = document.getElementById('demo-log');
    if (!logEl) return;
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.textContent = `[${timestamp}] ${message}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }
}
