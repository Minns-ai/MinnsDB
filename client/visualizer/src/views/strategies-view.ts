// Strategies View
// View extracted strategies with reasoning steps

import { EventGraphDBClient, StrategyResponse } from '../../../dist/index.js';

export class StrategiesView {
  private client: EventGraphDBClient;
  private strategies: StrategyResponse[] = [];
  private selectedAgentId: number = 1;

  constructor(client: EventGraphDBClient) {
    this.client = client;
  }

  async render() {
    const container = document.getElementById('strategies-view');
    if (!container) return;

    container.innerHTML = `
      <div class="view-header">
        <h2>Strategies</h2>
        <div class="view-controls">
          <input type="number" id="strategies-agent-filter" placeholder="Agent ID" class="filter-input" value="${this.selectedAgentId}">
          <button onclick="window.strategiesView?.applyFilter()" class="btn">Filter</button>
        </div>
      </div>
      <div class="strategies-list" id="strategies-list"></div>
    `;

    this.setupEventListeners();
    await this.loadStrategies();
  }

  async refresh() {
    await this.loadStrategies();
  }

  private setupEventListeners() {
    // Will be set up after render
  }

  private async loadStrategies() {
    try {
      this.strategies = await this.client.getAgentStrategies(this.selectedAgentId, 100);
      this.renderStrategies();
    } catch (error) {
      console.error('Failed to load strategies:', error);
      this.strategies = [];
      this.renderStrategies();
    }
  }

  applyFilter() {
    const agentInput = document.getElementById('strategies-agent-filter') as HTMLInputElement;
    if (agentInput) {
      const agentId = agentInput.value ? parseInt(agentInput.value) : 1;
      this.selectedAgentId = agentId;
      this.loadStrategies();
    }
  }

  private renderStrategies() {
    const listContainer = document.getElementById('strategies-list');
    if (!listContainer) return;

    if (this.strategies.length === 0) {
      listContainer.innerHTML = '<div class="empty-state">No strategies found</div>';
      return;
    }

    listContainer.innerHTML = this.strategies.map(strategy => {
      const successRate = strategy.success_count + strategy.failure_count > 0
        ? (strategy.success_count / (strategy.success_count + strategy.failure_count) * 100).toFixed(1)
        : '0';
      const qualityPercent = (strategy.quality_score * 100).toFixed(1);

      return `
        <div class="strategy-item" onclick="window.strategiesView?.showStrategyDetails(${strategy.id})">
          <div class="strategy-header">
            <span class="strategy-name">${strategy.name}</span>
            <span class="strategy-quality">${strategy.strategy_type} • Quality: ${qualityPercent}%</span>
          </div>
          <div class="strategy-body">
            <div class="strategy-stats">
              <div class="stat">
                <span class="stat-label">Success Rate</span>
                <span class="stat-value">${successRate}%</span>
              </div>
              <div class="stat">
                <span class="stat-label">Successes</span>
                <span class="stat-value">${strategy.success_count}</span>
              </div>
              <div class="stat">
                <span class="stat-label">Failures</span>
                <span class="stat-value">${strategy.failure_count}</span>
              </div>
              <div class="stat">
                <span class="stat-label">Steps</span>
                <span class="stat-value">${strategy.reasoning_steps.length}</span>
              </div>
            </div>
            <div class="strategy-steps-preview">
              ${strategy.reasoning_steps.slice(0, 3).map(step => 
                `<div class="step-preview">${step.sequence_order + 1}. ${step.description.substring(0, 50)}...</div>`
              ).join('')}
              ${strategy.reasoning_steps.length > 3 ? `<div class="step-preview text-muted">+${strategy.reasoning_steps.length - 3} more steps</div>` : ''}
            </div>
          </div>
        </div>
      `;
    }).join('');

    (window as any).strategiesView = this;
  }

  showStrategyDetails(strategyId: number) {
    const strategy = this.strategies.find(s => s.id === strategyId);
    if (!strategy) return;

    const successRate = strategy.success_count + strategy.failure_count > 0
      ? (strategy.success_count / (strategy.success_count + strategy.failure_count) * 100).toFixed(1)
      : '0';

    const panel = document.getElementById('details-panel');
    if (panel) {
      panel.innerHTML = `
        <div class="detail-header">
          <h3>${strategy.name}</h3>
        </div>
        <div class="detail-content">
          <div class="detail-section">
            <h4>Overview</h4>
            <table class="detail-table">
              <tr><td>ID</td><td>${strategy.id}</td></tr>
              <tr><td>Agent ID</td><td>${strategy.agent_id}</td></tr>
              <tr><td>Quality Score</td><td>${(strategy.quality_score * 100).toFixed(1)}%</td></tr>
              <tr><td>Success Rate</td><td>${successRate}%</td></tr>
              <tr><td>Success Count</td><td>${strategy.success_count}</td></tr>
              <tr><td>Failure Count</td><td>${strategy.failure_count}</td></tr>
            </table>
          </div>
          <div class="detail-section">
            <h4>Reasoning Steps</h4>
            <ol class="reasoning-steps">
              ${strategy.reasoning_steps.map(step => 
                `<li><div class="step-description">${step.description}</div></li>`
              ).join('')}
            </ol>
          </div>
        </div>
      `;
    }
  }
}
