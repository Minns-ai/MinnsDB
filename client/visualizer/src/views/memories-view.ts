// Memories View
// View formed memories with strength and access patterns

import { EventGraphDBClient, MemoryResponse } from '../../../dist/index.js';

export class MemoriesView {
  private client: EventGraphDBClient;
  private memories: MemoryResponse[] = [];
  private selectedAgentId: number = 1;

  constructor(client: EventGraphDBClient) {
    this.client = client;
  }

  async render() {
    const container = document.getElementById('memories-view');
    if (!container) return;

    container.innerHTML = `
      <div class="view-header">
        <h2>Memories</h2>
        <div class="view-controls">
          <input type="number" id="memories-agent-filter" placeholder="Agent ID" class="filter-input" value="${this.selectedAgentId}">
          <button onclick="window.memoriesView?.applyFilter()" class="btn">Filter</button>
        </div>
      </div>
      <div class="memories-list" id="memories-list"></div>
    `;

    this.setupEventListeners();
    await this.loadMemories();
  }

  async refresh() {
    await this.loadMemories();
  }

  private setupEventListeners() {
    // Will be set up after render
  }

  private async loadMemories() {
    try {
      this.memories = await this.client.getAgentMemories(this.selectedAgentId, 100);
      this.renderMemories();
    } catch (error) {
      console.error('Failed to load memories:', error);
      this.memories = [];
      this.renderMemories();
    }
  }

  applyFilter() {
    const agentInput = document.getElementById('memories-agent-filter') as HTMLInputElement;
    if (agentInput) {
      const agentId = agentInput.value ? parseInt(agentInput.value) : 1;
      this.selectedAgentId = agentId;
      this.loadMemories();
    }
  }

  private renderMemories() {
    const listContainer = document.getElementById('memories-list');
    if (!listContainer) return;

    if (this.memories.length === 0) {
      listContainer.innerHTML = '<div class="empty-state">No memories found</div>';
      return;
    }

    listContainer.innerHTML = this.memories.map(memory => {
      const strengthPercent = (memory.strength * 100).toFixed(1);
      const relevancePercent = (memory.relevance_score * 100).toFixed(1);
      const created = new Date(memory.formed_at / 1000000).toLocaleString();
      const lastAccessed = new Date(memory.last_accessed / 1000000).toLocaleString();
      const feedback = (memory.context?.environment?.variables as any)?.feedback;

      return `
        <div class="memory-item" onclick="window.memoriesView?.showMemoryDetails(${memory.id})">
          <div class="memory-header">
            <span class="memory-id">Memory #${memory.id}</span>
            <span class="memory-strength" style="width: ${strengthPercent}%"></span>
          </div>
          <div class="memory-body">
            <div class="memory-stats">
              <div class="stat">
                <span class="stat-label">Strength</span>
                <span class="stat-value">${strengthPercent}%</span>
              </div>
              <div class="stat">
                <span class="stat-label">Relevance</span>
                <span class="stat-value">${relevancePercent}%</span>
              </div>
              <div class="stat">
                <span class="stat-label">Accesses</span>
                <span class="stat-value">${memory.access_count}</span>
              </div>
            </div>
            <div class="memory-meta">
              Agent: ${memory.agent_id} • ${memory.memory_type} • Outcome: ${memory.outcome}
            </div>
            ${feedback ? `<div class="memory-meta">Feedback: ${feedback}</div>` : ''}
          </div>
        </div>
      `;
    }).join('');

    (window as any).memoriesView = this;
  }

  showMemoryDetails(memoryId: number) {
    const memory = this.memories.find(m => m.id === memoryId);
    if (!memory) return;

    const panel = document.getElementById('details-panel');
    if (panel) {
      panel.innerHTML = `
        <div class="detail-header">
          <h3>Memory #${memory.id}</h3>
        </div>
        <div class="detail-content">
          <div class="detail-section">
            <h4>Memory Metrics</h4>
            <table class="detail-table">
              <tr><td>ID</td><td>${memory.id}</td></tr>
              <tr><td>Agent ID</td><td>${memory.agent_id}</td></tr>
              <tr><td>Strength</td><td>${(memory.strength * 100).toFixed(1)}%</td></tr>
              <tr><td>Relevance Score</td><td>${(memory.relevance_score * 100).toFixed(1)}%</td></tr>
              <tr><td>Access Count</td><td>${memory.access_count}</td></tr>
              <tr><td>Created</td><td>${new Date(memory.formed_at / 1000000).toLocaleString()}</td></tr>
              <tr><td>Last Accessed</td><td>${new Date(memory.last_accessed / 1000000).toLocaleString()}</td></tr>
            </table>
          </div>
        </div>
      `;
    }
  }
}
