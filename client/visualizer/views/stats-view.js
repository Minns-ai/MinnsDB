// Statistics Dashboard View
// System-wide statistics and health metrics
export class StatsView {
    constructor(client) {
        this.stats = null;
        this.health = null;
        this.client = client;
    }
    async render() {
        await this.loadData();
        this.renderStats();
    }
    async refresh() {
        await this.loadData();
        this.renderStats();
    }
    async loadData() {
        try {
            [this.stats, this.health] = await Promise.all([
                this.client.getStats(),
                this.client.healthCheck(),
            ]);
        }
        catch (error) {
            console.error('Failed to load stats:', error);
            this.stats = null;
            this.health = null;
        }
    }
    renderStats() {
        const container = document.getElementById('stats-view');
        if (!container)
            return;
        if (!this.stats || !this.health) {
            container.innerHTML = '<div class="empty-state">Failed to load statistics</div>';
            return;
        }
        const successRate = this.stats.total_episodes_detected > 0
            ? ((this.stats.total_memories_formed / this.stats.total_episodes_detected) * 100).toFixed(1)
            : '0';
        container.innerHTML = `
      <div class="view-header">
        <h2>System Statistics</h2>
        <div class="health-indicator ${this.health.is_healthy ? 'healthy' : 'unhealthy'}">
          ${this.health.is_healthy ? '✓ Healthy' : '✗ Unhealthy'}
        </div>
      </div>
      <div class="stats-grid">
        <div class="stat-card large">
          <div class="stat-icon">📊</div>
          <div class="stat-content">
            <div class="stat-value">${this.stats.total_events_processed.toLocaleString()}</div>
            <div class="stat-label">Total Events Processed</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.health.node_count.toLocaleString()}</div>
          <div class="stat-label">Graph Nodes</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.health.edge_count.toLocaleString()}</div>
          <div class="stat-label">Graph Edges</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.stats.total_episodes_detected.toLocaleString()}</div>
          <div class="stat-label">Episodes Detected</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.stats.total_memories_formed.toLocaleString()}</div>
          <div class="stat-label">Memories Formed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.stats.total_strategies_extracted.toLocaleString()}</div>
          <div class="stat-label">Strategies Extracted</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.stats.total_reinforcements_applied.toLocaleString()}</div>
          <div class="stat-label">Reinforcements Applied</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.stats.average_processing_time_ms.toFixed(2)}ms</div>
          <div class="stat-label">Avg Processing Time</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${this.health.processing_rate.toFixed(0)}</div>
          <div class="stat-label">Events/Second</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${successRate}%</div>
          <div class="stat-label">Memory Formation Rate</div>
        </div>
      </div>
      <div class="stats-details">
        <div class="detail-section">
          <h3>System Health</h3>
          <table class="detail-table">
            <tr><td>Status</td><td>${this.health.status}</td></tr>
            <tr><td>Version</td><td>${this.health.version}</td></tr>
            <tr><td>Uptime</td><td>${this.formatUptime(this.health.uptime_seconds)}</td></tr>
            <tr><td>Is Healthy</td><td>${this.health.is_healthy ? 'Yes' : 'No'}</td></tr>
          </table>
        </div>
      </div>
    `;
    }
    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (days > 0)
            return `${days}d ${hours}h ${minutes}m`;
        if (hours > 0)
            return `${hours}h ${minutes}m`;
        return `${minutes}m`;
    }
}
