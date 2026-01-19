// Episodes View
// View detected episodes and their outcomes

import { EventGraphDBClient, EpisodeResponse } from '../../../dist/index.js';

export class EpisodesView {
  private client: EventGraphDBClient;
  private episodes: EpisodeResponse[] = [];

  constructor(client: EventGraphDBClient) {
    this.client = client;
  }

  async render() {
    await this.loadEpisodes();
    this.renderEpisodes();
  }

  async refresh() {
    await this.loadEpisodes();
    this.renderEpisodes();
  }

  private async loadEpisodes() {
    try {
      this.episodes = await this.client.getEpisodes(100);
    } catch (error) {
      console.error('Failed to load episodes:', error);
      this.episodes = [];
    }
  }

  private renderEpisodes() {
    const container = document.getElementById('episodes-view');
    if (!container) return;

    container.innerHTML = `
      <div class="view-header">
        <h2>Episodes</h2>
        <div class="view-stats">
          <span>Total: ${this.episodes.length}</span>
        </div>
      </div>
      <div class="episodes-grid">
        ${this.episodes.map(episode => this.renderEpisodeCard(episode)).join('')}
      </div>
    `;
  }

  private renderEpisodeCard(episode: EpisodeResponse): string {
    const outcomeClass = episode.outcome ? 
      (episode.outcome === 'Success' ? 'success' : 
       episode.outcome === 'Failure' ? 'failure' : 'partial') : 'unknown';

    return `
      <div class="episode-card ${outcomeClass}" onclick="window.episodesView?.showEpisodeDetails(${episode.id})">
        <div class="episode-header">
          <span class="episode-id">Episode #${episode.id}</span>
          ${episode.outcome ? `<span class="outcome-badge ${outcomeClass}">${episode.outcome}</span>` : ''}
        </div>
        <div class="episode-body">
          <div class="episode-stat">
            <span class="stat-label">Events</span>
            <span class="stat-value">${episode.event_count}</span>
          </div>
          <div class="episode-stat">
            <span class="stat-label">Significance</span>
            <span class="stat-value">${(episode.significance * 100).toFixed(1)}%</span>
          </div>
          <div class="episode-stat">
            <span class="stat-label">Agent</span>
            <span class="stat-value">${episode.agent_id}</span>
          </div>
        </div>
      </div>
    `;
  }

  showEpisodeDetails(episodeId: number) {
    const episode = this.episodes.find(e => e.id === episodeId);
    if (!episode) return;

    // Show in details panel
    const panel = document.getElementById('details-panel');
    if (panel) {
      panel.innerHTML = `
        <div class="detail-header">
          <h3>Episode #${episode.id}</h3>
        </div>
        <div class="detail-content">
          <div class="detail-section">
            <h4>Overview</h4>
            <table class="detail-table">
              <tr><td>ID</td><td>${episode.id}</td></tr>
              <tr><td>Agent ID</td><td>${episode.agent_id}</td></tr>
              <tr><td>Event Count</td><td>${episode.event_count}</td></tr>
              <tr><td>Significance</td><td>${(episode.significance * 100).toFixed(1)}%</td></tr>
              <tr><td>Outcome</td><td>${episode.outcome || 'Pending'}</td></tr>
            </table>
          </div>
        </div>
      `;
    }
  }
}

// Expose for onclick handlers - will be set when view is instantiated
