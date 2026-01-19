// EventGraphDB Visualizer - Main Application
// Prisma Studio-style debugger for EventGraphDB
import { EventGraphDBClient } from '../../dist/index.js';
import { GraphView } from './views/graph-view.js';
import { AgentDemoView } from './views/agent-demo-view.js';
import { EventsView } from './views/events-view.js';
import { EpisodesView } from './views/episodes-view.js';
import { MemoriesView } from './views/memories-view.js';
import { StrategiesView } from './views/strategies-view.js';
import { StatsView } from './views/stats-view.js';
export class EventGraphDBVisualizer {
    constructor(baseUrl = 'http://127.0.0.1:3000') {
        this.currentView = 'graph';
        this.views = new Map();
        this.client = new EventGraphDBClient({ baseUrl });
        this.initializeViews();
        this.setupEventListeners();
    }
    initializeViews() {
        this.views.set('graph', new GraphView(this.client));
        this.views.set('agent-demo', new AgentDemoView(this.client));
        this.views.set('events', new EventsView(this.client));
        const episodesView = new EpisodesView(this.client);
        this.views.set('episodes', episodesView);
        window.episodesView = episodesView; // Expose for onclick
        this.views.set('memories', new MemoriesView(this.client));
        this.views.set('strategies', new StrategiesView(this.client));
        this.views.set('stats', new StatsView(this.client));
    }
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('[data-view]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const view = e.target.getAttribute('data-view');
                if (view)
                    this.switchView(view);
            });
        });
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh-toggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startAutoRefresh();
                }
                else {
                    this.stopAutoRefresh();
                }
            });
        }
        // Manual refresh
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refresh());
        }
    }
    async switchView(viewName) {
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`)?.classList.add('active');
        // Hide all views
        document.querySelectorAll('.view-container').forEach(view => {
            view.style.display = 'none';
        });
        // Show selected view
        const viewContainer = document.getElementById(`${viewName}-view`);
        if (viewContainer) {
            viewContainer.style.display = 'block';
            this.currentView = viewName;
            // Initialize view if it exists
            const view = this.views.get(viewName);
            if (view && typeof view.render === 'function') {
                await view.render();
            }
        }
    }
    async refresh() {
        const view = this.views.get(this.currentView);
        if (view && typeof view.refresh === 'function') {
            await view.refresh();
        }
    }
    startAutoRefresh() {
        this.refreshInterval = window.setInterval(() => {
            this.refresh();
        }, 5000); // Refresh every 5 seconds
    }
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = undefined;
        }
    }
    async init() {
        // Check connection
        try {
            const health = await this.client.healthCheck();
            this.updateConnectionStatus(health.is_healthy);
        }
        catch (error) {
            this.updateConnectionStatus(false);
            console.error('Failed to connect to EventGraphDB:', error);
        }
        // Load initial view
        await this.switchView('graph');
    }
    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = connected ? 'Connected' : 'Disconnected';
            statusEl.className = connected ? 'status connected' : 'status disconnected';
        }
    }
}
// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const app = new EventGraphDBVisualizer();
        window.visualizer = app; // Expose for debugging
        app.init();
    });
}
else {
    const app = new EventGraphDBVisualizer();
    window.visualizer = app;
    app.init();
}
