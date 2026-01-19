// Events Timeline View
// List and inspect events with filtering
export class EventsView {
    constructor(client) {
        this.events = [];
        this.filteredEvents = [];
        this.selectedEvent = null;
        this.client = client;
    }
    async render() {
        const container = document.getElementById('events-view');
        if (!container)
            return;
        container.innerHTML = `
      <div class="event-compose">
        <div class="compose-header">
          <h3>Send Event</h3>
          <div class="compose-actions">
            <button id="events-load-sample" class="btn btn-secondary">Load Sample</button>
            <button id="events-send" class="btn">Send</button>
          </div>
        </div>
        <textarea id="events-payload" class="code-input" rows="12" spellcheck="false"></textarea>
        <div id="events-send-status" class="send-status"></div>
      </div>
      <div class="view-header">
        <h2>Events Timeline</h2>
        <div class="view-controls">
          <input type="text" id="events-search" placeholder="Search events..." class="search-input">
          <select id="events-filter-type" class="filter-select">
            <option value="all">All Types</option>
            <option value="Action">Actions</option>
            <option value="Observation">Observations</option>
            <option value="Cognitive">Cognitive</option>
          </select>
          <select id="events-filter-agent" class="filter-select">
            <option value="all">All Agents</option>
          </select>
        </div>
      </div>
      <div class="view-content">
        <div class="events-list" id="events-list"></div>
        <div class="events-details is-hidden" id="events-details"></div>
      </div>
    `;
        this.setupEventListeners();
        this.loadSampleIntoEditor();
        this.updateLayout(false);
        await this.loadEvents();
    }
    async refresh() {
        await this.loadEvents();
    }
    setupEventListeners() {
        const searchInput = document.getElementById('events-search');
        const typeFilter = document.getElementById('events-filter-type');
        const agentFilter = document.getElementById('events-filter-agent');
        const sendButton = document.getElementById('events-send');
        const sampleButton = document.getElementById('events-load-sample');
        if (searchInput) {
            searchInput.addEventListener('input', () => this.applyFilters());
        }
        if (typeFilter) {
            typeFilter.addEventListener('change', () => this.applyFilters());
        }
        if (agentFilter) {
            agentFilter.addEventListener('change', () => this.applyFilters());
        }
        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendEventFromEditor());
        }
        if (sampleButton) {
            sampleButton.addEventListener('click', () => this.loadSampleIntoEditor());
        }
    }
    async loadEvents() {
        try {
            const response = await fetch(this.buildGraphUrl(200));
            if (!response.ok) {
                this.events = [];
                this.applyFilters();
                return;
            }
            const data = await response.json();
            const nodes = data.nodes || [];
            this.events = nodes.filter(node => node.node_type.startsWith('Event::'));
            this.applyFilters();
        }
        catch (error) {
            console.error('Failed to load events:', error);
            this.events = [];
            this.renderEventsList();
        }
    }
    buildGraphUrl(limit) {
        const params = new URLSearchParams({ limit: String(limit) });
        const urlParams = new URLSearchParams(window.location.search);
        const sessionId = urlParams.get('session_id');
        const agentType = urlParams.get('agent_type');
        if (sessionId)
            params.set('session_id', sessionId);
        if (agentType)
            params.set('agent_type', agentType);
        return `${this.client['baseUrl']}/api/graph?${params.toString()}`;
    }
    applyFilters() {
        const searchTerm = document.getElementById('events-search')?.value.toLowerCase() || '';
        const typeFilter = document.getElementById('events-filter-type')?.value || 'all';
        const agentFilter = document.getElementById('events-filter-agent')?.value || 'all';
        this.filteredEvents = this.events.filter(event => {
            // Search filter
            if (searchTerm) {
                const searchable = JSON.stringify(event).toLowerCase();
                if (!searchable.includes(searchTerm))
                    return false;
            }
            // Type filter
            if (typeFilter !== 'all') {
                if (!event.node_type.toLowerCase().includes(typeFilter.toLowerCase()))
                    return false;
            }
            // Agent filter
            if (agentFilter !== 'all' && (event.properties?.agent_id?.toString() || '') !== agentFilter) {
                return false;
            }
            return true;
        });
        this.renderEventsList();
    }
    renderEventsList() {
        const listContainer = document.getElementById('events-list');
        if (!listContainer)
            return;
        if (this.filteredEvents.length === 0) {
            listContainer.innerHTML = '<div class="empty-state">No events yet</div>';
            this.selectedEvent = null;
            this.renderEventDetails();
            return;
        }
        listContainer.innerHTML = this.filteredEvents.map((event, index) => {
            const eventType = this.getEventType(event);
            const outcome = this.getEventOutcome(event);
            const timestamp = new Date(Number(event.created_at) / 1000000).toLocaleString();
            return `
        <div class="event-item ${this.selectedEvent?.id === event.id ? 'active' : ''}" 
             onclick="window.eventsView?.selectEvent(${index})">
          <div class="event-header">
            <span class="event-type-badge ${eventType.toLowerCase()}">${eventType}</span>
            <span class="event-time">${timestamp}</span>
          </div>
          <div class="event-body">
            <div class="event-title">${this.getEventTitle(event)}</div>
            <div class="event-meta">
              Node: ${event.node_type}
              ${outcome ? ` • ${outcome}` : ''}
            </div>
          </div>
        </div>
      `;
        }).join('');
        // Expose selectEvent globally for onclick
        window.eventsView = this;
    }
    selectEvent(index) {
        this.selectedEvent = this.filteredEvents[index];
        this.renderEventsList();
        this.renderEventDetails();
    }
    clearSelection() {
        this.selectedEvent = null;
        this.renderEventsList();
        this.renderEventDetails();
    }
    renderEventDetails() {
        const detailsContainer = document.getElementById('events-details');
        if (!detailsContainer || !this.selectedEvent) {
            if (detailsContainer) {
                detailsContainer.innerHTML = '';
                detailsContainer.classList.add('is-hidden');
            }
            this.updateLayout(false);
            return;
        }
        const event = this.selectedEvent;
        detailsContainer.classList.remove('is-hidden');
        this.updateLayout(true);
        detailsContainer.innerHTML = `
      <div class="detail-header">
        <h3>Event Node Details</h3>
        <button class="btn-close" onclick="window.eventsView?.clearSelection()">×</button>
      </div>
      <div class="detail-content">
        <div class="detail-section">
          <h4>Basic Information</h4>
          <table class="detail-table">
            <tr><td>ID</td><td><code>${event.id}</code></td></tr>
            <tr><td>Created</td><td>${new Date(Number(event.created_at) / 1000000).toLocaleString()}</td></tr>
            <tr><td>Node Type</td><td>${event.node_type}</td></tr>
            <tr><td>Label</td><td>${event.label || 'N/A'}</td></tr>
          </table>
        </div>
        <div class="detail-section">
          <h4>Properties</h4>
          <pre>${JSON.stringify(event.properties || {}, null, 2)}</pre>
        </div>
      </div>
    `;
    }
    getEventType(event) {
        if (event.node_type.includes('Action'))
            return 'Action';
        if (event.node_type.includes('Observation'))
            return 'Observation';
        if (event.node_type.includes('Cognitive'))
            return 'Cognitive';
        return 'Event';
    }
    getEventTitle(event) {
        return event.label || event.node_type;
    }
    getEventOutcome(event) {
        const outcome = event.properties?.outcome;
        if (!outcome)
            return '';
        if (outcome.Success)
            return '✓ Success';
        if (outcome.Failure)
            return '✗ Failure';
        if (outcome.Partial)
            return '~ Partial';
        return '';
    }
    loadSampleIntoEditor() {
        const payload = document.getElementById('events-payload');
        if (!payload)
            return;
        payload.value = JSON.stringify(this.buildSampleEvent(), null, 2);
        this.setSendStatus('', false);
    }
    async sendEventFromEditor() {
        const payload = document.getElementById('events-payload');
        if (!payload)
            return;
        try {
            const parsed = JSON.parse(payload.value);
            const rawEvent = parsed.event ? parsed.event : parsed;
            const event = this.normalizeEvent(rawEvent);
            const response = await this.client.processEvent(event);
            this.applyFilters();
            this.setSendStatus(`Event sent. Nodes created: ${response.nodes_created}, patterns: ${response.patterns_detected}`, false);
        }
        catch (error) {
            const message = error instanceof Error ? error.message : 'Unknown error';
            this.setSendStatus(`Failed to send event: ${message}`, true);
        }
    }
    setSendStatus(message, isError) {
        const statusEl = document.getElementById('events-send-status');
        if (!statusEl)
            return;
        if (!message) {
            statusEl.textContent = '';
            statusEl.className = 'send-status';
            return;
        }
        statusEl.textContent = message;
        statusEl.className = `send-status active ${isError ? 'error' : 'success'}`;
    }
    normalizeEvent(raw) {
        const now = Date.now();
        const event = raw;
        if (!event.id)
            event.id = now;
        if (!event.timestamp)
            event.timestamp = now * 1000000;
        if (!event.agent_id)
            event.agent_id = 1;
        if (!event.agent_type)
            event.agent_type = 'studio';
        if (!event.session_id)
            event.session_id = 1;
        if (!event.causality_chain)
            event.causality_chain = [];
        if (!event.metadata)
            event.metadata = {};
        if (!event.context)
            event.context = this.buildSampleEvent().context;
        return event;
    }
    buildSampleEvent() {
        const now = Date.now();
        return {
            id: now,
            timestamp: now * 1000000,
            agent_id: 1,
            agent_type: 'studio',
            session_id: 1,
            event_type: {
                Action: {
                    action_name: 'demo_action',
                    parameters: { source: 'visualizer' },
                    outcome: { Success: { result: { ok: true } } },
                    duration_ns: 1000000
                }
            },
            causality_chain: [],
            context: {
                environment: {
                    variables: { source: 'visualizer' },
                    spatial: null,
                    temporal: {
                        time_of_day: null,
                        deadlines: [],
                        patterns: []
                    }
                },
                active_goals: [],
                resources: {
                    computational: {
                        cpu_percent: 10,
                        memory_bytes: 1024 * 1024 * 1024,
                        storage_bytes: 10 * 1024 * 1024 * 1024,
                        network_bandwidth: 1024 * 1024
                    },
                    external: {}
                },
                embeddings: null
            },
            metadata: {}
        };
    }
    updateLayout(hasDetails) {
        const content = document.querySelector('.view-content');
        if (!content)
            return;
        if (hasDetails) {
            content.classList.remove('single-column');
        }
        else {
            content.classList.add('single-column');
        }
    }
}
