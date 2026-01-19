// Graph Visualization View
// Interactive graph visualization using vis-network

import { EventGraphDBClient } from '../../../dist/index.js';

// Declare vis global from CDN
declare const vis: any;

export class GraphView {
  private client: EventGraphDBClient;
  private network: any = null;
  private nodes: any = null;
  private edges: any = null;

  constructor(client: EventGraphDBClient) {
    this.client = client;
  }

  async render() {
    await this.loadGraphData();
    this.initializeNetwork();
  }

  async refresh() {
    await this.loadGraphData();
    if (this.network) {
      this.updateNetwork();
    }
  }

  private async loadGraphData() {
    try {
      const response = await fetch(this.buildGraphUrl(200));
      if (!response.ok) {
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        this.renderStatus(`Failed to load graph (HTTP ${response.status}).`);
        return;
      }
      const data = await response.json();
      const nodes = (data.nodes || []).map((node: any) => ({
        id: node.id,
        label: node.label || `${node.node_type}:${node.id}`,
        group: node.node_type,
        title: JSON.stringify(node.properties || {}, null, 2),
        properties: node.properties || {}
      }));
      const edges = (data.edges || []).map((edge: any) => ({
        id: edge.id,
        from: edge.from,
        to: edge.to,
        label: `${(edge.weight ?? 0).toFixed(3)}`,
        value: 1,
        title: `Type: ${edge.edge_type}<br/>Weight: ${(edge.weight ?? 0).toFixed(3)}<br/>Confidence: ${(edge.confidence ?? 0).toFixed(3)}`,
        weight: edge.weight ?? 0,
        confidence: edge.confidence ?? 0,
        arrows: 'to'
      }));

      this.nodes = new vis.DataSet(nodes);
      this.edges = new vis.DataSet(edges);
      this.renderStatus(`Loaded ${nodes.length} nodes, ${edges.length} edges.`);
    } catch (error) {
      console.error('Failed to load graph:', error);
      this.nodes = new vis.DataSet([]);
      this.edges = new vis.DataSet([]);
      this.renderStatus('Failed to load graph. Check server and CORS.');
    }
  }

  private buildGraphUrl(limit: number) {
    const params = new URLSearchParams({ limit: String(limit) });
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session_id');
    const agentType = urlParams.get('agent_type');
    if (sessionId) params.set('session_id', sessionId);
    if (agentType) params.set('agent_type', agentType);
    return `${(this.client as any)['baseUrl']}/api/graph?${params.toString()}`;
  }

  private initializeNetwork() {
    const container = document.getElementById('graph-canvas');
    if (!container) return;
    if (typeof vis === 'undefined') {
      this.renderStatus('vis-network failed to load. Check CDN access.');
      return;
    }

    const data = {
      nodes: this.nodes,
      edges: this.edges
    };

    const options = {
      nodes: {
        shape: 'dot',
        size: 4,
        font: {
          size: 9,
          color: '#6b7280',
          face: 'Arial'
        },
        borderWidth: 1,
        shadow: false,
        color: {
          background: '#ffffff',
          border: '#e5e7eb',
          highlight: {
            background: '#f9fafb',
            border: '#d1d5db'
          },
          hover: {
            background: '#f9fafb',
            border: '#d1d5db'
          }
        }
      },
      edges: {
        width: 0.1,
        color: {
          color: '#e5e7eb',
          highlight: '#9ca3af',
          hover: '#9ca3af',
          inherit: false
        },
        font: {
          size: 7,
          color: '#9ca3af',
          face: 'Arial',
          align: 'middle'
        },
        smooth: {
          type: 'continuous'
        },
        arrows: {
          to: { enabled: true, scaleFactor: 0.3 }
        }
      },
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 200
        },
        barnesHut: {
          gravitationalConstant: -2000,
          springConstant: 0.001,
          springLength: 200
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        navigationButtons: true,
        keyboard: true,
        dragNodes: true,
        dragView: true
      }
    };

    this.network = new vis.Network(container, data, options);

    // Add event listeners
    this.network.on('click', (params: any) => {
      if (params.nodes.length > 0) {
        this.showNodeDetails(params.nodes[0]);
      } else if (params.edges.length > 0) {
        this.showEdgeDetails(params.edges[0]);
      }
    });

    // Show empty state when graph has no nodes
    if (!this.nodes || this.nodes.getIds().length === 0) {
      this.showEmptyState();
    }
  }

  private updateNetwork() {
    if (this.network && this.nodes && this.edges) {
      this.network.setData({
        nodes: this.nodes,
        edges: this.edges
      });
      this.network.fit({ animation: true });
    }
  }

  private showNodeDetails(nodeId: string) {
    const detailsPanel = document.getElementById('details-panel');
    if (!detailsPanel) return;

    const node = this.nodes.get(nodeId);
    if (!node) return;
    const properties = JSON.stringify(node.properties || {}, null, 2);

    detailsPanel.innerHTML = `
      <div class="detail-header">
        <h3>Node Details</h3>
      </div>
      <div class="detail-content">
        <div class="detail-item">
          <span class="detail-label">ID:</span>
          <span class="detail-value">${node.id}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Label:</span>
          <span class="detail-value">${node.label || 'N/A'}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Type:</span>
          <span class="detail-value">${node.group || 'N/A'}</span>
        </div>
        <div class="detail-section">
          <h4>Properties</h4>
          <pre>${properties}</pre>
        </div>
      </div>
    `;
    detailsPanel.style.display = 'block';
  }

  private showEdgeDetails(edgeId: string) {
    const detailsPanel = document.getElementById('details-panel');
    if (!detailsPanel) return;

    const edge = this.edges.get(edgeId);
    if (!edge) return;

    detailsPanel.innerHTML = `
      <div class="detail-header">
        <h3>Edge Details</h3>
      </div>
      <div class="detail-content">
        <div class="detail-item">
          <span class="detail-label">ID:</span>
          <span class="detail-value">${edge.id}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">From:</span>
          <span class="detail-value">${edge.from}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">To:</span>
          <span class="detail-value">${edge.to}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Type:</span>
          <span class="detail-value">${edge.label || 'N/A'}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Weight:</span>
          <span class="detail-value">${(edge.weight ?? 0).toFixed(3)}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">Confidence:</span>
          <span class="detail-value">${(edge.confidence ?? 0).toFixed(3)}</span>
        </div>
      </div>
    `;
    detailsPanel.style.display = 'block';
  }

  private showEmptyState() {
    const detailsPanel = document.getElementById('details-panel');
    if (!detailsPanel) return;

    detailsPanel.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🕸️</div>
        <h3>No Graph Data</h3>
        <p>Send some events to the server to see the graph visualization.</p>
        <pre class="code-block">curl -X POST http://127.0.0.1:3000/api/events \\
  -H "Content-Type: application/json" \\
  -d '{
    "event_type": "user_action",
    "event_data": {"action": "test"},
    "agent_id": "agent-1",
    "timestamp": ${Date.now()}
  }'</pre>
      </div>
    `;
    detailsPanel.style.display = 'block';
  }

  private renderStatus(message: string) {
    const detailsPanel = document.getElementById('details-panel');
    if (!detailsPanel) return;
    detailsPanel.innerHTML = `
      <div class="detail-header">
        <h3>Graph Status</h3>
      </div>
      <div class="detail-content">
        <div class="detail-item">
          <span class="detail-value">${message}</span>
        </div>
      </div>
    `;
    detailsPanel.style.display = 'block';
  }
}
