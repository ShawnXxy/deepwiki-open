'use client';

import React, { useMemo, useState, useCallback } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Panel,
  useNodesState,
  useEdgesState,
  useReactFlow,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import type { CodeMapData, CodeMapNode, CodeMapEdge } from '@/types/codemap';

// ============================================================================
// Types
// ============================================================================

type ViewMode = 'files' | 'symbols' | 'classes';

interface SelectedInfo {
  node: CodeMapNode;
  children: CodeMapNode[];
  incomingEdges: { edge: CodeMapEdge; sourceNode?: CodeMapNode }[];
  outgoingEdges: { edge: CodeMapEdge; targetNode?: CodeMapNode }[];
}

/* eslint-disable @typescript-eslint/no-explicit-any */

// ============================================================================
// Custom Node Components
// ============================================================================

function FileNode({ data }: { data: any }) {
  return (
    <div className={`px-3 py-2 rounded-lg border-2 shadow-sm min-w-[150px] max-w-[240px] cursor-pointer transition-all
      ${data.isSelected ? 'border-blue-500 ring-2 ring-blue-300 bg-blue-50 dark:bg-blue-950' :
        data.isConnected ? 'border-blue-400 bg-blue-50/60 dark:bg-blue-900/20' :
        data.isDimmed ? 'border-gray-200 dark:border-gray-700 opacity-25' :
        'border-blue-300 bg-white dark:bg-gray-800 hover:border-blue-400'}`}
    >
      <Handle type="target" position={Position.Top} className="!bg-blue-400 !w-2 !h-2" />
      <div className="flex items-center gap-1.5">
        <span className="text-sm flex-shrink-0">{data.expanded ? '📂' : '📄'}</span>
        <span className="text-xs font-semibold truncate">{data.label}</span>
      </div>
      <div className="text-[10px] text-gray-500 mt-0.5">
        {data.language && <>{data.language} · </>}{data.symbolCount} symbols
      </div>
      {data.expanded && data.symbols?.length > 0 && (
        <div className="mt-1.5 pt-1.5 border-t border-blue-200 dark:border-gray-600 space-y-0.5 max-h-[200px] overflow-y-auto">
          {data.symbols.map((s: any) => (
            <div key={s.id} className="flex items-center gap-1 text-[10px] py-0.5">
              <span className="flex-shrink-0">{s.kind === 'class' ? '🏛️' : s.kind === 'method' ? '🔧' : '⚡'}</span>
              <span className="truncate text-gray-600 dark:text-gray-300">{s.name}</span>
            </div>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-blue-400 !w-2 !h-2" />
    </div>
  );
}

function ClassNode({ data }: { data: any }) {
  return (
    <div className={`px-3 py-2 rounded border-2 shadow-sm min-w-[140px] max-w-[240px] cursor-pointer transition-all
      ${data.isSelected ? 'border-purple-500 ring-2 ring-purple-300 bg-purple-50 dark:bg-purple-950' :
        data.isConnected ? 'border-purple-400 bg-purple-50/60 dark:bg-purple-900/20' :
        data.isDimmed ? 'border-gray-200 dark:border-gray-700 opacity-25' :
        'border-purple-300 bg-white dark:bg-gray-800 hover:border-purple-400'}`}
    >
      <Handle type="target" position={Position.Top} className="!bg-purple-400 !w-2 !h-2" />
      <div className="flex items-center gap-1.5">
        <span className="text-sm flex-shrink-0">🏛️</span>
        <span className="text-xs font-semibold">{data.label}</span>
      </div>
      <div className="text-[10px] text-gray-500 mt-0.5">
        {data.methodCount} methods · {data.filePath}
      </div>
      {data.expanded && data.methods?.length > 0 && (
        <div className="mt-1.5 pt-1.5 border-t border-purple-200 dark:border-gray-600 space-y-0.5 max-h-[200px] overflow-y-auto">
          {data.methods.map((m: any) => (
            <div key={m.id} className="flex items-center gap-1 text-[10px] py-0.5">
              <span className="flex-shrink-0">🔧</span>
              <span className="truncate text-gray-600 dark:text-gray-300 font-mono">{m.signature || m.name}</span>
            </div>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-purple-400 !w-2 !h-2" />
    </div>
  );
}

function FunctionNode({ data }: { data: any }) {
  return (
    <div className={`px-2.5 py-1.5 rounded border shadow-sm min-w-[110px] max-w-[200px] cursor-pointer transition-all
      ${data.isSelected ? 'border-green-500 ring-2 ring-green-300 bg-green-50 dark:bg-green-950' :
        data.isConnected ? 'border-green-400 bg-green-50/60 dark:bg-green-900/20' :
        data.isDimmed ? 'border-gray-200 dark:border-gray-700 opacity-25' :
        'border-green-300 bg-white dark:bg-gray-800 hover:border-green-400'}`}
    >
      <Handle type="target" position={Position.Top} className="!bg-green-400 !w-2 !h-2" />
      <div className="flex items-center gap-1">
        <span className="text-xs flex-shrink-0">⚡</span>
        <span className="text-[11px] font-medium truncate">{data.label}</span>
      </div>
      {data.signature && (
        <div className="text-[10px] text-gray-400 font-mono truncate mt-0.5">{data.signature}</div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-green-400 !w-2 !h-2" />
    </div>
  );
}

/* eslint-enable @typescript-eslint/no-explicit-any */

const nodeTypes: NodeTypes = { fileNode: FileNode, classNode: ClassNode, functionNode: FunctionNode };

// ============================================================================
// Layout & edge styles
// ============================================================================

function edgeColor(kind: string, highlighted: boolean): { stroke: string; dash?: string } {
  const colors: Record<string, string> = { imports: '#3B82F6', calls: '#6B7280', inherits: '#10B981', implements: '#8B5CF6' };
  const dashed = new Set(['imports', 'implements']);
  const c = colors[kind] || '#6B7280';
  return { stroke: highlighted ? c : `${c}30`, dash: dashed.has(kind) ? '6 3' : undefined };
}

function computeLayout(flowNodes: Node[], flowEdges: Edge[], direction: 'TB' | 'LR'): Node[] {
  if (flowNodes.length === 0) return [];
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: direction, nodesep: 80, ranksep: 100, marginx: 30, marginy: 30 });
  g.setDefaultEdgeLabel(() => ({}));
  flowNodes.forEach(node => {
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const d = node.data as any;
    const items = d.expanded ? (d.symbols?.length || d.methods?.length || 0) : 0;
    const h = (node.type === 'functionNode' ? 38 : 50) + Math.min(items, 20) * 18;
    const w = node.type === 'functionNode' ? 160 : 230;
    /* eslint-enable @typescript-eslint/no-explicit-any */
    g.setNode(node.id, { width: w, height: h });
  });
  flowEdges.forEach(e => { if (g.hasNode(e.source) && g.hasNode(e.target)) g.setEdge(e.source, e.target); });
  dagre.layout(g);
  return flowNodes.map(node => {
    const p = g.node(node.id);
    return p ? { ...node, position: { x: p.x - (p.width || 0) / 2, y: p.y - (p.height || 0) / 2 } } : node;
  });
}

// ============================================================================
// Graph builder
// ============================================================================

const MAX_NODES = 400;

function buildGraph(
  data: CodeMapData, viewMode: ViewMode, query: string,
  expandedIds: Set<string>, selectedId: string | null,
) {
  const raw = data.nodes;
  const rawEdges = data.edges;
  const q = query.toLowerCase();

  // Indexes
  const methodsOf: Record<string, CodeMapNode[]> = {};
  const symbolsOf: Record<string, CodeMapNode[]> = {};
  raw.forEach(n => {
    if (n.kind === 'method' && n.parentId) (methodsOf[n.parentId] ??= []).push(n);
    if (n.kind !== 'file') (symbolsOf[n.filePath] ??= []).push(n);
  });

  // Filter by view
  let filtered: CodeMapNode[];
  let edgeKinds: Set<string>;
  switch (viewMode) {
    case 'files':
      filtered = raw.filter(n => n.kind === 'file');
      edgeKinds = new Set(['imports']);
      break;
    case 'classes':
      filtered = raw.filter(n => n.kind === 'class' || n.kind === 'function');
      edgeKinds = new Set(['inherits', 'implements', 'calls']);
      break;
    default:
      filtered = raw.filter(n => n.kind !== 'method');
      edgeKinds = new Set(['imports', 'calls', 'inherits']);
      break;
  }

  if (q) filtered = filtered.filter(n => n.name.toLowerCase().includes(q) || n.filePath.toLowerCase().includes(q));
  const truncated = filtered.length > MAX_NODES;
  if (truncated) filtered = filtered.slice(0, MAX_NODES);

  // Connected set for highlight
  const connected = new Set<string>();
  if (selectedId) {
    connected.add(selectedId);
    rawEdges.forEach(e => {
      if (e.sourceId === selectedId || e.targetId === selectedId) {
        connected.add(e.sourceId);
        connected.add(e.targetId);
      }
    });
  }

  const visibleIds = new Set(filtered.map(n => n.id));

  const flowNodes: Node[] = filtered.map(n => {
    const sel = n.id === selectedId;
    const conn = selectedId ? connected.has(n.id) : false;
    const dim = selectedId ? !connected.has(n.id) : false;
    const exp = expandedIds.has(n.id);

    if (n.kind === 'file') {
      const syms = symbolsOf[n.filePath] || [];
      return {
        id: n.id, position: { x: 0, y: 0 }, type: 'fileNode',
        data: {
          label: n.name, language: n.language, symbolCount: syms.length,
          expanded: exp, symbols: exp ? syms.slice(0, 25).map(s => ({ name: s.name, kind: s.kind, id: s.id })) : [],
          isSelected: sel, isConnected: conn, isDimmed: dim,
        },
      };
    }
    if (n.kind === 'class') {
      const methods = methodsOf[n.id] || [];
      return {
        id: n.id, position: { x: 0, y: 0 }, type: 'classNode',
        data: {
          label: n.name, methodCount: methods.length,
          filePath: n.filePath?.split('/').pop() || '',
          expanded: exp, methods: exp ? methods.slice(0, 25).map(m => ({ name: m.name, signature: m.signature, id: m.id })) : [],
          isSelected: sel, isConnected: conn, isDimmed: dim,
        },
      };
    }
    return {
      id: n.id, position: { x: 0, y: 0 }, type: 'functionNode',
      data: { label: n.name, signature: n.signature, isSelected: sel, isConnected: conn, isDimmed: dim },
    };
  });

  const fEdges = rawEdges.filter(e => edgeKinds.has(e.kind) && visibleIds.has(e.sourceId) && visibleIds.has(e.targetId));
  const flowEdges: Edge[] = fEdges.map((e, i) => {
    const hi = selectedId ? (connected.has(e.sourceId) && connected.has(e.targetId)) : true;
    const s = edgeColor(e.kind, hi);
    return {
      id: `e-${i}`, source: e.sourceId, target: e.targetId,
      style: { stroke: s.stroke, strokeWidth: hi ? 2 : 1, strokeDasharray: s.dash },
      markerEnd: { type: MarkerType.ArrowClosed, color: s.stroke },
      label: hi && selectedId ? e.kind : undefined,
      labelStyle: { fontSize: 9, fill: '#999' },
    };
  });

  const dir = viewMode === 'files' ? 'TB' : 'LR';
  return { nodes: computeLayout(flowNodes, flowEdges, dir), edges: flowEdges, truncated };
}

// ============================================================================
// Detail Panel
// ============================================================================

function DetailPanel({ info, onClose }: { info: SelectedInfo; onClose: () => void }) {
  const n = info.node;
  return (
    <div className="absolute right-3 top-3 bottom-3 w-72 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-gray-100 dark:border-gray-800 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span>{n.kind === 'file' ? '📄' : n.kind === 'class' ? '🏛️' : '⚡'}</span>
            <span className="text-sm font-bold truncate">{n.name}</span>
          </div>
          <div className="text-[10px] text-gray-400 mt-0.5">{n.kind} · {n.language || ''}</div>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none mt-0.5">×</button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 text-xs space-y-4">
        <div>
          <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">Location</div>
          <div className="font-mono text-[11px] text-gray-600 dark:text-gray-300 break-all bg-gray-50 dark:bg-gray-800 rounded p-1.5">{n.filePath}</div>
          {n.startLine && <div className="text-gray-400 mt-0.5 text-[10px]">Lines {n.startLine}–{n.endLine}</div>}
        </div>
        {n.signature && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">Signature</div>
            <div className="font-mono text-[11px] bg-gray-50 dark:bg-gray-800 rounded p-1.5 break-all">{n.signature}</div>
          </div>
        )}
        {info.children.length > 0 && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">
              {n.kind === 'class' ? 'Methods' : 'Contains'} ({info.children.length})
            </div>
            <div className="space-y-0.5 max-h-44 overflow-y-auto">
              {info.children.map(c => (
                <div key={c.id} className="flex items-center gap-1.5 py-0.5 text-[11px]">
                  <span className="flex-shrink-0">{c.kind === 'class' ? '🏛️' : c.kind === 'method' ? '🔧' : '⚡'}</span>
                  <span className="truncate">{c.signature || c.name}</span>
                  {c.startLine && <span className="text-gray-400 ml-auto text-[10px] flex-shrink-0">L{c.startLine}</span>}
                </div>
              ))}
            </div>
          </div>
        )}
        {info.incomingEdges.length > 0 && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">← Incoming ({info.incomingEdges.length})</div>
            <div className="space-y-0.5 max-h-36 overflow-y-auto">
              {info.incomingEdges.map(({ edge, sourceNode }, i) => (
                <div key={i} className="flex items-center gap-1.5 text-[11px] py-0.5">
                  <span className={`text-[9px] px-1 rounded ${edge.kind === 'imports' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600' : edge.kind === 'inherits' ? 'bg-green-100 dark:bg-green-900/30 text-green-600' : 'bg-gray-100 dark:bg-gray-800 text-gray-500'}`}>{edge.kind}</span>
                  <span className="truncate">{sourceNode?.name || '?'}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {info.outgoingEdges.length > 0 && (
          <div>
            <div className="text-[10px] text-gray-400 uppercase font-semibold mb-1">→ Outgoing ({info.outgoingEdges.length})</div>
            <div className="space-y-0.5 max-h-36 overflow-y-auto">
              {info.outgoingEdges.map(({ edge, targetNode }, i) => (
                <div key={i} className="flex items-center gap-1.5 text-[11px] py-0.5">
                  <span className={`text-[9px] px-1 rounded ${edge.kind === 'imports' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600' : edge.kind === 'calls' ? 'bg-gray-100 dark:bg-gray-800 text-gray-500' : 'bg-green-100 dark:bg-green-900/30 text-green-600'}`}>{edge.kind}</span>
                  <span className="truncate">{targetNode?.name || '?'}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Inner Component
// ============================================================================

interface CodeMapProps {
  data: CodeMapData;
  onNavigateToFile?: (filePath: string) => void;
}

function CodeMapInner({ data, onNavigateToFile }: CodeMapProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('files');
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLayouting, setIsLayouting] = useState(false);
  const { fitView } = useReactFlow();

  React.useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(t);
  }, [searchQuery]);

  React.useEffect(() => {
    setExpandedIds(new Set());
    setSelectedId(null);
  }, [viewMode]);

  const { nodes: lnodes, edges: ledges, truncated } = useMemo(
    () => buildGraph(data, viewMode, debouncedQuery, expandedIds, selectedId),
    [data, viewMode, debouncedQuery, expandedIds, selectedId]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  React.useEffect(() => {
    setIsLayouting(true);
    const frame = requestAnimationFrame(() => {
      setNodes(lnodes);
      setEdges(ledges);
      setIsLayouting(false);
      setTimeout(() => fitView({ padding: 0.15, duration: 200 }), 50);
    });
    return () => cancelAnimationFrame(frame);
  }, [lnodes, ledges, setNodes, setEdges, fitView]);

  const selectedInfo = useMemo<SelectedInfo | null>(() => {
    if (!selectedId) return null;
    const map = new Map(data.nodes.map(n => [n.id, n]));
    const node = map.get(selectedId);
    if (!node) return null;
    const children = data.nodes.filter(n =>
      n.parentId === selectedId || (node.kind === 'file' && n.filePath === node.filePath && n.kind !== 'file')
    );
    return {
      node, children,
      incomingEdges: data.edges.filter(e => e.targetId === selectedId).map(e => ({ edge: e, sourceNode: map.get(e.sourceId) })),
      outgoingEdges: data.edges.filter(e => e.sourceId === selectedId).map(e => ({ edge: e, targetNode: map.get(e.targetId) })),
    };
  }, [selectedId, data]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    if (selectedId === node.id) {
      setExpandedIds(prev => {
        const next = new Set(prev);
        next.has(node.id) ? next.delete(node.id) : next.add(node.id);
        return next;
      });
    } else {
      setSelectedId(node.id);
    }
  }, [selectedId]);

  const onPaneClick = useCallback(() => setSelectedId(null), []);

  const stats = data.metadata;

  return (
    <div className="relative" style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick} onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView fitViewOptions={{ padding: 0.15 }}
        minZoom={0.02} maxZoom={2.5}
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap pannable zoomable style={{ height: 90, width: 130 }} />

        <Panel position="top-left">
          <div className="flex flex-col gap-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-2.5 shadow-lg" style={{ maxWidth: 210 }}>
            <div className="flex gap-1">
              {(['files', 'symbols', 'classes'] as ViewMode[]).map(m => (
                <button key={m} onClick={() => setViewMode(m)}
                  className={`px-2 py-1 text-[11px] rounded transition-colors ${viewMode === m ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-700'}`}>
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
            <input type="text" placeholder="Search..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
              className="px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 outline-none focus:border-blue-400" />
            {truncated && <div className="text-[10px] text-amber-600">Showing {MAX_NODES} of {data.nodes.length}. Search to filter.</div>}
            {isLayouting && <div className="text-[10px] text-blue-500 animate-pulse">Computing layout...</div>}
            <div className="text-[10px] text-gray-400 leading-tight">Click: select & highlight connections<br/>Click again: expand node<br/>Click canvas: deselect</div>
          </div>
        </Panel>

        {!selectedInfo && (
          <Panel position="top-right">
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-2 shadow-lg text-xs text-gray-500">
              <div>{stats?.totalFiles ?? 0} files · {stats?.totalSymbols ?? 0} symbols · {stats?.totalEdges ?? 0} edges</div>
              {stats?.languageStats && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {Object.entries(stats.languageStats).map(([l, c]) => (
                    <span key={l} className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-[10px]">{l}: {c as number}</span>
                  ))}
                </div>
              )}
            </div>
          </Panel>
        )}

        <Panel position="bottom-left">
          <div className="bg-white/90 dark:bg-gray-900/90 border border-gray-200 dark:border-gray-700 rounded px-2 py-1.5 text-[10px] text-gray-500 flex gap-3">
            <span><span className="inline-block w-3 border-t-2 border-dashed border-blue-400 mr-1" />imports</span>
            <span><span className="inline-block w-3 border-t-2 border-gray-400 mr-1" />calls</span>
            <span><span className="inline-block w-3 border-t-2 border-green-500 mr-1" />inherits</span>
          </div>
        </Panel>
      </ReactFlow>

      {selectedInfo && <DetailPanel info={selectedInfo} onClose={() => setSelectedId(null)} />}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function CodeMap({ data, onNavigateToFile }: CodeMapProps) {
  return (
    <div style={{ width: '100%', height: 'calc(100vh - 220px)', minHeight: '500px' }}>
      <ReactFlowProvider>
        <CodeMapInner data={data} onNavigateToFile={onNavigateToFile} />
      </ReactFlowProvider>
    </div>
  );
}
