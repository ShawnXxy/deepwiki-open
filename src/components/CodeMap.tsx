'use client';

import React, { useMemo, useState, useCallback, useRef } from 'react';
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
  useOnViewportChange,
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
        {data.expanded && <span className="ml-1 text-blue-500">(expanded)</span>}
      </div>
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
        {data.expanded && <span className="ml-1 text-purple-500">(expanded)</span>}
      </div>
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

  // Separate connected and disconnected nodes
  const connectedNodeIds = new Set<string>();
  flowEdges.forEach(e => {
    connectedNodeIds.add(e.source);
    connectedNodeIds.add(e.target);
  });

  const connectedNodes = flowNodes.filter(n => connectedNodeIds.has(n.id));
  const disconnectedNodes = flowNodes.filter(n => !connectedNodeIds.has(n.id));

  // Layout connected nodes with dagre
  const positioned: Node[] = [];
  if (connectedNodes.length > 0) {
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: direction, nodesep: 80, ranksep: 100, marginx: 30, marginy: 30 });
    g.setDefaultEdgeLabel(() => ({}));
    connectedNodes.forEach(node => {
      const h = node.type === 'functionNode' ? 42 : 55;
      const w = node.type === 'functionNode' ? 170 : 230;
      g.setNode(node.id, { width: w, height: h });
    });
    flowEdges.forEach(e => { if (g.hasNode(e.source) && g.hasNode(e.target)) g.setEdge(e.source, e.target); });
    dagre.layout(g);
    connectedNodes.forEach(node => {
      const p = g.node(node.id);
      positioned.push(p
        ? { ...node, position: { x: p.x - (p.width || 0) / 2, y: p.y - (p.height || 0) / 2 } }
        : node
      );
    });
  }

  // Layout disconnected nodes in a grid below the dagre layout
  if (disconnectedNodes.length > 0) {
    const cols = Math.ceil(Math.sqrt(disconnectedNodes.length));
    const cellW = 250;
    const cellH = 75;
    // Find the bottom of the dagre layout
    let offsetY = 0;
    positioned.forEach(n => { offsetY = Math.max(offsetY, n.position.y + 80); });
    if (positioned.length > 0) offsetY += 60; // gap

    disconnectedNodes.forEach((node, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      positioned.push({
        ...node,
        position: { x: col * cellW, y: offsetY + row * cellH },
      });
    });
  }

  return positioned;
}

/**
 * Incremental layout: reuse existing positions for nodes already on screen,
 * only position truly new nodes near their neighbours.
 * Avoids the full dagre re-layout that shifts everything on LOD changes.
 */
function incrementalLayout(
  flowNodes: Node[], flowEdges: Edge[],
  prevPositions: Map<string, { x: number; y: number }>,
): Node[] {
  if (flowNodes.length === 0) return [];

  const positioned: Node[] = [];
  const newNodes: Node[] = [];

  for (const node of flowNodes) {
    const prev = prevPositions.get(node.id);
    if (prev) {
      positioned.push({ ...node, position: { ...prev } });
    } else {
      newNodes.push(node);
    }
  }

  if (newNodes.length > 0) {
    const posMap = new Map(positioned.map(n => [n.id, n.position]));
    const edgeIndex = new Map<string, string[]>();
    flowEdges.forEach(e => {
      if (!edgeIndex.has(e.source)) edgeIndex.set(e.source, []);
      edgeIndex.get(e.source)!.push(e.target);
      if (!edgeIndex.has(e.target)) edgeIndex.set(e.target, []);
      edgeIndex.get(e.target)!.push(e.source);
    });

    let gridIdx = 0;
    let maxY = 0;
    positioned.forEach(n => { maxY = Math.max(maxY, n.position.y + 80); });
    const gridTop = maxY + 60;
    const cols = Math.max(Math.ceil(Math.sqrt(newNodes.length)), 1);

    for (const node of newNodes) {
      const neighbours = edgeIndex.get(node.id) || [];
      let placed = false;
      for (const nid of neighbours) {
        const np = posMap.get(nid);
        if (np) {
          const jitter = (Math.random() - 0.5) * 120;
          const pos = { x: np.x + 200 + jitter, y: np.y + jitter };
          positioned.push({ ...node, position: pos });
          posMap.set(node.id, pos);
          placed = true;
          break;
        }
      }
      if (!placed) {
        const col = gridIdx % cols;
        const row = Math.floor(gridIdx / cols);
        const pos = { x: col * 250, y: gridTop + row * 75 };
        positioned.push({ ...node, position: pos });
        posMap.set(node.id, pos);
        gridIdx++;
      }
    }
  }

  return positioned;
}

// ============================================================================
// LOD (Level-of-Detail) tiers — map-like zoom behaviour
// ============================================================================

interface LodTier { label: string; maxNodes: number; }
const LOD_TIERS: { zoomThreshold: number; tier: LodTier }[] = [
  { zoomThreshold: 0.15, tier: { label: 'Overview',       maxNodes: 30  } },
  { zoomThreshold: 0.50, tier: { label: 'Intermediate',   maxNodes: 150 } },
  { zoomThreshold: Infinity, tier: { label: 'Detail',     maxNodes: 400 } },
];

function getLodTier(zoom: number): LodTier {
  for (const t of LOD_TIERS) {
    if (zoom < t.zoomThreshold) return t.tier;
  }
  return LOD_TIERS[LOD_TIERS.length - 1].tier;
}

// ============================================================================
// Graph builder
// ============================================================================

function buildGraph(
  data: CodeMapData, viewMode: ViewMode, query: string,
  expandedIds: Set<string>, selectedId: string | null,
  maxNodes: number,
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

  // LOD: when importance scores are available, sort by score so the most
  // critical nodes survive truncation at lower zoom levels.
  const hasScores = filtered.some(n => (n.importanceScore ?? 0) > 0);
  if (hasScores) {
    filtered.sort((a, b) => (b.importanceScore ?? 0) - (a.importanceScore ?? 0));
  }

  const budget = q ? 400 : maxNodes;           // search overrides LOD
  const truncated = filtered.length > budget;
  if (truncated) filtered = filtered.slice(0, budget);

  // When expanded, inject children as real graph nodes
  const expandedChildren: CodeMapNode[] = [];
  const existingIds = new Set(filtered.map(n => n.id));
  expandedIds.forEach(parentId => {
    const parentNode = raw.find(n => n.id === parentId);
    if (!parentNode) return;
    const children = parentNode.kind === 'file'
      ? (symbolsOf[parentNode.filePath] || []).slice(0, 30)
      : parentNode.kind === 'class'
        ? (methodsOf[parentId] || []).slice(0, 30)
        : [];
    children.forEach(c => {
      if (!existingIds.has(c.id)) {
        expandedChildren.push(c);
        existingIds.add(c.id);
      }
    });
  });

  const allVisible = [...filtered, ...expandedChildren];
  const visibleIds = new Set(allVisible.map(n => n.id));
  const expandedChildIds = new Set(expandedChildren.map(n => n.id));

  // Show call edges for expanded children
  const allEdgeKinds = new Set(edgeKinds);
  if (expandedIds.size > 0) allEdgeKinds.add('calls');

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

  // Build flow nodes — no more text lists, just simple boxes
  const flowNodes: Node[] = allVisible.map(n => {
    const sel = n.id === selectedId;
    const conn = selectedId ? connected.has(n.id) : false;
    const dim = selectedId ? !connected.has(n.id) : false;
    const isExpanded = expandedIds.has(n.id);
    const isChild = expandedChildIds.has(n.id);

    if (n.kind === 'file') {
      return {
        id: n.id, position: { x: 0, y: 0 }, type: 'fileNode',
        data: {
          label: n.name, language: n.language,
          symbolCount: (symbolsOf[n.filePath] || []).length,
          expanded: isExpanded,
          isSelected: sel, isConnected: conn, isDimmed: dim,
        },
      };
    }
    if (n.kind === 'class') {
      return {
        id: n.id, position: { x: 0, y: 0 }, type: 'classNode',
        data: {
          label: n.name,
          methodCount: (methodsOf[n.id] || []).length,
          filePath: n.filePath?.split('/').pop() || '',
          expanded: isExpanded, isChild,
          isSelected: sel, isConnected: conn, isDimmed: dim,
        },
      };
    }
    return {
      id: n.id, position: { x: 0, y: 0 }, type: 'functionNode',
      data: {
        label: n.name, signature: n.signature,
        isChild,
        isSelected: sel, isConnected: conn, isDimmed: dim,
      },
    };
  });

  // Real edges between visible nodes
  const realEdges = rawEdges.filter(e =>
    allEdgeKinds.has(e.kind) && visibleIds.has(e.sourceId) && visibleIds.has(e.targetId)
  );

  // Parent→child containment edges (animated dashed)
  const containsEdges: { sourceId: string; targetId: string; kind: string }[] = [];
  expandedIds.forEach(parentId => {
    const parentNode = raw.find(p => p.id === parentId);
    if (!parentNode) return;
    expandedChildren.forEach(c => {
      const isMyChild = c.parentId === parentId ||
        (parentNode.kind === 'file' && c.filePath === parentNode.filePath && c.parentId === undefined);
      if (isMyChild) {
        containsEdges.push({ sourceId: parentId, targetId: c.id, kind: 'contains' });
      }
    });
  });

  const allEdges = [...realEdges, ...containsEdges];
  const flowEdges: Edge[] = allEdges.map((e, i) => {
    if (e.kind === 'contains') {
      return {
        id: `e-${i}`, source: e.sourceId, target: e.targetId,
        style: { stroke: '#94a3b8', strokeWidth: 1.5, strokeDasharray: '4 2' },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
        animated: true,
      };
    }
    const hi = selectedId ? (connected.has(e.sourceId) && connected.has(e.targetId)) : true;
    const s = edgeColor(e.kind, hi);
    return {
      id: `e-${i}`, source: e.sourceId, target: e.targetId,
      style: { stroke: s.stroke, strokeWidth: hi ? 2.5 : 1, strokeDasharray: s.dash },
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
  const [viewMode, setViewMode] = useState<ViewMode>('classes');
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLayouting, setIsLayouting] = useState(false);
  const { fitView } = useReactFlow();

  // LOD zoom tracking — only react to USER-initiated zoom, not programmatic fitView.
  // isFitting suppresses viewport events during layout/fitView transitions to break
  // the loop: fitView→zoom change→tier change→rebuild→fitView.
  const [currentZoom, setCurrentZoom] = useState(1);
  const isFitting = useRef(false);
  const shouldFitView = useRef(true);
  const zoomTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useOnViewportChange({
    onChange: ({ zoom }) => {
      if (isFitting.current) return;
      if (zoomTimer.current) clearTimeout(zoomTimer.current);
      zoomTimer.current = setTimeout(() => {
        shouldFitView.current = false;         // LOD change from user zoom — don't reset viewport
        setCurrentZoom(zoom);
      }, 150);
    },
  });
  const lodTier = useMemo(() => getLodTier(currentZoom), [currentZoom]);

  React.useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(t);
  }, [searchQuery]);

  React.useEffect(() => {
    setExpandedIds(new Set());
    setSelectedId(null);
    shouldFitView.current = true;
  }, [viewMode]);

  const prevQuery = useRef(debouncedQuery);
  React.useEffect(() => {
    if (prevQuery.current !== debouncedQuery) {
      shouldFitView.current = true;
      prevQuery.current = debouncedQuery;
    }
  }, [debouncedQuery]);

  const { nodes: lnodes, edges: ledges, truncated } = useMemo(
    () => buildGraph(data, viewMode, debouncedQuery, expandedIds, selectedId, lodTier.maxNodes),
    [data, viewMode, debouncedQuery, expandedIds, selectedId, lodTier.maxNodes]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const prevNodePositions = useRef<Map<string, { x: number; y: number }>>(new Map());

  React.useEffect(() => {
    setIsLayouting(true);
    const doFit = shouldFitView.current;
    if (doFit) isFitting.current = true;
    const frame = requestAnimationFrame(() => {
      // For LOD zoom changes (doFit=false), use incremental layout to keep positions stable.
      // For intentional changes (view switch, search, initial load), do full dagre layout.
      let finalNodes: Node[];
      if (!doFit && prevNodePositions.current.size > 0) {
        finalNodes = incrementalLayout(lnodes, ledges, prevNodePositions.current);
      } else {
        finalNodes = lnodes;
      }
      setNodes(finalNodes);
      setEdges(ledges);
      prevNodePositions.current = new Map(finalNodes.map(n => [n.id, n.position]));
      setIsLayouting(false);
      if (doFit) {
        setTimeout(() => {
          fitView({ padding: 0.15, duration: 200 });
          setTimeout(() => { isFitting.current = false; }, 300);
        }, 50);
      }
      shouldFitView.current = true;
    });
    return () => cancelAnimationFrame(frame);
  }, [lnodes, ledges, setNodes, setEdges, fitView]);

  // Deselect if the selected node is no longer visible after LOD change
  React.useEffect(() => {
    if (selectedId && !lnodes.some(n => n.id === selectedId)) {
      setSelectedId(null);
    }
  }, [lnodes, selectedId]);

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
        fitView fitViewOptions={{ padding: 0.1, maxZoom: 1 }}
        minZoom={0.02} maxZoom={2.5}
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap pannable zoomable style={{ height: 90, width: 130 }} />

        <Panel position="top-left">
          <div className="flex flex-col gap-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-2.5 shadow-lg" style={{ maxWidth: 210 }}>
            <div className="flex gap-1">
              {(['classes', 'symbols', 'files'] as ViewMode[]).map(m => (
                <button key={m} onClick={() => setViewMode(m)}
                  className={`px-2 py-1 text-[11px] rounded transition-colors ${viewMode === m ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-700'}`}>
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
            <input type="text" placeholder="Search..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
              className="px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 outline-none focus:border-blue-400" />
            <div className="text-[10px] text-gray-400">{lodTier.label} · {lnodes.length} nodes</div>
            {truncated && <div className="text-[10px] text-amber-600">Showing {lodTier.maxNodes} of {data.nodes.length}. Zoom in or search to see more.</div>}
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
            <span><span className="inline-block w-3 border-t-2 border-dashed border-gray-400 mr-1" />contains</span>
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
