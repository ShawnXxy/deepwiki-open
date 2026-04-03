export interface CodeMapNode {
  id: string;
  name: string;
  kind: 'file' | 'class' | 'function' | 'method' | 'module';
  filePath: string;
  startLine?: number;
  endLine?: number;
  parentId?: string;
  language?: string;
  signature?: string;
}

export interface CodeMapEdge {
  sourceId: string;
  targetId: string;
  kind: 'imports' | 'calls' | 'inherits' | 'implements';
}

export interface CodeMapMetadata {
  owner: string;
  repo: string;
  repoType: string;
  branch?: string;
  commitHash?: string;
  generatedAt?: string;
  totalFiles: number;
  totalSymbols: number;
  totalEdges: number;
  languageStats: Record<string, number>;
}

export interface CodeMapData {
  nodes: CodeMapNode[];
  edges: CodeMapEdge[];
  metadata: CodeMapMetadata;
}
