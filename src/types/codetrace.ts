export interface CodeReference {
  refId: string;
  filePath: string;
  startLine: number;
  endLine: number;
  snippet: string;
  annotation: string;
}

export interface CodeTraceSection {
  id: string;
  title: string;
  motivation: string;
  details: string;
  codeRefs: CodeReference[];
  connections: string[];
}

export interface CodeTraceResult {
  query: string;
  title: string;
  sections: CodeTraceSection[];
  sourceFiles: string[];
  sourceContents: Record<string, SourceChunk[]>;
  generatedAt?: string;
}

export interface SourceChunk {
  filePath: string;
  startLine: number;
  endLine: number;
  content: string;
  language: string;
}
